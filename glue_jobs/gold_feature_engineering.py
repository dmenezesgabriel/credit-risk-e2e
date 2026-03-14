import logging
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    monotonically_increasing_id,
    to_timestamp,
    when,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gold_feature_engineering")

spark = SparkSession.builder.appName("gold_feature_engineering").getOrCreate()

try:
    execution_date = sys.argv[1]  # passed from Airflow {{ ds }}
except IndexError:
    logger.error("Missing execution_date argument.")
    sys.exit(1)

BUCKET = "data-lake"
silver_path = f"s3a://{BUCKET}/silver/credit_risk/cleaned/ingestion_date={execution_date}/"
gold_path = f"s3a://{BUCKET}/gold/credit_risk/features/"

# ---------------------------------------------------------------------------
# EDA decisions hardcoded here (documented in notebooks/01_eda.ipynb)
# ---------------------------------------------------------------------------
# Outlier caps — derived from train p99, applied universally at feature eng.
# stage because these are business-logic caps, not statistical ones.
# revolving_utilization > 1.0 is physically impossible (>100% utilisation).
# debt_ratio > 4985 and monthly_income > 25000 are extreme data errors.
REVOLVING_CAP = 1.0  # hard business-logic ceiling
DEBT_RATIO_CAP = 4985.1  # p99 from EDA
MONTHLY_INCOME_CAP = 25000.0  # p99 from EDA
AGE_MIN = 18  # legal minimum — 1 row removed

logger.info(f"Starting gold feature engineering for date: {execution_date}")

try:
    df = spark.read.parquet(silver_path)
    row_count_in = df.count()
    logger.info(f"Silver rows read: {row_count_in}")

    # -----------------------------------------------------------------------
    # Step 1 — Quality filter (EDA finding: 1 row with age=0)
    # -----------------------------------------------------------------------
    df = df.filter(col("age") >= AGE_MIN)
    logger.info(f"Rows after age filter (>={AGE_MIN}): {df.count()}")

    # -----------------------------------------------------------------------
    # Step 2 — Synthesise entity_id and event_timestamp
    # Required by Feast. customer_id is stable across runs because we use
    # monotonically_increasing_id on a deterministically ordered dataset.
    # In production this would be a real customer surrogate key from CRM.
    # -----------------------------------------------------------------------
    df = df.withColumn(
        "customer_id", monotonically_increasing_id().cast("long")
    ).withColumn(
        "event_timestamp",
        to_timestamp(lit(execution_date), "yyyy-MM-dd"),
    )

    # -----------------------------------------------------------------------
    # Step 3 — Outlier capping
    # These are deterministic transforms — no fitting needed.
    # revolving_utilization: >1.0 is a data error (>100% utilisation)
    # debt_ratio + monthly_income: capped at p99 from EDA
    # -----------------------------------------------------------------------
    df = (
        df.withColumn(
            "revolving_utilization_of_unsecured_lines",
            when(
                col("revolving_utilization_of_unsecured_lines")
                > REVOLVING_CAP,
                REVOLVING_CAP,
            ).otherwise(col("revolving_utilization_of_unsecured_lines")),
        )
        .withColumn(
            "debt_ratio",
            when(col("debt_ratio") > DEBT_RATIO_CAP, DEBT_RATIO_CAP).otherwise(
                col("debt_ratio")
            ),
        )
        .withColumn(
            "monthly_income",
            when(
                col("monthly_income") > MONTHLY_INCOME_CAP, MONTHLY_INCOME_CAP
            ).otherwise(col("monthly_income")),
        )
    )

    # -----------------------------------------------------------------------
    # Step 4 — Missing value indicators
    # EDA finding: missingness is INFORMATIVE for both features.
    # monthly_income missing → default rate 5.66% vs 6.95% present.
    # number_of_dependents missing → default rate 4.65% vs 6.75% present.
    # These binary flags capture that signal BEFORE imputation erases it.
    # -----------------------------------------------------------------------
    df = df.withColumn(
        "monthly_income_is_missing",
        when(col("monthly_income").isNull(), 1).otherwise(0),
    ).withColumn(
        "number_of_dependents_is_missing",
        when(col("number_of_dependents").isNull(), 1).otherwise(0),
    )

    # -----------------------------------------------------------------------
    # Step 5 — Feature engineering
    # All transforms below are stateless (no fitting required).
    # They can be recomputed identically at inference time via Feast online store.
    # -----------------------------------------------------------------------

    # 5a. Delinquency composite score
    # EDA finding: 30-59, 60-89, 90-days features have 0.98-0.99 pairwise
    # correlation — multicollinear for LogReg. Weighted composite collapses
    # this while preserving severity ordering. Used alongside raw features
    # for tree models, replaces them for LogReg.
    df = df.withColumn(
        "delinquency_score",
        (
            col("number_of_time30_59_days_past_due_not_worse") * 1
            + col("number_of_time60_89_days_past_due_not_worse") * 2
            + col("number_of_times90_days_late") * 3
        ).cast("float"),
    )

    # 5b. Financial stress ratios
    # Ratio features tend to be strong predictors in credit scoring because
    # they normalise absolute values across income levels.
    df = df.withColumn(
        "debt_to_income_ratio",
        when(
            col("monthly_income").isNotNull() & (col("monthly_income") > 0),
            (col("debt_ratio") / col("monthly_income")).cast("float"),
        ).otherwise(lit(None).cast("float")),
    )

    # 5c. Credit line utilisation density
    # Open lines relative to real-estate exposure — measures credit breadth.
    df = df.withColumn(
        "unsecured_to_total_lines_ratio",
        when(
            col("number_of_open_credit_lines_and_loans") > 0,
            (
                col("number_of_open_credit_lines_and_loans")
                - col("number_real_estate_loans_or_lines")
            ).cast("float")
            / col("number_of_open_credit_lines_and_loans").cast("float"),
        ).otherwise(lit(0.0).cast("float")),
    )

    # 5d. Age risk bucket
    # EDA decile chart shows non-linear age effect — young and very old
    # borrowers have different risk profiles. Buckets capture this for
    # models that benefit from explicit non-linearity (e.g. LogReg).
    df = df.withColumn(
        "age_risk_bucket",
        when(col("age") < 30, lit("young"))
        .when(col("age") < 50, lit("middle"))
        .when(col("age") < 65, lit("senior"))
        .otherwise(lit("elderly")),
    )

    # 5e. Any delinquency flag (binary)
    # Simple but often one of the top SHAP features in practice.
    df = df.withColumn(
        "has_any_delinquency",
        when(col("delinquency_score") > 0, 1).otherwise(0),
    )

    # -----------------------------------------------------------------------
    # Step 6 — Select and order final feature set
    # Keep target (serious_dlqin2yrs) in gold for Feast label association.
    # -----------------------------------------------------------------------
    GOLD_COLUMNS = [
        # identifiers (required by Feast)
        "customer_id",
        "event_timestamp",
        # target
        "serious_dlqin2yrs",
        # original features (capped)
        "revolving_utilization_of_unsecured_lines",
        "age",
        "number_of_time30_59_days_past_due_not_worse",
        "debt_ratio",
        "monthly_income",
        "number_of_open_credit_lines_and_loans",
        "number_of_times90_days_late",
        "number_real_estate_loans_or_lines",
        "number_of_time60_89_days_past_due_not_worse",
        "number_of_dependents",
        # engineered features
        "monthly_income_is_missing",
        "number_of_dependents_is_missing",
        "delinquency_score",
        "debt_to_income_ratio",
        "unsecured_to_total_lines_ratio",
        "age_risk_bucket",
        "has_any_delinquency",
    ]

    df = df.select(*GOLD_COLUMNS)

    # -----------------------------------------------------------------------
    # Step 7 — Quality gate before write
    # -----------------------------------------------------------------------
    row_count_out = df.count()

    if row_count_out < 100_000:
        raise ValueError(
            f"Gold row count too low: {row_count_out}. Pipeline aborted."
        )

    null_customer = df.filter(col("customer_id").isNull()).count()
    if null_customer > 0:
        raise ValueError(f"{null_customer} rows with null customer_id.")

    logger.info(f"Gold rows to write: {row_count_out}")
    logger.info(
        f"Engineered features: {len(GOLD_COLUMNS) - 3} (excl. id, timestamp, target)"
    )

    # -----------------------------------------------------------------------
    # Step 8 — Write gold layer (Feast offline store format)
    # Partitioned by ingestion_date so Feast can efficiently scan time ranges.
    # mode=overwrite on the partition only — idempotent reruns.
    # -----------------------------------------------------------------------
    df.withColumn("ingestion_date", lit(execution_date)).write.mode(
        "overwrite"
    ).partitionBy("ingestion_date").parquet(gold_path)

    logger.info(
        f"Gold write OK — {row_count_out} rows → s3://{BUCKET}/gold/credit_risk/features/"
    )
    logger.info("Feature engineering complete. Ready for Feast registration.")

except Exception:
    logger.exception("Fatal error in gold feature engineering job.")
    sys.exit(1)
