# type: ignore
import logging
import sys

from pyspark.sql import SparkSession  # type: ignore
from pyspark.sql.functions import col, lit, when  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gold_feature_engineering")

spark = SparkSession.builder.appName("gold_feature_engineering").getOrCreate()

try:
    execution_date = sys.argv[1]
except IndexError:
    logger.error("Missing execution_date argument.")
    sys.exit(1)

BUCKET = "data-lake"
silver_path = f"s3a://{BUCKET}/silver/credit_risk/cleaned/ingestion_date={execution_date}/"
gold_path = f"s3a://{BUCKET}/gold/credit_risk/features/"

# EDA decisions — documented in notebooks/01_eda.ipynb
REVOLVING_CAP = (
    1.0  # hard business-logic ceiling (>100% utilisation impossible)
)
DEBT_RATIO_CAP = 4985.1  # p99 from EDA
MONTHLY_INCOME_CAP = 25000.0  # p99 from EDA
AGE_MIN = 18  # legal minimum — 1 row removed

logger.info(f"Starting gold feature engineering for date: {execution_date}")

try:
    df = spark.read.parquet(silver_path)
    logger.info(f"Silver rows read: {df.count()}")

    # Step 1 — Quality filter
    df = df.filter(col("age") >= AGE_MIN)
    logger.info(f"Rows after age filter: {df.count()}")

    # Step 2 — Outlier capping (deterministic, no fitting needed)
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

    # Step 3 — Missing value indicators (EDA: missingness is informative)
    df = df.withColumn(
        "monthly_income_is_missing",
        when(col("monthly_income").isNull(), 1).otherwise(0),
    ).withColumn(
        "number_of_dependents_is_missing",
        when(col("number_of_dependents").isNull(), 1).otherwise(0),
    )

    # Step 4 — Feature engineering (all stateless transforms)

    # 4a. Delinquency composite — collapses 0.98-0.99 multicollinearity for LogReg
    df = df.withColumn(
        "delinquency_score",
        (
            col("number_of_time30_59_days_past_due_not_worse") * 1
            + col("number_of_time60_89_days_past_due_not_worse") * 2
            + col("number_of_times90_days_late") * 3
        ).cast("float"),
    )

    # 4b. Debt-to-income ratio
    df = df.withColumn(
        "debt_to_income_ratio",
        when(
            col("monthly_income").isNotNull() & (col("monthly_income") > 0),
            (col("debt_ratio") / col("monthly_income")).cast("float"),
        ).otherwise(lit(None).cast("float")),
    )

    # 4c. Unsecured credit line density
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

    # 4d. Age risk bucket (non-linear age effect confirmed in EDA decile chart)
    df = df.withColumn(
        "age_risk_bucket",
        when(col("age") < 30, lit("young"))
        .when(col("age") < 50, lit("middle"))
        .when(col("age") < 65, lit("senior"))
        .otherwise(lit("elderly")),
    )

    # 4e. Any delinquency binary flag
    df = df.withColumn(
        "has_any_delinquency",
        when(col("delinquency_score") > 0, 1).otherwise(0),
    )

    # Step 5 — Select final feature set
    GOLD_COLUMNS = [
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

    # Step 6 — Quality gate
    row_count = df.count()
    if row_count < 100_000:
        raise ValueError(
            f"Gold row count too low: {row_count}. Pipeline aborted."
        )

    logger.info(f"Gold rows to write: {row_count}")
    logger.info(f"Features: {len(GOLD_COLUMNS) - 1} (excl. target)")

    # Step 7 — Write partitioned parquet
    # Training pipeline reads directly from this path — no feature store needed
    # for batch training on a static dataset without entity keys or timestamps.
    df.withColumn("ingestion_date", lit(execution_date)).write.mode(
        "overwrite"
    ).partitionBy("ingestion_date").parquet(gold_path)

    logger.info(
        f"Gold write OK — {row_count} rows → s3://{BUCKET}/gold/credit_risk/features/"
    )

except Exception:
    logger.exception("Fatal error in gold feature engineering job.")
    sys.exit(1)
