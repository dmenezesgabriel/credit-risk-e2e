import logging
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import FloatType, IntegerType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("silver_cleaning")

spark = SparkSession.builder.appName("silver_cleaning").getOrCreate()

try:
    execution_date = sys.argv[1]
except IndexError:
    logger.error("Missing execution_date argument.")
    sys.exit(1)

BUCKET = "data-lake"
FILE_NAME = "cs-training.csv"
bronze_path = (
    f"s3a://{BUCKET}/bronze/credit_risk/kaggle/{execution_date}/{FILE_NAME}"
)
silver_path = f"s3a://{BUCKET}/silver/credit_risk/cleaned/"

RENAME_MAP = {
    "_c0": "index_to_drop",
    "SeriousDlqin2yrs": "serious_dlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization_of_unsecured_lines",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "number_of_time30_59_days_past_due_not_worse",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "number_of_open_credit_lines_and_loans",
    "NumberOfTimes90DaysLate": "number_of_times90_days_late",
    "NumberRealEstateLoansOrLines": "number_real_estate_loans_or_lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "number_of_time60_89_days_past_due_not_worse",
    "NumberOfDependents": "number_of_dependents",
}

TYPE_MAP = {
    "serious_dlqin2yrs": IntegerType(),
    "revolving_utilization_of_unsecured_lines": FloatType(),
    "age": IntegerType(),
    "number_of_time30_59_days_past_due_not_worse": IntegerType(),
    "debt_ratio": FloatType(),
    "monthly_income": FloatType(),
    "number_of_open_credit_lines_and_loans": IntegerType(),
    "number_of_times90_days_late": IntegerType(),
    "number_real_estate_loans_or_lines": IntegerType(),
    "number_of_time60_89_days_past_due_not_worse": IntegerType(),
    "number_of_dependents": FloatType(),
}

try:
    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .csv(bronze_path)
    )

    df = raw_df.toDF(*[RENAME_MAP.get(c, c) for c in raw_df.columns])

    df = df.drop("index_to_drop")

    df = df.dropDuplicates()

    for col_name, col_type in TYPE_MAP.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(col_type))
        else:
            logger.warning(
                f"Expected column '{col_name}' not found — skipping cast."
            )

    if "serious_dlqin2yrs" not in df.columns:
        raise ValueError(
            "Target column 'serious_dlqin2yrs' missing after rename."
        )

    row_count = df.count()
    if row_count < 100_000:
        raise ValueError(f"Row count too low after cleaning: {row_count}")

    df = df.withColumn("ingestion_date", lit(execution_date))

    df.write.mode("overwrite").partitionBy("ingestion_date").parquet(
        silver_path
    )

    logger.info(f"Silver write OK — {row_count} rows, date={execution_date}")

except Exception as e:
    logger.exception(
        "A fatal error occurred during the Silver cleaning Spark job."
    )
    sys.exit(1)
