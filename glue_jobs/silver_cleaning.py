import re
from datetime import datetime

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("silver_cleaning").getOrCreate()

BUCKET = "data-lake"

bronze_path = f"s3a://{BUCKET}/bronze/credit_risk/kaggle/"
silver_path = f"s3a://{BUCKET}/silver/credit_risk/cleaned/"


def normalize_column_name(name: str) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)

    name = name.lower()
    name = name.replace(" ", "_")
    name = name.replace("-", "_")

    return name


today = datetime.now().strftime("%Y-%m-%d")
bronze_path = f"s3a://{BUCKET}/bronze/credit_risk/kaggle/{today}/{FILE_NAME}"

df = spark.read.option("header", True).csv(bronze_path)


df = df.toDF(*[normalize_column_name(c) for c in df.columns])

df = df.dropDuplicates()

df.write.mode("overwrite").parquet(silver_path)
