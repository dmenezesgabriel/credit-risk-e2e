import os
import sys
from datetime import datetime

import boto3
from pyspark.sql import SparkSession

from src.ingestion.kaggle_extractor import (
    KaggleDataExtractor,
    KaggleExtractionConfig,
)

spark = SparkSession.builder.appName("bronze_ingestion").getOrCreate()

LOCAL_PATH = "/tmp"
FILE_NAME = "cs-training.csv"

BUCKET = "data-lake"

AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")


def upload_to_s3(file_path, key):
    s3 = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL if AWS_ENDPOINT_URL else None,
    )

    s3.upload_file(file_path, BUCKET, key)


def main():
    username = os.environ["KAGGLE_USERNAME"]
    token = os.environ["KAGGLE_KEY"]

    extractor = KaggleDataExtractor(username, token)

    config = KaggleExtractionConfig(
        dataset_slug="brycecf/give-me-some-credit-dataset/",
        file_name=FILE_NAME,
        destination_path=LOCAL_PATH,
    )

    extractor.download_dataset(config=config)

    today = datetime.now().strftime("%Y-%m-%d")

    key = f"bronze/credit_risk/kaggle/{today}/{FILE_NAME}"

    upload_to_s3(f"{LOCAL_PATH}/{FILE_NAME}", key)

    print(f"Uploaded to s3://{BUCKET}/{key}")


if __name__ == "__main__":
    main()
