# type: ignore
"""
bronze_ingestion.py — Glue job: download a Kaggle CSV and upload to S3.

Usage:
    spark-submit bronze_ingestion.py <execution_date> [file_name] [dataset_type]

Arguments:
    execution_date  Airflow logical date ({{ ds }})
    file_name       CSV inside the Kaggle zip   (default: cs-training.csv)
    dataset_type    "training" | "inference"     (default: training)

S3 layout:
    training  → bronze/credit_risk/kaggle/{date}/{file_name}
    inference → bronze/credit_risk/kaggle/{date}/inference/{file_name}
"""

import logging
import os
import sys

import boto3  # type: ignore
from src.ingestion.kaggle_extractor import (
    KaggleDataExtractor,
    KaggleExtractionConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("glue_local_ingestion")

VALID_DATASET_TYPES = {"training", "inference"}

try:
    execution_date = sys.argv[1]  # passed from Airflow {{ ds }}
except IndexError:
    logger.error("Execution date argument is missing.")
    sys.exit(1)

FILE_NAME = sys.argv[2] if len(sys.argv) > 2 else "cs-training.csv"
DATASET_TYPE = sys.argv[3] if len(sys.argv) > 3 else "training"

if DATASET_TYPE not in VALID_DATASET_TYPES:
    logger.error(
        f"Invalid dataset_type '{DATASET_TYPE}'. Must be one of {VALID_DATASET_TYPES}."
    )
    sys.exit(1)

LOCAL_PATH = "/tmp"
BUCKET = "data-lake"

# Training keeps original path for backward compatibility;
# inference gets its own sub-directory.
BRONZE_KEY_TEMPLATES = {
    "training": "bronze/credit_risk/kaggle/{date}/{file}",
    "inference": "bronze/credit_risk/kaggle/{date}/inference/{file}",
}
key = BRONZE_KEY_TEMPLATES[DATASET_TYPE].format(
    date=execution_date, file=FILE_NAME
)

logger.info(
    f"Starting ingestion for execution date: {execution_date} "
    f"(file={FILE_NAME}, type={DATASET_TYPE})"
)

extractor = KaggleDataExtractor(
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
)

logger.info(f"Downloading {FILE_NAME} from Kaggle...")
extractor.download_dataset(
    KaggleExtractionConfig(
        dataset_slug="brycecf/give-me-some-credit-dataset",
        file_name=FILE_NAME,
        destination_path=LOCAL_PATH,
    )
)

s3_endpoint = os.getenv("AWS_ENDPOINT_URL")

logger.info(f"Uploading file to S3 (Endpoint: {s3_endpoint})...")
s3 = boto3.client("s3", endpoint_url=s3_endpoint)

try:
    s3.upload_file(f"{LOCAL_PATH}/{FILE_NAME}", BUCKET, key)

    obj = s3.head_object(Bucket=BUCKET, Key=key)

    if obj["ContentLength"] <= 0:
        logger.error(f"Upload failed: File at s3://{BUCKET}/{key} is empty.")
        raise ValueError("Uploaded file is empty")

    logger.info(
        f"Bronze OK s3://{BUCKET}/{key} ({obj['ContentLength']} bytes)"
    )

except Exception as e:
    logger.exception(f"An error occurred during the S3 transfer: {str(e)}")
    sys.exit(1)
