# type: ignore
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


try:
    execution_date = sys.argv[1]  # passed from Airflow {{ ds }}
except IndexError:
    logger.error("Execution date argument is missing.")
    sys.exit(1)

LOCAL_PATH = "/tmp"
FILE_NAME = "cs-training.csv"
BUCKET = "data-lake"

logger.info(f"Starting ingestion for execution date: {execution_date}")


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

key = f"bronze/credit_risk/kaggle/{execution_date}/{FILE_NAME}"
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
        f"Bronze OK — s3://{BUCKET}/{key} ({obj['ContentLength']} bytes)"
    )

except Exception as e:
    logger.exception(f"An error occurred during the S3 transfer: {str(e)}")
    sys.exit(1)
