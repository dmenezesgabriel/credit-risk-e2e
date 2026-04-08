#!/usr/bin/env python3
# /// script
# dependencies = [
#   "requests==2.33.1",
#   "boto3==1.42.45",
#   "pandas==2.3.2",
#   "pandas-stubs==2.3.2.250827",
#   "pyarrow==22.0.0",
#   "rich==14.3.3",
# ]
# ///


import argparse
import logging
import os
import tempfile
from typing import Optional

import boto3
import pandas as pd
import requests


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


DEFAULT_SAMPLE_SIZE: int = 10000
DEFAULT_S3_BUCKET: str = "data-lake"
DEFAULT_YEAR: int = 2009
DEFAULT_MONTH: int = 1


def build_nyc_url(year: int, month: int) -> str:
    return f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"


def build_output_filename(year: int, month: int) -> str:
    return f"yellow_tripdata_{year}-{month:02d}_sample.parquet"


def build_s3_key(year: int, month: int) -> str:
    return f"bronze/nyc_taxi_trip/yellow_tripdata_{year}-{month:02d}_sample.parquet"


def download_parquet(url: str) -> str:
    logger = logging.getLogger("bronze_ingestion.download")
    logger.info(f"Downloading: {url}")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", delete=False
        ) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name
    logger.info(f"Downloaded to: {temp_path}")
    return temp_path


def sample_dataframe(parquet_path: str, sample_size: int) -> pd.DataFrame:
    logger = logging.getLogger("bronze_ingestion.sample")
    logger.info(f"Reading Parquet and sampling %d rows...", sample_size)
    dataframe = pd.read_parquet(parquet_path, engine="pyarrow")
    if sample_size < len(dataframe):
        return dataframe.sample(n=sample_size, random_state=42)
    return dataframe


def save_parquet(dataframe: pd.DataFrame, output_path: str) -> str:
    logger = logging.getLogger("bronze_ingestion.save")
    dataframe.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Sample saved to: {output_path}")
    return output_path


def upload_to_s3(
    local_path: str, bucket: str, key: str, endpoint_url: Optional[str] = None
) -> None:
    logger = logging.getLogger("bronze_ingestion.s3")
    logger.info(f"Uploading to S3: s3://{bucket}/{key}")
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        logger.info(f"Bucket {bucket} does not exist, creating it.")
        s3.create_bucket(Bucket=bucket)
    s3.upload_file(local_path, bucket, key)
    logger.info("Upload complete.")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Bronze ingestion: NYC Yellow Taxi sample to S3 (Localstack, Parquet format)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="Year of data (e.g., 2009)",
    )
    parser.add_argument(
        "--month", type=int, default=DEFAULT_MONTH, help="Month of data (1-12)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of rows to sample",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=DEFAULT_S3_BUCKET,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--s3-key",
        type=str,
        default=None,
        help="S3 key (object path). If not set, auto-generated.",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        type=str,
        default=os.environ.get("S3_ENDPOINT_URL", "http://localhost:4566"),
        help="S3 endpoint URL (Localstack)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local output Parquet path. If not set, auto-generated.",
    )
    args = parser.parse_args()

    url: str = build_nyc_url(args.year, args.month)
    output_path: str = args.output or build_output_filename(
        args.year, args.month
    )
    s3_key: str = args.s3_key or build_s3_key(args.year, args.month)

    parquet_path: str = download_parquet(url)
    sampled_data: pd.DataFrame = sample_dataframe(
        parquet_path, args.sample_size
    )
    save_parquet(sampled_data, output_path)
    try:
        os.remove(parquet_path)
    except Exception:
        pass
    upload_to_s3(
        output_path, args.s3_bucket, s3_key, endpoint_url=args.s3_endpoint_url
    )


if __name__ == "__main__":
    main()
