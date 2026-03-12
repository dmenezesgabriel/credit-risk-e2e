#!/bin/sh
set -e

echo "Creating Data Lake S3 bucket..."

aws --endpoint-url=http://localhost:4566 \
    s3 mb s3://data-lake || true

echo "Creating MLflow S3 bucket..."

aws --endpoint-url=http://localhost:4566 \
    s3 mb s3://mlflow-artifacts || true

