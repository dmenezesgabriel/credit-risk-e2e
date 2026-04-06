#!/bin/sh
set -e

# Path inside the persistent volume
BACKUP_PATH="/var/lib/localstack/s3_backups"

echo "================================================"
echo "LocalStack bootstrap - mlops-lab"
echo "================================================"

# Function to create and restore bucket
setup_bucket() {
    BUCKET=$1
    echo "[S3] Setting up $BUCKET..."
    aws --endpoint-url=http://localhost:4566 s3 mb s3://$BUCKET || true

    if [ -d "$BACKUP_PATH/$BUCKET" ]; then
        echo "[S3] Restoring existing data from volume for $BUCKET..."
        aws --endpoint-url=http://localhost:4566 s3 sync "$BACKUP_PATH/$BUCKET" s3://$BUCKET
    fi
}

setup_bucket "data-lake"
setup_bucket "mlflow-artifacts"

# ---------------------------------------------------------------------------
# S3 prefix scaffolding (only runs if bucket was empty)
# ---------------------------------------------------------------------------
echo "[S3] Ensuring prefix structure..."
aws --endpoint-url=http://localhost:4566 s3api put-object --bucket data-lake --key bronze/ || true
aws --endpoint-url=http://localhost:4566 s3api put-object --bucket data-lake --key silver/ || true
aws --endpoint-url=http://localhost:4566 s3api put-object --bucket data-lake --key gold/ || true
aws --endpoint-url=http://localhost:4566 s3api put-object --bucket data-lake --key feast/registry/ || true

echo "[DynamoDB] Skipping Feast table pre-creation - managed by feast apply."
echo "================================================"
echo "LocalStack bootstrap complete."
echo "================================================"