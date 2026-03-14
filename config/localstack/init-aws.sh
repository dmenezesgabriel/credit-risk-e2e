#!/bin/sh
set -e

echo "================================================"
echo "LocalStack bootstrap — credit-risk-e2e"
echo "================================================"

# ---------------------------------------------------------------------------
# S3 buckets
# ---------------------------------------------------------------------------
echo "[S3] Creating data-lake bucket..."
aws --endpoint-url=http://localhost:4566 s3 mb s3://data-lake || true

echo "[S3] Creating mlflow-artifacts bucket..."
aws --endpoint-url=http://localhost:4566 s3 mb s3://mlflow-artifacts || true

# ---------------------------------------------------------------------------
# S3 prefix scaffolding
# ---------------------------------------------------------------------------
echo "[S3] Scaffolding prefix structure..."
aws --endpoint-url=http://localhost:4566 s3api put-object \
    --bucket data-lake --key bronze/ || true
aws --endpoint-url=http://localhost:4566 s3api put-object \
    --bucket data-lake --key silver/ || true
aws --endpoint-url=http://localhost:4566 s3api put-object \
    --bucket data-lake --key gold/ || true
aws --endpoint-url=http://localhost:4566 s3api put-object \
    --bucket data-lake --key feast/registry/ || true

# ---------------------------------------------------------------------------
# DynamoDB — DO NOT pre-create Feast tables here.
# Feast creates its own DynamoDB tables with its own schema during
# `feast apply`. Pre-creating them with a different schema causes
# ValidationException: "The provided key element does not match the schema".
# Let feast apply own the table lifecycle entirely.
# ---------------------------------------------------------------------------
echo "[DynamoDB] Skipping Feast table pre-creation — managed by feast apply."

echo "================================================"
echo "LocalStack bootstrap complete."
echo "================================================"
