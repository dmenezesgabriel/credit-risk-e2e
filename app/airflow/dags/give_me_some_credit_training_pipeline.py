# type: ignore
"""
Airflow DAG — Credit Risk Training Pipeline
============================================
Orchestrates the SageMaker training pipeline using @task.virtualenv
so each step runs in an isolated venv with its own Python dependencies.

Tasks:
  1. detect_ingestion_date — find latest train_features partition on S3
  2. build_training_image  — docker build the SageMaker training image
  3. run_training_pipeline — execute sm_pipeline.py (SageMaker local mode)

The existing sm_pipeline.py is invoked via subprocess; no changes needed.
"""

import os
from datetime import datetime

from airflow.sdk import DAG, task

# ── env vars inherited from docker-compose ──────────────────────────
S3_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localstack:4566")
S3_BUCKET = "data-lake"
NETWORK = os.environ.get("NETWORK", "mlops-lab-net")

with DAG(
    dag_id="give_me_some_credit_training_pipeline",
    description="SageMaker training pipeline: preprocess → train → tune → evaluate",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["give-me-some-credit", "sagemaker", "training"],
    params={
        "ingestion_date": "",
        "n_trials": 3,
        "auc_threshold": 0.85,
    },
) as dag:

    # ----------------------------------------------------------------
    # Task 1 — Detect latest ingestion date from S3
    # ----------------------------------------------------------------
    @task.virtualenv(
        task_id="detect_ingestion_date",
        requirements=["boto3==1.42.82"],
        system_site_packages=False,
    )
    def detect_ingestion_date(
        s3_endpoint: str,
        s3_bucket: str,
        ingestion_date_override: str,
    ) -> str:
        """Return explicit date if given, otherwise discover latest partition."""
        if ingestion_date_override:
            return ingestion_date_override

        import boto3

        s3 = boto3.client("s3", endpoint_url=s3_endpoint)
        prefix = "gold/give_me_some_credit/train_features/"

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=s3_bucket, Prefix=prefix, Delimiter="/"
        )

        dates = []
        for page in pages:
            for obj in page.get("CommonPrefixes", []):
                folder = obj.get("Prefix", "").split("=")[-1].strip("/")
                dates.append(folder)

        if not dates:
            raise ValueError(
                f"No partitions found in s3://{s3_bucket}/{prefix}"
            )
        return sorted(dates)[-1]

    # ----------------------------------------------------------------
    # Task 2 — Build training Docker image
    # ----------------------------------------------------------------
    @task(task_id="build_training_image")
    def build_training_image() -> None:
        """Build credit-risk-training:latest from the training Dockerfile."""
        import subprocess

        result = subprocess.run(
            [
                "docker",
                "build",
                "--quiet",
                "-t",
                "credit-risk-training:latest",
                "-f",
                "/workspace/give_me_some_credit/sagemaker/training/Dockerfile",
                "/workspace/give_me_some_credit/sagemaker/training/",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker build failed (rc={result.returncode}):\n"
                f"{result.stderr}"
            )
        print(f"Image built: {result.stdout.strip()}")

    # ----------------------------------------------------------------
    # Task 3 — Run SageMaker training pipeline
    # ----------------------------------------------------------------
    @task.virtualenv(
        task_id="run_training_pipeline",
        requirements=[
            "sagemaker>=2.0.0,<3.0.0",
            "sagemaker[local]",
            "boto3==1.42.82",
            "pytz==2026.1.post1",
            "requests==2.33.1",
        ],
        system_site_packages=False,
    )
    def run_training_pipeline(
        ingestion_date: str,
        n_trials: int,
        auc_threshold: float,
        s3_endpoint: str,
        network: str,
    ) -> None:
        """Call sm_pipeline.py via subprocess inside the virtualenv."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "/workspace/give_me_some_credit/sagemaker/training/sm_pipeline.py",
            "--mode",
            "local",
            "--ingestion-date",
            ingestion_date,
            "--s3-endpoint",
            s3_endpoint,
            "--n-trials",
            str(n_trials),
            "--auc-threshold",
            str(auc_threshold),
            "--network",
            network,
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sm_pipeline.py failed (rc={result.returncode})"
            )

    # ── Wire the DAG ────────────────────────────────────────────────
    date = detect_ingestion_date(
        s3_endpoint=S3_ENDPOINT,
        s3_bucket=S3_BUCKET,
        ingestion_date_override="{{ params.ingestion_date }}",
    )
    img = build_training_image()

    [date, img] >> run_training_pipeline(
        ingestion_date=date,
        n_trials="{{ params.n_trials }}",
        auc_threshold="{{ params.auc_threshold }}",
        s3_endpoint=S3_ENDPOINT,
        network=NETWORK,
    )
