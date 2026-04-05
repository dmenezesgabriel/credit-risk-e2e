# type: ignore
"""
Airflow DAG — Credit Risk Batch Inference Pipeline
===================================================
Orchestrates SageMaker batch inference using @task.virtualenv
so each step runs in an isolated venv with its own Python dependencies.

Tasks:
  1. detect_ingestion_date — find latest test_features partition on S3
  2. build_inference_image  — docker build the SageMaker inference image
  3. check_champion_model   — verify a @champion model exists in MLflow
  4. run_batch_inference     — execute sm_batch_inference.py (local mode)
"""

import os
from datetime import datetime

from airflow.sdk import DAG, task

# ── env vars inherited from docker-compose ──────────────────────────
S3_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localstack:4566")
S3_BUCKET = "data-lake"
MLFLOW_URI = "http://mlflow:5000"
NETWORK = os.environ.get("NETWORK", "mlops-lab-net")

with DAG(
    dag_id="give_me_some_credit_batch_inference_pipeline",
    description="SageMaker batch inference: prepare model → transform → monitor",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["give-me-some-credit", "sagemaker", "batch-inference"],
    params={
        "ingestion_date": "",
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
        prefix = "gold/give_me_some_credit/test_features/"

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
    # Task 2 — Build inference Docker image
    # ----------------------------------------------------------------
    @task(task_id="build_inference_image")
    def build_inference_image() -> None:
        """Build credit-risk-inference:latest from the inference Dockerfile."""
        import subprocess

        result = subprocess.run(
            [
                "docker",
                "build",
                "--quiet",
                "-t",
                "credit-risk-inference:latest",
                "-f",
                "/workspace/give_me_some_credit/sagemaker/batch_inference/Dockerfile",
                "/workspace/give_me_some_credit/sagemaker/batch_inference/",
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
    # Task 3 — Check champion model exists in MLflow
    # ----------------------------------------------------------------
    @task.virtualenv(
        task_id="check_champion_model",
        requirements=["mlflow==2.22.1", "requests==2.33.1"],
        system_site_packages=False,
    )
    def check_champion_model(mlflow_uri: str) -> str:
        """Verify give_me_some_credit_champion@champion exists in MLflow."""
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        client = mlflow.MlflowClient()

        model_name = "give_me_some_credit_champion"
        try:
            mv = client.get_model_version_by_alias(model_name, "champion")
            info = f"Model: {model_name} v{mv.version} (champion)"
            print(info)
            return info
        except Exception as exc:
            raise ValueError(
                f"No @champion alias found for '{model_name}'. "
                f"Run the training pipeline first. Error: {exc}"
            ) from exc

    # ----------------------------------------------------------------
    # Task 4 — Run batch inference
    # ----------------------------------------------------------------
    @task.virtualenv(
        task_id="run_batch_inference",
        requirements=[
            "sagemaker>=2.0.0,<3.0.0",
            "sagemaker[local]",
            "boto3==1.42.82",
            "pytz==2026.1.post1",
            "requests==2.33.1",
            "pyarrow==23.0.1",
        ],
        system_site_packages=False,
    )
    def run_batch_inference(
        ingestion_date: str,
        s3_endpoint: str,
        network: str,
    ) -> None:
        """Call sm_batch_inference.py via subprocess inside the virtualenv."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "/workspace/give_me_some_credit/sagemaker/batch_inference/sm_batch_inference.py",
            "--mode",
            "local",
            "--ingestion-date",
            ingestion_date,
            "--s3-endpoint",
            s3_endpoint,
            "--network",
            network,
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sm_batch_inference.py failed (rc={result.returncode})"
            )

    # ── Wire the DAG ────────────────────────────────────────────────
    date = detect_ingestion_date(
        s3_endpoint=S3_ENDPOINT,
        s3_bucket=S3_BUCKET,
        ingestion_date_override="{{ params.ingestion_date }}",
    )
    img = build_inference_image()
    champ = check_champion_model(mlflow_uri=MLFLOW_URI)

    [date, img, champ] >> run_batch_inference(
        ingestion_date=date,
        s3_endpoint=S3_ENDPOINT,
        network=NETWORK,
    )
