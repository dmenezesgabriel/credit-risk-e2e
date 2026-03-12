"""
In a real world situation, instead of using docker operator
it would do a python script to trigger glue via aws API
"""

import os
from datetime import datetime

from airflow.providers.docker.operators.docker import (
    DockerOperator,
)  # type: ignore
from docker.types import Mount  # type: ignore

from airflow import DAG  # type: ignore

PROJECT_ROOT = os.environ["PROJECT_ROOT"]


def glue_task(task_id: str, script: str) -> DockerOperator:
    return DockerOperator(  # type: ignore
        task_id=f"{task_id}",
        image="public.ecr.aws/glue/aws-glue-libs:5",
        command=f"""
        spark-submit
        --conf spark.hadoop.fs.s3a.endpoint={os.environ["AWS_ENDPOINT_URL"]}
        --conf spark.hadoop.fs.s3a.access.key=test
        --conf spark.hadoop.fs.s3a.secret.key=test
        --conf spark.hadoop.fs.s3a.path.style.access=true
        /workspace/glue_jobs/{script}
        """,
        docker_url="unix://var/run/docker.sock",
        network_mode="credit-risk-e2e_credit-risk-net",
        mounts=[
            Mount(
                source=PROJECT_ROOT,  # From host, set on .env
                target="/workspace",
                type="bind",
            )
        ],
        mount_tmp_dir=False,
        environment={
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"],
            "AWS_ENDPOINT_URL": os.environ["AWS_ENDPOINT_URL"],
            "PYTHONPATH": "/workspace",
            "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", ""),
            "KAGGLE_KEY": os.environ.get("KAGGLE_KEY", ""),
        },
        auto_remove="success",
    )


with DAG(
    dag_id="credit_risk_bronze_ingestion",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    ingest = glue_task("bronze_ingestion", "bronze_ingestion.py")
    silver = glue_task("silver_cleaning", "silver_cleaning.py")

    ingest >> silver
