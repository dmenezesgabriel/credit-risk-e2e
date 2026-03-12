"""
In a real world situation, instead of using docker operator
it would do a python script to trigger glue via aws API
"""

import os
from datetime import datetime

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from airflow import DAG

PROJECT_ROOT = os.environ["PROJECT_ROOT"]

with DAG(
    dag_id="credit_risk_bronze_ingestion",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    ingest = DockerOperator(
        task_id="glue_bronze_ingestion",
        image="public.ecr.aws/glue/aws-glue-libs:5",
        command="spark-submit /workspace/glue_jobs/bronze_ingestion.py",
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
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
            "PYTHONPATH": "/workspace",
            "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", ""),
            "KAGGLE_KEY": os.environ.get("KAGGLE_KEY", ""),
        },
        auto_remove="success",
    )
