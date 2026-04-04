# type: ignore
"""
Airflow DAG — Inference data pipeline for credit risk.

Ingests cs-test.csv through the same Bronze => Silver => Gold medallion

In a real world situation, instead of using docker operator
it would do a python script to trigger glue via aws API
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
AWS_ENDPOINT_URL = os.environ["AWS_ENDPOINT_URL"]

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")

NETWORK = os.environ["NETWORK"]
PROJECT_ROOT = os.environ["PROJECT_ROOT"]

FILE_NAME = "cs-test.csv"
DATASET_TYPE = "inference"


def glue_task(
    task_id: str,
    script: str,
    extra_args: str = "",
) -> DockerOperator:
    """Run a Glue/Spark job via Docker with optional positional args."""
    return DockerOperator(
        task_id=task_id,
        image="public.ecr.aws/glue/aws-glue-libs:5",
        command=f"""
        spark-submit
        --conf spark.driver.memory=512m
        --conf spark.executor.memory=512m
        --conf spark.sql.shuffle.partitions=2
        --conf spark.default.parallelism=2
        --conf spark.hadoop.fs.s3a.endpoint={AWS_ENDPOINT_URL}
        --conf spark.hadoop.fs.s3a.access.key=test
        --conf spark.hadoop.fs.s3a.secret.key=test
        --conf spark.hadoop.fs.s3a.path.style.access=true
        --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
        --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider
        /workspace/give_me_some_credit/glue_jobs/{script} {{{{ macros.datetime.now().strftime('%Y-%m-%d') }}}} {extra_args}
        """,
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        mounts=[
            Mount(
                source=PROJECT_ROOT,
                target="/workspace",
                type="bind",
            )
        ],
        mount_tmp_dir=False,
        environment={
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
            "AWS_ENDPOINT_URL": AWS_ENDPOINT_URL,
            "PYTHONPATH": "/workspace",
            "KAGGLE_USERNAME": KAGGLE_USERNAME,
            "KAGGLE_KEY": KAGGLE_KEY,
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
        auto_remove="success",
    )


with DAG(
    dag_id="give_me_some_credit_test_data",
    description="Bronze => Silver => Gold",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # triggered manually or by upstream sensor
    catchup=False,
    tags=["give-me-some-credit", "data-pipeline", "kaggle", "test-data"],
) as dag:

    t1_ingest = glue_task(
        "bronze_ingestion",
        "bronze_ingestion.py",
        extra_args=f"{FILE_NAME} {DATASET_TYPE}",
    )
    t2_silver = glue_task(
        "silver_cleaning",
        "silver_cleaning.py",
        extra_args=f"{FILE_NAME} {DATASET_TYPE}",
    )
    # gold_feature_engineering.py takes (execution_date, dataset_type) — no file_name
    t3_gold = glue_task(
        "gold_feature_engineering",
        "gold_feature_engineering.py",
        extra_args=DATASET_TYPE,
    )

    t1_ingest >> t2_silver >> t3_gold
