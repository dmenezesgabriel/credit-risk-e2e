# type: ignore
"""
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


def glue_task(task_id: str, script: str) -> DockerOperator:
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
        /workspace/give_me_some_credit/glue_jobs/{script} {{{{ ds }}}}
        """,  # {{ ds }} is Airflow's logical execution date
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        mounts=[
            Mount(
                source="mlops-workspace",  # Docker volume (fast)
                target="/workspace",
                type="volume",
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
    dag_id="credit_risk_data_pipeline",
    description="Bronze => Silver => Gold => Feast materialise",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # triggered manually or by upstream sensor
    catchup=False,
    tags=["credit-risk", "data-pipeline"],
) as dag:

    t1_ingest = glue_task(
        "bronze_ingestion",
        "bronze_ingestion.py",
    )
    t2_silver = glue_task(
        "silver_cleaning",
        "silver_cleaning.py",
    )
    t3_gold = glue_task(
        "gold_feature_engineering",
        "gold_feature_engineering.py",
    )

    t1_ingest >> t2_silver >> t3_gold
