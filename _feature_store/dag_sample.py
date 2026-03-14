from airflow.providers.docker.operators.docker import (
    DockerOperator,
)  # type: ignore
from docker.types import Mount  # type: ignore

from airflow import DAG


def feast_materialize_task() -> DockerOperator:
    return DockerOperator(  # type: ignore
        task_id="feast_materialize",
        image="python:3.11-slim",
        entrypoint="bash",
        command=[
            "-c",
            "pip install -q 'feast[aws]==0.40.1' 's3fs' && "
            "cd /workspace/feature_store/feature_repo && "
            "feast apply && "
            "feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)",
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        mounts=[Mount(source=PROJECT_ROOT, target="/workspace", type="bind")],
        mount_tmp_dir=False,
        environment={
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
            "AWS_ENDPOINT_URL": AWS_ENDPOINT_URL,
            "FEAST_S3_ENDPOINT_URL": AWS_ENDPOINT_URL,
        },
        auto_remove="success",
    )
