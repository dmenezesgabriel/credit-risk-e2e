FROM apache/airflow:slim-3.1.8-python3.11

# --- System-level deps for SageMaker local mode & @task.virtualenv ---
USER root

# procps: pgrep used by sagemaker.local.utils.kill_child_processes()
RUN apt-get update && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI + compose plugin: SageMaker SDK shells out to `docker` and
# `docker-compose` in local mode
COPY --from=docker:28-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker:28-cli /usr/local/libexec/docker/cli-plugins/docker-compose \
     /usr/local/libexec/docker/cli-plugins/docker-compose

# docker-compose shim: SDK calls `docker-compose` (V1); delegate to V2
RUN printf '#!/bin/sh\nexec docker compose "$@"\n' \
    > /usr/local/bin/docker-compose && chmod +x /usr/local/bin/docker-compose

# uv: fast virtualenv backend for @task.virtualenv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-postgres==6.6.2 \
    apache-airflow-providers-docker==3.14.0 \
    apache-airflow-providers-fab==3.6.0 \
    authlib==1.6.9
