#!/bin/bash
set -e

# Create mlflow database + user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    CREATE DATABASE mlflow OWNER mlflow;

    CREATE USER airflow WITH PASSWORD 'airflow';
    CREATE DATABASE airflow OWNER airflow;
EOSQL
