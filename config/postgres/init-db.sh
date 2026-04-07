#!/bin/bash
# config/init-db.sh
set -e

# Create mlflow database + user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    CREATE DATABASE mlflow OWNER mlflow;

    CREATE USER airflow WITH PASSWORD 'airflow';
    CREATE DATABASE airflow OWNER airflow;

    CREATE USER marquez WITH PASSWORD 'marquez';
    CREATE DATABASE marquez OWNER marquez;
    GRANT ALL PRIVILEGES ON DATABASE marquez TO marquez;
EOSQL
