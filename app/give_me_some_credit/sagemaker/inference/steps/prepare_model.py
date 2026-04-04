"""
prepare_model.py — SageMaker ProcessingStep: Model Preparation

Downloads the champion model, preprocessor, and evaluation report from
MLflow and packages them into a model.tar.gz that SageMaker Batch
Transform will extract into /opt/ml/model/ inside the inference container.

SageMaker mounts:
  Input  "evaluation" => /opt/ml/processing/input/evaluation/
  Input  "prep_meta"  => /opt/ml/processing/input/prep_meta/
  Output "model"      => /opt/ml/processing/output/model/

Expected output layout inside model.tar.gz:
  champion/           # MLflow sklearn model directory
  preprocessor/       # MLflow sklearn preprocessor directory
  evaluation_report.json

Run locally:
    python prepare_model.py \
        --mlflow-uri http://localhost:5000 \
        --experiment-name give_me_some_credit
"""

import argparse
import json
import logging
import os
import shutil
import tarfile

import mlflow
import mlflow.artifacts
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("prepare_model")

# ---------------------------------------------------------------------------
# SageMaker I/O paths
# ---------------------------------------------------------------------------
EVALUATION_DIR = "/opt/ml/processing/input/evaluation"
PREP_META_DIR = "/opt/ml/processing/input/prep_meta"
OUTPUT_DIR = "/opt/ml/processing/output/model"

REGISTRY_MODEL_NAME = "give_me_some_credit_champion"


def setup_otel_logging(service_name: str):
    """Configures OTEL logging (Loki) and Console logging (stdout)."""
    provider = LoggerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    set_logger_provider(provider)

    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]

    exporter = OTLPLogExporter(endpoint=endpoint, insecure=True)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    otel_handler = LoggingHandler(level=logging.INFO, logger_provider=provider)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(otel_handler)
    root_logger.addHandler(console_handler)

    return provider


# ---------------------------------------------------------------------------
# Resolve artifacts from MLflow
# ---------------------------------------------------------------------------
def get_staging_run_id(client: mlflow.tracking.MlflowClient) -> str:
    """Get the run_id of the latest Staging version of the champion model."""
    versions = client.get_latest_versions(
        REGISTRY_MODEL_NAME, stages=["Staging"]
    )
    if not versions:
        raise RuntimeError(
            f"No Staging version found for '{REGISTRY_MODEL_NAME}'"
        )
    version = versions[0]
    logger.info(
        f"Found {REGISTRY_MODEL_NAME} v{version.version} "
        f"(stage=Staging, run_id={version.run_id})"
    )
    return version.run_id


def get_preprocessor_run_id(prep_meta_dir: str) -> str:
    """Read the preprocessor run_id from prep_meta.json."""
    meta_path = os.path.join(prep_meta_dir, "prep_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    run_id = meta["run_id"]
    logger.info(f"Preprocessor run_id from prep_meta.json: {run_id}")
    return run_id


def download_champion(run_id: str, dst: str) -> str:
    """Download the champion model artifacts from MLflow."""
    path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/model", dst_path=dst
    )
    logger.info(f"Champion model downloaded to {path}")
    return path


def download_preprocessor(run_id: str, dst: str) -> str:
    """Download the preprocessor artifacts from MLflow."""
    path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/preprocessor", dst_path=dst
    )
    logger.info(f"Preprocessor downloaded to {path}")
    return path


# ---------------------------------------------------------------------------
# Package model.tar.gz
# ---------------------------------------------------------------------------
def create_model_tarball(staging_dir: str, output_dir: str) -> str:
    """
    Create model.tar.gz from the staging directory.

    SageMaker extracts the tarball into /opt/ml/model/ in the container,
    so the archive root should contain champion/, preprocessor/, and
    evaluation_report.json directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    tarball_path = os.path.join(output_dir, "model.tar.gz")

    with tarfile.open(tarball_path, "w:gz") as tar:
        for item in os.listdir(staging_dir):
            full_path = os.path.join(staging_dir, item)
            tar.add(full_path, arcname=item)

    size_mb = os.path.getsize(tarball_path) / (1024 * 1024)
    logger.info(f"Created {tarball_path} ({size_mb:.1f} MB)")
    return tarball_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    otel_provider = setup_otel_logging("sagemaker-pipeline-prepare-model")

    logger.info(f"Args: {vars(args)}")

    try:
        mlflow.set_tracking_uri(args.mlflow_uri)

        client = mlflow.tracking.MlflowClient()
        champion_run_id = get_staging_run_id(client)
        preprocessor_run_id = get_preprocessor_run_id(PREP_META_DIR)

        # Stage artifacts into a temp directory before tarring
        staging_dir = "/tmp/model_staging"
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir)

        download_champion(champion_run_id, staging_dir)
        # mlflow.artifacts.download_artifacts creates model/ and preprocessor/
        # subdirs inside staging_dir. Rename to champion/ for clarity.
        os.rename(
            os.path.join(staging_dir, "model"),
            os.path.join(staging_dir, "champion"),
        )
        download_preprocessor(preprocessor_run_id, staging_dir)

        # Copy evaluation report into the staging directory
        eval_src = os.path.join(EVALUATION_DIR, "evaluation_report.json")
        eval_dst = os.path.join(staging_dir, "evaluation_report.json")
        shutil.copy2(eval_src, eval_dst)
        logger.info(f"Copied evaluation report to {eval_dst}")

        create_model_tarball(staging_dir, OUTPUT_DIR)

        logger.info("Model preparation complete.")
    finally:
        otel_provider.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare model artifacts for batch inference"
    )
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="give_me_some_credit")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
