"""
sm_batch_inference.py — SageMaker Batch Transform Orchestrator
==============================================================
Runs batch inference on gold inference data using the champion model
from MLflow Registry.

  Step 1  ProcessingStep  prepare_model  => model.tar.gz on S3
  Step 2  Batch Transform               => predictions on S3

The inference container (BYOC) implements the SageMaker serving contract
(/ping + /invocations) and handles preprocessing + scoring internally.

Switch environments:
    Local : --mode local  --s3-endpoint http://localstack:4566 --network <n>
    AWS   : --mode aws    (no s3-endpoint or network needed)

Build image:
    docker build -t credit-risk-inference:latest -f inference/Dockerfile inference/

Run locally:
    python inference/sm_batch_inference.py --mode local --ingestion-date 2026-03-21

Local-mode prerequisites (Jupyter container):
    - ``procps`` apt package must be installed.  The SageMaker SDK uses
      ``pgrep`` (from procps) to find and kill child processes of the
      ``docker-compose up`` serving container during cleanup
      (``sagemaker.local.utils.kill_child_processes``).  Without it,
      Batch Transform succeeds but crashes on ``stop_serving()``.
    - ``sagemaker_session.make_sagemaker_session`` patches
      ``get_docker_host`` so the SDK connects to the Docker host gateway
      IP instead of ``localhost`` (which inside a container refers to
      the caller, not the Docker host where port 8080 is mapped).
"""

import argparse
import logging
import os
import sys

import sagemaker
from sagemaker.model import Model
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)

# Allow importing sagemaker_session from the training sibling package
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "training"
    ),
)
from sagemaker_session import make_sagemaker_session  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sagemaker_batch_inference")

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_step_path(filename: str) -> str:
    return os.path.join(BASE_DIR, "steps", filename)


def run_batch_inference(
    sagemaker_session: sagemaker.Session,
    role: str,
    ingestion_date: str,
    s3_bucket: str,
    mlflow_uri: str,
    experiment_name: str,
    training_image: str,
    inference_image: str,
    instance_type: str,
    aws_region: str,
    s3_endpoint: str,
) -> None:
    pipeline_s3_prefix = f"projects/{experiment_name}/sagemaker/pipeline"
    pipeline_s3 = f"s3://{s3_bucket}/{pipeline_s3_prefix}"
    gold_inference_s3 = (
        f"s3://{s3_bucket}/gold/give_me_some_credit/"
        f"test_features/ingestion_date={ingestion_date}"
    )
    predictions_s3 = (
        f"{pipeline_s3}/inference/predictions/ingestion_date={ingestion_date}"
    )
    model_s3 = f"{pipeline_s3}/inference/model"

    shared_env = {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": aws_region,
        "AWS_ENDPOINT_URL": s3_endpoint,
        "GIT_PYTHON_REFRESH": "quiet",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector:4317",
        "OTEL_EXPORTER_OTLP_INSECURE": "true",
        "OTEL_SERVICE_NAME": "sagemaker-batch-inference",
        "OTEL_RESOURCE_ATTRIBUTES": (
            "service.namespace=mlops,project=credit-risk,service.version=1.0"
        ),
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_TRACES_SAMPLER": "always_on",
    }

    # -----------------------------------------------------------------
    # Step 1 — Prepare model artifacts (uses training image: has mlflow)
    # -----------------------------------------------------------------
    logger.info("Step 1: Preparing model artifacts...")

    prepare_processor = ScriptProcessor(
        command=["opentelemetry-instrument", "python"],
        image_uri=training_image,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        env={
            **shared_env,
            "OTEL_SERVICE_NAME": "prepare-model",
        },
    )

    prepare_processor.run(
        code=get_step_path("prepare_model.py"),
        arguments=[
            "--mlflow-uri",
            mlflow_uri,
            "--experiment-name",
            experiment_name,
        ],
        inputs=[
            ProcessingInput(
                source=f"{pipeline_s3}/evaluation/evaluation_report.json",
                destination="/opt/ml/processing/input/evaluation",
                input_name="evaluation",
            ),
            ProcessingInput(
                source=f"{pipeline_s3}/preprocessing/preprocessor/",
                destination="/opt/ml/processing/input/prep_meta",
                input_name="prep_meta",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/output/model",
                destination=model_s3,
            ),
        ],
    )
    logger.info("Step 1 complete: model.tar.gz uploaded to S3.")

    # -----------------------------------------------------------------
    # Step 2 — Batch Transform using inference container
    # -----------------------------------------------------------------
    logger.info("Step 2: Running Batch Transform...")

    model_data = f"{model_s3}/model.tar.gz"

    model = Model(
        image_uri=inference_image,
        model_data=model_data,
        role=role,
        sagemaker_session=sagemaker_session,
        env={
            "MODEL_SERVER_WORKERS": "1",
            "MODEL_SERVER_TIMEOUT": "300",
        },
    )

    transformer = model.transformer(
        instance_count=1,
        instance_type=instance_type,
        output_path=predictions_s3,
        accept="application/x-parquet",
        strategy="SingleRecord",
    )

    # SageMaker appends a ".out" extension to each output file.
    # E.g. part-00000.parquet becomes part-00000.parquet.out
    transformer.transform(
        data=gold_inference_s3,
        content_type="application/x-parquet",
        split_type=None,
    )

    logger.info(f"Step 2 complete: predictions written to {predictions_s3}")

    # -----------------------------------------------------------------
    # Step 3 — Monitoring (uses training image: has evidently, mlflow)
    # -----------------------------------------------------------------
    logger.info("Step 3: Running batch monitoring...")

    monitoring_s3 = (
        f"{pipeline_s3}/inference/monitoring/ingestion_date={ingestion_date}"
    )

    monitor_processor = ScriptProcessor(
        command=["opentelemetry-instrument", "python"],
        image_uri=training_image,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        env={
            **shared_env,
            "OTEL_SERVICE_NAME": "monitor-batch",
        },
    )

    monitor_processor.run(
        code=get_step_path("monitor_batch.py"),
        arguments=[
            "--mlflow-uri",
            mlflow_uri,
            "--experiment-name",
            experiment_name,
            "--ingestion-date",
            ingestion_date,
        ],
        inputs=[
            ProcessingInput(
                source=gold_inference_s3,
                destination="/opt/ml/processing/input/features",
                input_name="features",
            ),
            ProcessingInput(
                source=predictions_s3,
                destination="/opt/ml/processing/input/predictions",
                input_name="predictions",
            ),
            ProcessingInput(
                source=f"{pipeline_s3}/preprocessing/preprocessor/",
                destination="/opt/ml/processing/input/model_data",
                input_name="model_data",
            ),
            ProcessingInput(
                source=f"{pipeline_s3}/evaluation/evaluation_report.json",
                destination="/opt/ml/processing/input/model_data",
                input_name="evaluation",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="report",
                source="/opt/ml/processing/output/report",
                destination=monitoring_s3,
            ),
        ],
    )
    logger.info(
        f"Step 3 complete: monitoring report written to {monitoring_s3}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit risk batch inference via SageMaker Batch Transform"
    )
    parser.add_argument("--mode", default="local", choices=["local", "aws"])
    parser.add_argument("--ingestion-date", default="2026-03-21")
    parser.add_argument("--s3-endpoint", default="http://localstack:4566")
    parser.add_argument("--s3-bucket", default="data-lake")
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="give_me_some_credit")
    parser.add_argument(
        "--training-image", default="credit-risk-training:latest"
    )
    parser.add_argument(
        "--inference-image", default="credit-risk-inference:latest"
    )
    parser.add_argument("--network", default="mlops-lab_mlops-lab-net")
    parser.add_argument("--aws-region", default="us-east-1")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logger.info(f"Args: {vars(args)}")

    sagemaker_session, _ = make_sagemaker_session(
        mode=args.mode,
        s3_bucket=args.s3_bucket,
        s3_endpoint=args.s3_endpoint if args.mode == "local" else None,
        network=args.network if args.mode == "local" else None,
        aws_region=args.aws_region,
    )

    role = "arn:aws:iam::111111111111:role/SageMakerExecutionRole"
    instance_type = "local" if args.mode == "local" else "ml.m5.xlarge"

    run_batch_inference(
        sagemaker_session=sagemaker_session,
        role=role,
        ingestion_date=args.ingestion_date,
        s3_bucket=args.s3_bucket,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
        training_image=args.training_image,
        inference_image=args.inference_image,
        instance_type=instance_type,
        aws_region=args.aws_region,
        s3_endpoint=args.s3_endpoint,
    )

    logger.info("Batch inference pipeline complete.")


if __name__ == "__main__":
    main(parse_args())
