"""
sm_batch_inference.py — SageMaker Batch Inference Pipeline
==========================================================
Defines a single-step pipeline that loads the champion model and
fitted preprocessor from MLflow, scores inference features from the
gold layer, and writes predictions to S3.

Requires:
    1. The training pipeline has completed and registered a champion
       model in MLflow Model Registry ("credit_risk_champion" / Staging).
    2. The inference data pipeline has produced gold inference features
       at s3://data-lake/gold/credit_risk/inference_features/ingestion_date=<date>/

Switch environments:
    Local : --mode local  --s3-endpoint http://localstack:4566 --network <n>
    AWS   : --mode aws    (no s3-endpoint or network needed)

Run locally:
    python inference/sm_batch_inference.py --mode local --ingestion-date 2026-03-21
"""

import argparse
import logging
import os

# sagemaker_session.py lives in ../training/ — add to path
import sys

import sagemaker
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "training"
    ),
)
from sagemaker_session import make_sagemaker_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sagemaker_batch_inference")

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_pipeline(
    sagemaker_session: sagemaker.Session,
    role: str,
    ingestion_date: str,
    s3_bucket: str,
    mlflow_uri: str,
    experiment_name: str,
    model_stage: str,
    training_image: str,
    instance_type: str,
    aws_region: str,
    s3_endpoint: str,
) -> Pipeline:
    pipeline_s3_prefix = f"projects/{experiment_name}/sagemaker/pipeline"
    pipeline_s3 = f"s3://{s3_bucket}/{pipeline_s3_prefix}"
    gold_inference_s3 = (
        f"s3://{s3_bucket}/gold/credit_risk/inference_features/"
        f"ingestion_date={ingestion_date}"
    )

    shared_env = {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": aws_region,
        "AWS_ENDPOINT_URL": s3_endpoint,
        "GIT_PYTHON_REFRESH": "quiet",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector:4317",
        "OTEL_EXPORTER_OTLP_INSECURE": "true",
        "OTEL_SERVICE_NAME": "batch-inference",
        "OTEL_RESOURCE_ATTRIBUTES": "service.namespace=mlops,project=credit-risk,service.version=1.0",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_TRACES_SAMPLER": "always_on",
    }

    # Single step — batch transform
    step_batch_transform = ProcessingStep(
        name="BatchTransform",
        processor=ScriptProcessor(
            command=["opentelemetry-instrument", "python"],
            image_uri=training_image,
            role=role,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            env=shared_env,
        ),
        code=os.path.join(BASE_DIR, "batch_transform.py"),
        job_arguments=[
            "--mlflow-uri",
            mlflow_uri,
            "--experiment-name",
            experiment_name,
            "--model-stage",
            model_stage,
        ],
        inputs=[
            ProcessingInput(
                source=gold_inference_s3,
                destination="/opt/ml/processing/input/inference",
                input_name="inference",
            ),
            ProcessingInput(
                source=f"{pipeline_s3}/evaluation/evaluation_report.json",
                destination="/opt/ml/processing/input/eval_report",
                input_name="eval_report",
            ),
            ProcessingInput(
                source=f"{pipeline_s3}/preprocessing/preprocessor/prep_meta.json",
                destination="/opt/ml/processing/input/prep_meta",
                input_name="prep_meta",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source="/opt/ml/processing/output/predictions",
                destination=(
                    f"s3://{s3_bucket}/predictions/credit_risk/{ingestion_date}"
                ),
            ),
        ],
    )

    return Pipeline(
        name="CreditRiskBatchInferencePipeline",
        steps=[step_batch_transform],
        sagemaker_session=sagemaker_session,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit risk batch inference pipeline"
    )
    parser.add_argument("--mode", default="local", choices=["local", "aws"])
    parser.add_argument("--ingestion-date", default="2026-03-21")
    parser.add_argument("--s3-endpoint", default="http://localstack:4566")
    parser.add_argument("--s3-bucket", default="data-lake")
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="credit_risk_pipeline")
    parser.add_argument("--model-stage", default="Staging")
    parser.add_argument(
        "--training-image", default="credit-risk-training:latest"
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

    pipeline = get_pipeline(
        sagemaker_session=sagemaker_session,
        role=role,
        ingestion_date=args.ingestion_date,
        s3_bucket=args.s3_bucket,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
        model_stage=args.model_stage,
        training_image=args.training_image,
        instance_type=instance_type,
        aws_region=args.aws_region,
        s3_endpoint=args.s3_endpoint,
    )

    pipeline.upsert(role_arn=role)

    execution = pipeline.start()

    exec_id = getattr(execution, "name", None) or getattr(
        execution, "_execution_id", "local"
    )
    logger.info(f"Pipeline execution started: {exec_id}")

    try:
        steps = execution.list_steps()
        logger.info("Pipeline step summary:")
        for step in steps:
            name = step.get("StepName", "?")
            status = step.get("StepStatus", "?")
            fail = step.get("FailureReason", "")
            logger.info(f"  {name}: {status}{' - ' + fail if fail else ''}")
    except Exception as e:
        logger.info(f"Could not retrieve step summary: {e}")

    logger.info("Batch inference pipeline complete.")


if __name__ == "__main__":
    main(parse_args())
