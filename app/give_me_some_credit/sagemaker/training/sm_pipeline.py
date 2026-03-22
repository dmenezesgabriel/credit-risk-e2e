"""
sm_pipeline.py - SageMaker Pipeline Definition
===============================================
Defines a 4-step ML pipeline:

  Step 1 ProcessingStep  preprocess   => splits + prep_meta.json => S3
  Step 2 TrainingStep    train_step   => baseline models => S3
  Step 3 TrainingStep    tune_step    => tuned champion => S3
  Step 4 ProcessingStep  evaluate     => test metrics + conditional register

Each step runs in its own container with the same custom image.

Switch environments:
    Local : --mode local  --s3-endpoint http://localstack:4566 --network <n>
    AWS   : --mode aws    (no s3-endpoint or network needed)

Build image:
    docker build -t credit-risk-training:latest -f training/Dockerfile training/

Run locally:
    python training/sm_pipeline.py --mode local --ingestion-date 2026-03-21
"""

import argparse
import logging
import os

import sagemaker
import sagemaker.inputs
from sagemaker.estimator import Estimator
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker_session import make_sagemaker_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sagemaker_pipeline_give_me_some_credit")

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

RANDOM_STATE = 42


def get_step_path(filename: str) -> str:
    return os.path.join(BASE_DIR, "steps", filename)


def get_pipeline(
    sagemaker_session: sagemaker.Session,
    role: str,
    ingestion_date: str,
    s3_bucket: str,
    mlflow_uri: str,
    experiment_name: str,
    n_trials: int,
    auc_threshold: float,
    training_image: str,
    instance_type: str,
    aws_region: str,
    s3_endpoint: str,
) -> Pipeline:

    pipeline_s3 = f"s3://{s3_bucket}/sagemaker/pipeline"
    gold_s3 = f"s3://{s3_bucket}/gold/credit_risk/features/ingestion_date={ingestion_date}"

    # Infra/credentials vars shared across all containers.
    # Script-level config (mlflow, experiment, random state) is passed
    # explicitly via arguments= or hyperparameters= on each step.
    shared_env = {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": aws_region,
        "AWS_ENDPOINT_URL": s3_endpoint,
        "GIT_PYTHON_REFRESH": "quiet",
    }

    # Pipeline parameters — can be overridden at pipeline.start() time
    pipeline_ingestion_date = ParameterString(
        "IngestionDate", default_value=ingestion_date
    )
    pipeline_number_trials = ParameterInteger(
        "NTrials", default_value=n_trials
    )
    pipeline_auc_threshold = ParameterFloat(
        "AucThreshold", default_value=auc_threshold
    )
    pipeline_experiment_name = ParameterString(
        "ExperimentName", default_value=experiment_name
    )

    # -------------------------------------------------------------------------
    # Step 1 — Preprocessing
    # -------------------------------------------------------------------------
    step_preprocess = ProcessingStep(
        name="Preprocessing",
        processor=ScriptProcessor(
            command=["python"],
            image_uri=training_image,
            role=role,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            env=shared_env,
        ),
        code=get_step_path("preprocess.py"),
        job_arguments=[
            "--mlflow-uri",
            mlflow_uri,
            "--experiment-name",
            experiment_name,
            "--random-state",
            str(RANDOM_STATE),
        ],
        inputs=[
            ProcessingInput(
                source=gold_s3,
                destination="/opt/ml/processing/input/gold",
                input_name="gold",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"{pipeline_s3}/preprocessing/train",
            ),
            ProcessingOutput(
                output_name="val",
                source="/opt/ml/processing/output/val",
                destination=f"{pipeline_s3}/preprocessing/val",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"{pipeline_s3}/preprocessing/test",
            ),
            ProcessingOutput(
                output_name="preprocessor",
                source="/opt/ml/processing/output/preprocessor",
                destination=f"{pipeline_s3}/preprocessing/preprocessor",
            ),
        ],
    )

    # -------------------------------------------------------------------------
    # Step 2 — Baseline Training
    # -------------------------------------------------------------------------
    step_train = TrainingStep(
        name="BaselineTraining",
        estimator=Estimator(
            image_uri=training_image,
            role=role,
            instance_type=instance_type,
            instance_count=1,
            sagemaker_session=sagemaker_session,
            container_log_level=logging.INFO,
            hyperparameters={
                "mlflow-uri": mlflow_uri,
                "experiment-name": experiment_name,
                "random-state": RANDOM_STATE,
            },
            environment={
                **shared_env,
                "SAGEMAKER_PROGRAM": "train_step.py",
                "S3_BUCKET": s3_bucket,
                "PIPELINE_S3_PREFIX": "sagemaker/pipeline",
            },
        ),
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            ),
            "val": sagemaker.inputs.TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "val"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            ),
        },
    )

    # -------------------------------------------------------------------------
    # Step 3 — Hyperparameter Tuning
    # -------------------------------------------------------------------------
    # HyperparameterTuner is a managed AWS service unavailable in local mode.
    # A TrainingStep is used as a drop-in replacement.
    step_tune = TrainingStep(
        name="HyperparameterTuning",
        estimator=Estimator(
            image_uri=training_image,
            role=role,
            instance_type=instance_type,
            instance_count=1,
            sagemaker_session=sagemaker_session,
            container_log_level=logging.INFO,
            hyperparameters={
                "mlflow-uri": mlflow_uri,
                "experiment-name": experiment_name,
                "n-trials": n_trials,
                "random-state": RANDOM_STATE,
            },
            environment={
                **shared_env,
                "SAGEMAKER_PROGRAM": "tune_step.py",
                "S3_BUCKET": s3_bucket,
                "PIPELINE_S3_PREFIX": "sagemaker/pipeline",
            },
        ),
        depends_on=[step_train],
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            ),
            "val": sagemaker.inputs.TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "val"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            ),
        },
    )

    # -------------------------------------------------------------------------
    # Step 4 — Evaluation
    # -------------------------------------------------------------------------
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="report",
        path="evaluation_report.json",
    )

    step_evaluate = ProcessingStep(
        name="Evaluation",
        processor=ScriptProcessor(
            command=["python"],
            image_uri=training_image,
            role=role,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            env=shared_env,
        ),
        code=get_step_path("evaluate.py"),
        depends_on=[step_tune],
        job_arguments=[
            "--mlflow-uri",
            mlflow_uri,
            "--experiment-name",
            experiment_name,
            "--auc-threshold",
            str(auc_threshold),
        ],
        inputs=[
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/test",
                input_name="test",
            ),
            ProcessingInput(
                source=f"{pipeline_s3}/tuning/tuning_summary.json",
                destination="/opt/ml/processing/input/tuning",
                input_name="tuning",
            ),
            ProcessingInput(
                source=f"{pipeline_s3}/preprocessing/preprocessor/prep_meta.json",
                destination="/opt/ml/processing/input/prep_meta",
                input_name="prep_meta",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="report",
                source="/opt/ml/processing/output/report",
                destination=f"{pipeline_s3}/evaluation",
            ),
        ],
        property_files=[evaluation_report],
    )

    # -------------------------------------------------------------------------
    # Condition — register only if AUC >= threshold, else fail
    # -------------------------------------------------------------------------
    step_condition = ConditionStep(
        name="CheckAUCThreshold",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="test_auc",
                ),
                right=pipeline_auc_threshold,
            )
        ],
        if_steps=[],
        else_steps=[
            FailStep(
                name="ModelFailedThreshold",
                error_message=f"Test AUC below threshold {auc_threshold}. Model not registered.",
            )
        ],
    )

    return Pipeline(
        name="CreditRiskTrainingPipeline",
        parameters=[
            pipeline_ingestion_date,
            pipeline_number_trials,
            pipeline_auc_threshold,
            pipeline_experiment_name,
        ],
        steps=[
            step_preprocess,
            step_train,
            step_tune,
            step_evaluate,
            step_condition,
        ],
        sagemaker_session=sagemaker_session,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit risk SageMaker pipeline"
    )
    parser.add_argument("--mode", default="local", choices=["local", "aws"])
    parser.add_argument("--ingestion-date", default="2026-03-21")
    parser.add_argument("--s3-endpoint", default="http://localstack:4566")
    parser.add_argument("--s3-bucket", default="data-lake")
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="credit_risk_pipeline")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--auc-threshold", type=float, default=0.85)
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
        n_trials=args.n_trials,
        auc_threshold=args.auc_threshold,
        training_image=args.training_image,
        instance_type=instance_type,
        aws_region=args.aws_region,
        s3_endpoint=args.s3_endpoint,
    )

    pipeline.upsert(role_arn=role)

    execution = pipeline.start(
        parameters={
            "IngestionDate": args.ingestion_date,
            "NTrials": args.n_trials,
            "AucThreshold": args.auc_threshold,
            "ExperimentName": args.experiment_name,
        }
    )

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

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main(parse_args())
