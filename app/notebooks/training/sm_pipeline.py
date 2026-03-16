"""
sm_pipeline.py — SageMaker Pipeline Definition
===============================================
Defines a 4-step ML pipeline:

  Step 1 ProcessingStep  preprocess   → splits + preprocessor.pkl → S3
  Step 2 TrainingStep    train_step   → baseline models → S3
  Step 3 TrainingStep    tune_step    → tuned champion → S3
  Step 4 ProcessingStep  evaluate     → test metrics + conditional register

Each step runs in its own container with its own image.
Step outputs are wired as S3 URIs between steps — no hardcoded paths.

Local mode: identical to real AWS except session and instance_type.
To move to production: change LocalSession → Session, "local" → "ml.m5.xlarge".

Build images first:
    docker build -t credit-risk-training:latest   -f training/Dockerfile training/
    docker build -t credit-risk-processing:latest -f training/Dockerfile.processing training/

Run:
    python training/sm_pipeline.py --ingestion-date 2026-03-14
"""

import argparse
import logging
import os

import boto3
import sagemaker
import sagemaker.local.image as sm_image
import yaml
from sagemaker.estimator import Estimator
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
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
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sm_pipeline")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ingestion-date", default="2026-03-14")
parser.add_argument("--s3-endpoint", default="http://localstack:4566")
parser.add_argument("--s3-bucket", default="data-lake")
parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
parser.add_argument("--experiment-name", default="credit_risk_pipeline")
parser.add_argument("--n-trials", type=int, default=50)
parser.add_argument("--auc-threshold", type=float, default=0.85)
parser.add_argument("--training-image", default="credit-risk-training:latest")
parser.add_argument("--processing-image", default="credit-risk-processing:latest")
parser.add_argument("--network", default="credit-risk-e2e_credit-risk-net")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Network patch — same approach as pipeline.py
# Injects project network into SageMaker's generated compose file
# so containers can resolve localstack + mlflow hostnames.
# ---------------------------------------------------------------------------
_NETWORK = args.network
_original_compose = sm_image._SageMakerContainer._compose


def _patched_compose(self, *a, **kw):
    compose_cmd = _original_compose(self, *a, **kw)
    try:
        yaml_path = compose_cmd[compose_cmd.index("-f") + 1]
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        for svc in data.get("services", {}).values():
            svc.setdefault("networks", {})[_NETWORK] = {}
        data.setdefault("networks", {})[_NETWORK] = {"external": True, "name": _NETWORK}
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"Injected network '{_NETWORK}' into {yaml_path}")
    except Exception as e:
        logger.warning(f"Network patch failed: {e}")
    return compose_cmd


sm_image._SageMakerContainer._compose = _patched_compose

# ---------------------------------------------------------------------------
# SageMaker LocalSession
# ---------------------------------------------------------------------------
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

boto_session = boto3.Session(
    aws_access_key_id="test", aws_secret_access_key="test", region_name="us-east-1"
)
sagemaker_session = sagemaker.local.LocalSession(boto_session=boto_session)
sagemaker_session.local_mode = True

ls_s3 = boto_session.client("s3", endpoint_url=args.s3_endpoint)
ls_sts = boto_session.client("sts", endpoint_url=args.s3_endpoint)
sagemaker_session._s3_client = ls_s3
sagemaker_session._sts_client = ls_sts
sagemaker_session.default_bucket = lambda: args.s3_bucket

ROLE = "arn:aws:iam::111111111111:role/SageMakerExecutionRole"
GOLD_S3 = f"s3://{args.s3_bucket}/gold/credit_risk/features/ingestion_date={args.ingestion_date}"
PIPELINE_S3 = f"s3://{args.s3_bucket}/sagemaker/pipeline"

# Shared environment injected into every container
SHARED_ENV = {
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_ENDPOINT_URL": args.s3_endpoint,
    "MLFLOW_TRACKING_URI": args.mlflow_uri,
    "GIT_PYTHON_REFRESH": "quiet",
}

# ---------------------------------------------------------------------------
# Pipeline parameters — can be overridden at pipeline.start() time
# ---------------------------------------------------------------------------
p_ingestion_date = ParameterString("IngestionDate", default_value=args.ingestion_date)
p_n_trials = ParameterInteger("NTrials", default_value=args.n_trials)
p_auc_threshold = ParameterFloat("AucThreshold", default_value=args.auc_threshold)
p_experiment_name = ParameterString(
    "ExperimentName", default_value=args.experiment_name
)

# ---------------------------------------------------------------------------
# Step 1 — Preprocessing (ProcessingStep)
# Uses lightweight processing image.
# Reads gold from S3, outputs split parquets + preprocessor.pkl
# ---------------------------------------------------------------------------
preprocessor_processor = ScriptProcessor(
    command=["python"],
    image_uri=args.processing_image,
    role=ROLE,
    instance_count=1,
    instance_type="local",
    sagemaker_session=sagemaker_session,
    env=SHARED_ENV,
)

step_preprocess = ProcessingStep(
    name="Preprocessing",
    processor=preprocessor_processor,
    code="/app/notebooks/training/steps/preprocess.py",
    inputs=[
        ProcessingInput(
            source=GOLD_S3,
            destination="/opt/ml/processing/input/gold",
            input_name="gold",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"{PIPELINE_S3}/preprocessing/train",
        ),
        ProcessingOutput(
            output_name="val",
            source="/opt/ml/processing/output/val",
            destination=f"{PIPELINE_S3}/preprocessing/val",
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=f"{PIPELINE_S3}/preprocessing/test",
        ),
        ProcessingOutput(
            output_name="preprocessor",
            source="/opt/ml/processing/output/preprocessor",
            destination=f"{PIPELINE_S3}/preprocessing/preprocessor",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Step 2 — Baseline Training (TrainingStep)
# Uses heavy training image with all model libraries.
# Reads splits from Step 1 output URIs.
# ---------------------------------------------------------------------------
baseline_estimator = Estimator(
    image_uri=args.training_image,
    role=ROLE,
    instance_type="local",
    instance_count=1,
    sagemaker_session=sagemaker_session,
    container_log_level=logging.INFO,
    hyperparameters={
        "mlflow-uri": args.mlflow_uri,
        "experiment-name": args.experiment_name,
        "random-state": 42,
    },
    environment={**SHARED_ENV, "SAGEMAKER_PROGRAM": "train_step.py"},
)

step_train = TrainingStep(
    name="BaselineTraining",
    estimator=baseline_estimator,
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

# ---------------------------------------------------------------------------
# Step 3 — Tuning (TrainingStep)
# Reads splits from Step 1 + baseline results from Step 2.
# ---------------------------------------------------------------------------
tuning_estimator = Estimator(
    image_uri=args.training_image,
    role=ROLE,
    instance_type="local",
    instance_count=1,
    sagemaker_session=sagemaker_session,
    container_log_level=logging.INFO,
    hyperparameters={
        "mlflow-uri": args.mlflow_uri,
        "experiment-name": args.experiment_name,
        "n-trials": args.n_trials,
        "random-state": 42,
    },
    environment={**SHARED_ENV, "SAGEMAKER_PROGRAM": "tune_step.py"},
)

step_tune = TrainingStep(
    name="HyperparameterTuning",
    estimator=tuning_estimator,
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
        "baseline": sagemaker.inputs.TrainingInput(
            s3_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            content_type="application/x-parquet",
        ),
    },
)

# ---------------------------------------------------------------------------
# Step 4 — Evaluation + Conditional Register (ProcessingStep + ConditionStep)
# ---------------------------------------------------------------------------
evaluate_processor = ScriptProcessor(
    command=["python"],
    image_uri=args.processing_image,
    role=ROLE,
    instance_count=1,
    instance_type="local",
    sagemaker_session=sagemaker_session,
    env={
        **SHARED_ENV,
        "EXPERIMENT_NAME": args.experiment_name,
        "AUC_THRESHOLD": str(args.auc_threshold),
    },
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="report",
    path="evaluation_report.json",
)

step_evaluate = ProcessingStep(
    name="Evaluation",
    processor=evaluate_processor,
    code="/app/notebooks/training/steps/evaluate.py",
    inputs=[
        ProcessingInput(
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/input/test",
            input_name="test",
        ),
        ProcessingInput(
            source=step_tune.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/input/model",
            input_name="model",
        ),
        ProcessingInput(
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                "preprocessor"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/input/preprocessor",
            input_name="prep",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="report",
            source="/opt/ml/processing/output/report",
            destination=f"{PIPELINE_S3}/evaluation",
        ),
    ],
    property_files=[evaluation_report],
)

# ConditionStep — branches on test_auc from evaluation_report.json
condition_auc = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_evaluate.name,
        property_file=evaluation_report,
        json_path="test_auc",
    ),
    right=p_auc_threshold,
)

step_fail = FailStep(
    name="ModelFailedThreshold",
    error_message=f"Test AUC below threshold {args.auc_threshold}. Model not registered.",
)

step_condition = ConditionStep(
    name="CheckAUCThreshold",
    conditions=[condition_auc],
    if_steps=[],  # registration handled inside evaluate.py (mlflow)
    else_steps=[step_fail],
)

# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------
pipeline = Pipeline(
    name="CreditRiskTrainingPipeline",
    parameters=[p_ingestion_date, p_n_trials, p_auc_threshold, p_experiment_name],
    steps=[step_preprocess, step_train, step_tune, step_evaluate, step_condition],
    sagemaker_session=sagemaker_session,
)

logger.info("Starting CreditRiskTrainingPipeline...")
logger.info(f"  ingestion_date : {args.ingestion_date}")
logger.info(f"  n_trials       : {args.n_trials}")
logger.info(f"  auc_threshold  : {args.auc_threshold}")
logger.info(f"  mlflow_uri     : {args.mlflow_uri}")

pipeline.upsert(role_arn=ROLE)
execution = pipeline.start(
    parameters={
        "IngestionDate": args.ingestion_date,
        "NTrials": args.n_trials,
        "AucThreshold": args.auc_threshold,
        "ExperimentName": args.experiment_name,
    }
)

# LocalPipelineExecution has no .arn — that is a real AWS concept.
# Use execution.name for local mode identification.
exec_id = getattr(execution, "name", None) or getattr(
    execution, "_execution_id", "local"
)
logger.info(f"Pipeline execution started: {exec_id}")

# _LocalPipelineExecution is synchronous — pipeline.start() blocks until complete.
# .wait() does not exist on local executions (it is a real AWS polling method).
# Print step summary from the execution object directly.
try:
    steps = execution.list_steps()
    logger.info("Pipeline step summary:")
    for step in steps:
        name = step.get("StepName", "?")
        status = step.get("StepStatus", "?")
        fail = step.get("FailureReason", "")
        suffix = f" — {fail}" if fail else ""
        logger.info(f"  {name}: {status}{suffix}")
except Exception as e:
    logger.info(f"Could not retrieve step summary: {e}")

logger.info("Pipeline complete.")
