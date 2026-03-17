"""
pipeline.py - SageMaker Local Training Pipeline
================================================
Uses a custom Docker image. No ECR auth required.

Build first:
    docker build -t credit-risk-training:latest training/

Run:
    python training/pipeline.py --ingestion-date 2026-03-14
"""

import argparse
import logging
import os

import boto3
import sagemaker
import sagemaker.local.image as sm_image
import yaml
from sagemaker.estimator import Estimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ingestion-date", default="2026-03-14")
parser.add_argument("--s3-endpoint", default="http://localstack:4566")
parser.add_argument("--s3-bucket", default="data-lake")
parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
parser.add_argument("--experiment-name", default="credit_risk_training")
parser.add_argument("--n-trials", type=int, default=50)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--image", default="credit-risk-training:latest")
parser.add_argument("--network", default="mlops-lab_mlops-lab-net")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Monkey-patch _compose to inject project network.
#
# _compose() returns a list of CLI args:
#   ['docker-compose', '-f', '/tmp/.../docker-compose.yaml', 'up', ...]
# The yaml path is the 3rd element (index 2).
# We intercept after the original call, read the yaml file it points to,
# inject the network, and write it back - all before docker-compose reads it.
# This is synchronous and deterministic: _compose() writes the file,
# returns the command, we patch the file, then docker-compose runs.
# ---------------------------------------------------------------------------
_original_compose = sm_image._SageMakerContainer._compose


def _patched_compose(self, *args, **kwargs):
    # Call original - this writes the yaml to disk and returns the command list
    compose_cmd = _original_compose(self, *args, **kwargs)

    # Extract the yaml path from the command list: [..., '-f', '<path>', ...]
    try:
        yaml_path = compose_cmd[compose_cmd.index("-f") + 1]
    except (ValueError, IndexError):
        logger.warning(
            "Could not find -f in compose command - skipping network patch"
        )
        return compose_cmd

    # Read, patch, write back
    try:
        with open(yaml_path, "r") as f:
            compose_data = yaml.safe_load(f)

        project_network = args[0] if args else parser.parse_args().network

        # Add network to every service
        for service in compose_data.get("services", {}).values():
            if "networks" not in service:
                service["networks"] = {}
            service["networks"][project_network] = {}

        # Declare as external network at top level
        if "networks" not in compose_data:
            compose_data["networks"] = {}
        compose_data["networks"][project_network] = {
            "external": True,
            "name": project_network,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(compose_data, f, default_flow_style=False)

        logger.info(f"Injected network '{project_network}' into {yaml_path}")

    except Exception as e:
        logger.warning(f"Network patch failed: {e} - continuing without patch")

    return compose_cmd


# Store network for use inside the patch function
_NETWORK = args.network


def _patched_compose_with_network(self, *args, **kwargs):
    compose_cmd = _original_compose(self, *args, **kwargs)

    try:
        yaml_path = compose_cmd[compose_cmd.index("-f") + 1]
    except (ValueError, IndexError):
        logger.warning("Could not find yaml path in compose command")
        return compose_cmd

    try:
        with open(yaml_path, "r") as f:
            compose_data = yaml.safe_load(f)

        for service in compose_data.get("services", {}).values():
            if "networks" not in service:
                service["networks"] = {}
            service["networks"][_NETWORK] = {}

        if "networks" not in compose_data:
            compose_data["networks"] = {}
        compose_data["networks"][_NETWORK] = {
            "external": True,
            "name": _NETWORK,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(compose_data, f, default_flow_style=False)

        logger.info(
            f"Patched compose: injected network '{_NETWORK}' into {yaml_path}"
        )

    except Exception as e:
        logger.warning(
            f"Network patch failed ({e}) - container may not reach localstack/mlflow"
        )

    return compose_cmd


sm_image._SageMakerContainer._compose = _patched_compose_with_network

# ---------------------------------------------------------------------------
# LocalSession patched for localstack
# ---------------------------------------------------------------------------
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

boto_session = boto3.Session(
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

sagemaker_session = sagemaker.local.LocalSession(boto_session=boto_session)
sagemaker_session.local_mode = True

ls_s3 = boto_session.client("s3", endpoint_url=args.s3_endpoint)
ls_sts = boto_session.client("sts", endpoint_url=args.s3_endpoint)
sagemaker_session._s3_client = ls_s3
sagemaker_session._sts_client = ls_sts
sagemaker_session.default_bucket = lambda: args.s3_bucket

ROLE = "arn:aws:iam::111111111111:role/SageMakerExecutionRole"

# ---------------------------------------------------------------------------
# Estimator with custom image
# ---------------------------------------------------------------------------
estimator = Estimator(
    image_uri=args.image,
    role=ROLE,
    instance_type="local",
    instance_count=1,
    sagemaker_session=sagemaker_session,
    container_log_level=logging.INFO,
    hyperparameters={
        "s3-endpoint": args.s3_endpoint,
        "s3-bucket": args.s3_bucket,
        "ingestion-date": args.ingestion_date,
        "mlflow-uri": args.mlflow_uri,
        "experiment-name": args.experiment_name,
        "n-trials": args.n_trials,
        "random-state": args.random_state,
    },
    environment={
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_ENDPOINT_URL": args.s3_endpoint,
        "MLFLOW_TRACKING_URI": args.mlflow_uri,
    },
)

logger.info(f"Launching training container: {args.image}")
logger.info(f"  ingestion_date : {args.ingestion_date}")
logger.info(f"  mlflow_uri     : {args.mlflow_uri}")
logger.info(f"  n_trials       : {args.n_trials}")
logger.info(f"  network        : {args.network}")

estimator.fit()

logger.info("Training job complete.")
