"""
evaluate.py — SageMaker ProcessingStep: Step 4

Loads champion model + test set, computes metrics, conditionally registers
the model in MLflow Model Registry, and writes the evaluation report.

SageMaker mounts:
  Input  "test"      => /opt/ml/processing/input/test/
  Input  "tuning"    => /opt/ml/processing/input/tuning/
  Input  "prep_meta" => /opt/ml/processing/input/prep_meta/
  Output "report"    => /opt/ml/processing/output/report/

The ConditionStep in sm_pipeline.py reads evaluation_report.json and
branches: register if test_auc >= threshold, else fail.

Run locally:
    python evaluate.py \
        --mlflow-uri http://localhost:5000 \
        --experiment-name give_me_some_credit \
        --auc-threshold 0.85
"""

import argparse
import json
import logging
import os
import time

import mlflow
import mlflow.exceptions
import mlflow.sklearn
import numpy as np
import pandas as pd
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("evaluate")


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
# SageMaker I/O paths
# ---------------------------------------------------------------------------
TEST_PATH = "/opt/ml/processing/input/test"
TUNING_DIR = "/opt/ml/processing/input/tuning"
PREP_META_DIR = "/opt/ml/processing/input/prep_meta"
OUTPUT_PATH = "/opt/ml/processing/output/report"

# ---------------------------------------------------------------------------
# Threshold cost parameters
# ---------------------------------------------------------------------------
COST_FN = 10
COST_FP = 1
THRESHOLD_SEARCH_GRID = np.linspace(0.01, 0.99, 200)

# ---------------------------------------------------------------------------
# MLflow registry
# ---------------------------------------------------------------------------
REGISTRY_MODEL_NAME = "give_me_some_credit_champion"
REGISTRY_WAIT_RETRIES = 30
REGISTRY_WAIT_SECONDS = 2


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_test_data(test_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(os.path.join(test_path, "test.parquet"))
    feature_cols = [c for c in df.columns if c != "serious_dlqin2yrs"]
    return df[feature_cols].values, df["serious_dlqin2yrs"].values


def load_tuning_results(tuning_dir: str) -> tuple:
    with open(os.path.join(tuning_dir, "tuning_summary.json")) as f:
        summary = json.load(f)
    model = mlflow.sklearn.load_model(
        f"runs:/{summary['champion_run_id']}/model"
    )
    logger.info(
        f"Loaded champion: {summary['champion_name']} from run {summary['champion_run_id']}"
    )
    return model, summary


def assert_preprocessor_accessible(prep_meta_dir: str) -> None:
    with open(os.path.join(prep_meta_dir, "prep_meta.json")) as f:
        prep_meta = json.load(f)
    mlflow.sklearn.load_model(f"runs:/{prep_meta['run_id']}/preprocessor")
    logger.info(f"Preprocessor accessible from run {prep_meta['run_id']}")


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    costs = []
    for t in THRESHOLD_SEARCH_GRID:
        tn, fp, fn, tp = confusion_matrix(
            y_true, (y_prob >= t).astype(int)
        ).ravel()
        costs.append(cost_fn * fn + cost_fp * fp)
    return float(THRESHOLD_SEARCH_GRID[np.argmin(costs)])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_prob)
    return {
        "test_auc": round(auc, 4),
        "test_ks": round(ks_statistic(y_true, y_prob), 4),
        "test_gini": round(2 * auc - 1, 4),
        "test_pr_auc": round(average_precision_score(y_true, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
def wait_until_ready(client, version: str) -> None:
    for _ in range(REGISTRY_WAIT_RETRIES):
        mv = client.get_model_version(REGISTRY_MODEL_NAME, version)
        if mv.status == "READY":
            return
        time.sleep(REGISTRY_WAIT_SECONDS)


def register_champion(run_id: str) -> str:
    client = mlflow.tracking.MlflowClient()

    try:
        client.create_registered_model(REGISTRY_MODEL_NAME)
        logger.info(f"Created registered model: {REGISTRY_MODEL_NAME}")
    except mlflow.exceptions.MlflowException:
        logger.info(f"Registered model already exists: {REGISTRY_MODEL_NAME}")

    mv = client.create_model_version(
        name=REGISTRY_MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    wait_until_ready(client, mv.version)
    client.transition_model_version_stage(
        name=REGISTRY_MODEL_NAME,
        version=mv.version,
        stage="Staging",
    )
    logger.info(f"Registered: {REGISTRY_MODEL_NAME} v{mv.version} => Staging")
    return mv.version


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_report(report: dict, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    report_path = os.path.join(output_path, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report written: {report_path}")


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def update_champion_run_metrics(
    run_id: str, metrics: dict[str, float]
) -> None:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "test_auc_roc": metrics["test_auc"],
                "test_ks": metrics["test_ks"],
                "test_gini": metrics["test_gini"],
                "test_pr_auc": metrics["test_pr_auc"],
                "optimal_threshold_cost": metrics["optimal_threshold"],
            }
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    otel_provider = setup_otel_logging("sagemaker-pipeline-evaluate")

    logger.info(f"Args: {vars(args)}")

    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)

        X_test, y_test = load_test_data(TEST_PATH)
        model, tuning_summary = load_tuning_results(TUNING_DIR)
        assert_preprocessor_accessible(PREP_META_DIR)

        logger.info(
            f"Test set: {len(y_test):,} rows | default rate: {y_test.mean() * 100:.2f}%"
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(y_test, y_prob)
        optimal_threshold = find_optimal_threshold(y_test, y_prob)

        logger.info(
            f"TEST AUC={metrics['test_auc']:.4f} "
            f"KS={metrics['test_ks']:.4f} "
            f"Gini={metrics['test_gini']:.4f} "
            f"Optimal threshold={optimal_threshold:.4f}"
        )

        update_champion_run_metrics(
            tuning_summary["champion_run_id"],
            {**metrics, "optimal_threshold": optimal_threshold},
        )

        passes_threshold = bool(metrics["test_auc"] >= args.auc_threshold)
        registered_version = (
            register_champion(tuning_summary["champion_run_id"])
            if passes_threshold
            else None
        )

        logger.info(
            f"AUC {metrics['test_auc']:.4f} >= threshold {args.auc_threshold}: {passes_threshold}"
        )

        save_report(
            report={
                "champion_name": tuning_summary["champion_name"],
                "champion_run_id": tuning_summary["champion_run_id"],
                "val_auc": tuning_summary["val_auc"],
                "test_auc": metrics["test_auc"],
                "test_ks": metrics["test_ks"],
                "test_gini": metrics["test_gini"],
                "test_pr_auc": metrics["test_pr_auc"],
                "optimal_threshold": optimal_threshold,
                "auc_threshold": args.auc_threshold,
                "passes_threshold": passes_threshold,
                "registered_version": registered_version,
            },
            output_path=OUTPUT_PATH,
        )

        logger.info("Evaluation complete.")
    finally:
        otel_provider.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit risk evaluation step")
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="give_me_some_credit")
    parser.add_argument("--auc-threshold", type=float, default=0.85)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
