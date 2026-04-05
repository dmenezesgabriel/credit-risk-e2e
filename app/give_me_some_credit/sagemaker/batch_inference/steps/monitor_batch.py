"""
monitor_batch.py — SageMaker ProcessingStep: Batch Monitoring (Step 3)

Runs after Batch Transform to assess data quality, feature drift,
and score distribution. Results are persisted to MLflow and logged
via OTEL for Grafana/Loki visibility.

SageMaker mounts:
  Input  "features"     => /opt/ml/processing/input/features/
  Input  "predictions"  => /opt/ml/processing/input/predictions/
  Input  "model_data"   => /opt/ml/processing/input/model_data/
  Output "report"       => /opt/ml/processing/output/report/

Run locally:
    python monitor_batch.py \\
        --mlflow-uri http://localhost:5000 \\
        --experiment-name give_me_some_credit \\
        --ingestion-date 2026-03-21
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

import mlflow
import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("monitor_batch")

# ---------------------------------------------------------------------------
# SageMaker I/O paths
# ---------------------------------------------------------------------------
FEATURES_DIR = "/opt/ml/processing/input/features"
PREDICTIONS_DIR = "/opt/ml/processing/input/predictions"
MODEL_DATA_DIR = "/opt/ml/processing/input/model_data"
EVALUATION_DIR = "/opt/ml/processing/input/evaluation"
OUTPUT_DIR = "/opt/ml/processing/output/report"


def setup_otel_logging(service_name: str):
    """Configures OTEL logging (Loki) and Console logging (stdout)."""
    provider = LoggerProvider(
        resource=Resource.create(
            {
                "service.name": service_name,
                "service.namespace": "mlops",
            }
        )
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
# Result types
# ---------------------------------------------------------------------------
@dataclass
class DataQualityResult:
    row_count: int
    null_rates: dict[str, float]
    missing_columns: list[str]
    constant_columns: list[str]


@dataclass
class DriftResult:
    per_feature_drift: dict[str, float]
    drifted_features: list[str]
    drift_share: float
    overall_drift_detected: bool


# ---------------------------------------------------------------------------
# Drift adapter — isolates Evidently API
# ---------------------------------------------------------------------------
def compute_data_quality(
    current_df: pd.DataFrame,
    expected_columns: list[str],
) -> DataQualityResult:
    """Assess schema completeness, null rates, and constant columns."""
    missing_columns = [
        col for col in expected_columns if col not in current_df.columns
    ]
    available = [col for col in expected_columns if col in current_df.columns]
    null_rates = {
        col: round(float(current_df[col].isna().mean()), 6)
        for col in available
    }
    constant_columns = [
        col for col in available if current_df[col].nunique(dropna=True) <= 1
    ]

    return DataQualityResult(
        row_count=len(current_df),
        null_rates=null_rates,
        missing_columns=missing_columns,
        constant_columns=constant_columns,
    )


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> DriftResult:
    """Run Evidently DataDriftPreset and return per-feature drift scores."""
    all_features = numeric_features + categorical_features
    shared = [col for col in all_features if col in current_df.columns]
    if not shared:
        return DriftResult(
            per_feature_drift={},
            drifted_features=[],
            drift_share=0.0,
            overall_drift_detected=False,
        )

    column_mapping = ColumnMapping(
        numerical_features=[f for f in numeric_features if f in shared],
        categorical_features=[f for f in categorical_features if f in shared],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[shared],
        current_data=current_df[shared],
        column_mapping=column_mapping,
    )

    result = report.as_dict()
    drift_metrics = result["metrics"][0]["result"]

    per_feature = {}
    drifted = []
    for col_name, col_data in drift_metrics.get(
        "drift_by_columns", {}
    ).items():
        score = col_data.get("drift_score", 0.0)
        per_feature[col_name] = round(float(score), 6)
        if col_data.get("drift_detected", False):
            drifted.append(col_name)

    return DriftResult(
        per_feature_drift=per_feature,
        drifted_features=drifted,
        drift_share=round(
            float(drift_metrics.get("share_of_drifted_columns", 0.0)), 4
        ),
        overall_drift_detected=drift_metrics.get("dataset_drift", False),
    )


# ---------------------------------------------------------------------------
# Metrics adapter — isolates MLflow API
# ---------------------------------------------------------------------------
def log_monitoring_metrics(
    mlflow_uri: str,
    experiment_name: str,
    metrics: dict[str, float],
    artifacts: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log monitoring metrics and optional artifacts to MLflow."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    run_tags = {"step": "batch_monitoring"}
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name="batch_monitoring", tags=run_tags) as run:
        mlflow.log_metrics(metrics)
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)
        logger.info(
            f"Logged {len(metrics)} metrics to MLflow run {run.info.run_id}"
        )
        return run.info.run_id


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_parquet_dir(directory: str) -> pd.DataFrame:
    """Load all parquet files from a directory (recursively) into a single DataFrame."""
    files = []
    for root, _dirs, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".parquet") or f.endswith(".parquet.out"):
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_feature_reference(model_data_dir: str) -> pd.DataFrame | None:
    """Build a reference DataFrame from feature_reference.json stats."""
    ref_path = os.path.join(model_data_dir, "feature_reference.json")
    if not os.path.exists(ref_path):
        logger.warning(
            "feature_reference.json not found — skipping drift check"
        )
        return None

    ref_stats = load_json(ref_path)

    # Build a synthetic reference DataFrame from the stored statistics.
    # Evidently needs actual data rows; we generate them from mean/std.
    rows = 1000
    rng = np.random.default_rng(42)
    ref_data = {}
    for col, stats in ref_stats.items():
        if "mean" in stats and "std" in stats:
            std = max(stats["std"], 1e-9)
            ref_data[col] = rng.normal(loc=stats["mean"], scale=std, size=rows)
        else:
            ref_data[col] = rng.choice([0, 1, 2, 3], size=rows)

    return pd.DataFrame(ref_data)


# ---------------------------------------------------------------------------
# Monitoring checks — dict dispatch
# ---------------------------------------------------------------------------
def run_data_quality_check(ctx: dict) -> dict:
    """Check null rates, missing columns, constant columns."""
    feature_config = ctx["feature_config"]
    expected_columns = feature_config["features"]

    result = compute_data_quality(ctx["features_df"], expected_columns)
    return asdict(result)


def run_feature_drift_check(ctx: dict) -> dict:
    """Compare current features against training reference distribution."""
    reference_df = ctx.get("reference_df")
    if reference_df is None:
        return {"skipped": True, "reason": "no reference data"}

    feature_config = ctx["feature_config"]
    result = compute_feature_drift(
        reference_df=reference_df,
        current_df=ctx["features_df"],
        numeric_features=feature_config["numeric"],
        categorical_features=feature_config["categorical"],
    )
    return asdict(result)


def run_score_distribution_check(ctx: dict) -> dict:
    """Compute prediction score distribution summary."""
    predictions_df = ctx["predictions_df"]
    if "probability" not in predictions_df.columns:
        return {"skipped": True, "reason": "no probability column"}

    probs = predictions_df["probability"]
    threshold = ctx.get("threshold", 0.5)

    predictions = (probs >= threshold).astype(int)
    return {
        "mean_probability": round(float(probs.mean()), 6),
        "std_probability": round(float(probs.std()), 6),
        "p25": round(float(probs.quantile(0.25)), 6),
        "p50": round(float(probs.quantile(0.50)), 6),
        "p75": round(float(probs.quantile(0.75)), 6),
        "pct_predicted_default": round(float(predictions.mean()), 6),
        "record_count": len(probs),
    }


# Register checks — adding a new check = adding one dict entry
MONITORING_CHECKS = {
    "data_quality": run_data_quality_check,
    "feature_drift": run_feature_drift_check,
    "score_distribution": run_score_distribution_check,
}


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------
def build_monitoring_report(
    check_results: dict,
    ingestion_date: str,
    duration_seconds: float,
) -> dict:
    return {
        "ingestion_date": ingestion_date,
        "duration_seconds": round(duration_seconds, 2),
        "checks": check_results,
    }


def save_report(report: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "monitoring_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved monitoring report to {path}")
    return path


# ---------------------------------------------------------------------------
# Metric extraction for MLflow
# ---------------------------------------------------------------------------
def extract_flat_metrics(report: dict) -> dict[str, float]:
    """Extract top-level numeric metrics from the monitoring report."""
    metrics = {"batch_duration_seconds": report["duration_seconds"]}

    quality = report["checks"].get("data_quality", {})
    if "null_rates" in quality:
        null_values = list(quality["null_rates"].values())
        metrics["batch_null_max"] = max(null_values) if null_values else 0.0
        metrics["batch_null_mean"] = (
            sum(null_values) / len(null_values) if null_values else 0.0
        )
    if "row_count" in quality:
        metrics["batch_row_count"] = float(quality["row_count"])

    drift = report["checks"].get("feature_drift", {})
    if "drift_share" in drift:
        metrics["batch_drift_share"] = drift["drift_share"]
        metrics["batch_drift_detected"] = float(
            drift["overall_drift_detected"]
        )
        drift_scores = list(drift.get("per_feature_drift", {}).values())
        metrics["batch_psi_max"] = max(drift_scores) if drift_scores else 0.0

    scores = report["checks"].get("score_distribution", {})
    if "mean_probability" in scores:
        metrics["batch_mean_probability"] = scores["mean_probability"]
        metrics["batch_pct_default"] = scores["pct_predicted_default"]
        metrics["batch_score_std"] = scores["std_probability"]

    return metrics


# ---------------------------------------------------------------------------
# Structured logging for Loki/Grafana
# ---------------------------------------------------------------------------
def log_monitoring_summary(report: dict) -> None:
    """Emit structured log with key monitoring metrics for Loki ingestion."""
    metrics = extract_flat_metrics(report)
    logger.info(
        json.dumps(
            {
                "event": "batch_monitoring_complete",
                "ingestion_date": report["ingestion_date"],
                **metrics,
            }
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    otel_provider = setup_otel_logging("monitor-batch")
    logger.info(f"Args: {vars(args)}")
    start_time = time.time()

    try:
        # Load data
        features_df = load_parquet_dir(FEATURES_DIR)
        predictions_df = load_parquet_dir(PREDICTIONS_DIR)
        logger.info(
            f"Loaded {len(features_df):,} feature rows, "
            f"{len(predictions_df):,} prediction rows"
        )

        # Load model metadata
        eval_report = load_json(
            os.path.join(EVALUATION_DIR, "evaluation_report.json")
        )
        feature_config = load_json(
            os.path.join(MODEL_DATA_DIR, "feature_config.json")
        )
        reference_df = load_feature_reference(MODEL_DATA_DIR)

        # Build check context
        ctx = {
            "features_df": features_df,
            "predictions_df": predictions_df,
            "feature_config": feature_config,
            "reference_df": reference_df,
            "threshold": eval_report.get("optimal_threshold", 0.5),
        }

        # Run all registered checks
        check_results = {
            name: check(ctx) for name, check in MONITORING_CHECKS.items()
        }

        duration = time.time() - start_time
        report = build_monitoring_report(
            check_results, args.ingestion_date, duration
        )

        # Persist
        report_path = save_report(report, OUTPUT_DIR)
        log_monitoring_summary(report)

        flat_metrics = extract_flat_metrics(report)
        log_monitoring_metrics(
            mlflow_uri=args.mlflow_uri,
            experiment_name=args.experiment_name,
            metrics=flat_metrics,
            artifacts={"monitoring": report_path},
            tags={"ingestion_date": args.ingestion_date},
        )

        logger.info(
            f"Monitoring complete: {len(check_results)} checks in {duration:.1f}s"
        )
    finally:
        otel_provider.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference monitoring step"
    )
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="give_me_some_credit")
    parser.add_argument("--ingestion-date", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
