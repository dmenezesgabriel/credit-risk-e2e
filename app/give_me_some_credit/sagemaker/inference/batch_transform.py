"""
batch_transform.py — SageMaker ProcessingStep: Batch Inference

Loads the champion model + fitted preprocessor from MLflow, scores
inference features from the gold layer, and writes predictions to S3.

SageMaker mounts:
  Input  "inference"    => /opt/ml/processing/input/inference/
  Input  "eval_report"  => /opt/ml/processing/input/eval_report/
  Output "predictions"  => /opt/ml/processing/output/predictions/

Run locally:
    python batch_transform.py \
        --mlflow-uri http://mlflow:5000 \
        --experiment-name credit_risk_pipeline
"""

import argparse
import json
import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("batch_transform")


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
INPUT_INFERENCE = "/opt/ml/processing/input/inference"
INPUT_EVAL_REPORT = "/opt/ml/processing/input/eval_report"
OUTPUT_PREDICTIONS = "/opt/ml/processing/output/predictions"

# ---------------------------------------------------------------------------
# MLflow registry
# ---------------------------------------------------------------------------
REGISTRY_MODEL_NAME = "credit_risk_champion"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_inference_data(input_path: str) -> pd.DataFrame:
    parquet_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.endswith(".parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_path}")

    df = pd.concat(
        [pd.read_parquet(f) for f in parquet_files],
        ignore_index=True,
    )
    return df.drop(columns=["ingestion_date"], errors="ignore")


def load_evaluation_report(eval_report_dir: str) -> dict:
    report_path = os.path.join(eval_report_dir, "evaluation_report.json")
    with open(report_path) as f:
        return json.load(f)


def load_preprocessor(
    eval_report: dict,
):  # noqa: ANN201 — returns sklearn ColumnTransformer
    """Load the fitted preprocessor from the preprocessing MLflow run.

    The evaluation report contains the champion_run_id. The preprocessor
    is stored in a separate preprocessing run whose run_id is recorded in
    prep_meta.json. However, the batch inference pipeline receives the
    prep_meta directly via an input mount — we load it here.
    """
    prep_meta_path = "/opt/ml/processing/input/prep_meta/prep_meta.json"
    with open(prep_meta_path) as f:
        prep_meta = json.load(f)

    preprocessor = mlflow.sklearn.load_model(
        f"runs:/{prep_meta['run_id']}/preprocessor"
    )
    logger.info(f"Preprocessor loaded from run {prep_meta['run_id']}")
    return preprocessor


def load_champion_model(
    model_stage: str,
):  # noqa: ANN201 — returns sklearn estimator
    model_uri = f"models:/{REGISTRY_MODEL_NAME}/{model_stage}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Champion model loaded from {model_uri}")
    return model


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict(
    model,
    preprocessor,
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    row_ids = df["row_id"].values
    feature_cols = [c for c in df.columns if c != "row_id"]
    X = df[feature_cols]

    X_transformed = preprocessor.transform(X)
    logger.info(
        f"Transformed {len(X):,} rows, {X_transformed.shape[1]} features"
    )

    probabilities = model.predict_proba(X_transformed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "row_id": row_ids,
            "probability": np.round(probabilities, 6),
            "prediction": predictions,
            "threshold_used": threshold,
        }
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "predictions.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_file)
    logger.info(f"Predictions written: {len(df):,} rows => {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    otel_provider = setup_otel_logging("sagemaker-batch-transform")

    logger.info(f"Args: {vars(args)}")

    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)

        # Load inputs
        df = load_inference_data(INPUT_INFERENCE)
        logger.info(
            f"Inference data: {len(df):,} rows, {len(df.columns)} columns"
        )

        eval_report = load_evaluation_report(INPUT_EVAL_REPORT)
        threshold = eval_report["optimal_threshold"]
        logger.info(
            f"Champion: {eval_report['champion_name']} "
            f"(run={eval_report['champion_run_id']}, "
            f"test_auc={eval_report['test_auc']}, threshold={threshold})"
        )

        preprocessor = load_preprocessor(eval_report)
        model = load_champion_model(args.model_stage)

        # Score
        predictions_df = predict(model, preprocessor, df, threshold)

        logger.info(
            f"Prediction summary: "
            f"{predictions_df['prediction'].sum():,} positive / "
            f"{len(predictions_df):,} total "
            f"({predictions_df['prediction'].mean() * 100:.2f}% predicted default rate)"
        )

        # Save
        save_predictions(predictions_df, OUTPUT_PREDICTIONS)

        # Log to MLflow
        with mlflow.start_run(run_name="batch_inference") as run:
            mlflow.log_params(
                {
                    "step": "batch_inference",
                    "model_stage": args.model_stage,
                    "champion_run_id": eval_report["champion_run_id"],
                    "threshold": threshold,
                    "inference_rows": len(predictions_df),
                }
            )
            mlflow.log_metrics(
                {
                    "predicted_default_rate": float(
                        predictions_df["prediction"].mean()
                    ),
                    "mean_probability": float(
                        predictions_df["probability"].mean()
                    ),
                }
            )
            logger.info(f"Inference logged to MLflow run: {run.info.run_id}")

        logger.info("Batch inference complete.")
    finally:
        otel_provider.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit risk batch inference step"
    )
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="credit_risk_pipeline")
    parser.add_argument("--model-stage", default="Staging")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
