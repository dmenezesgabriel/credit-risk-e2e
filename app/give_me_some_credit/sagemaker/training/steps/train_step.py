"""
train_step.py — SageMaker TrainingStep: Step 2

Reads preprocessed splits, trains 4 baseline models, logs to MLflow,
uploads results summary to S3 for the tuning step.

SageMaker mounts:
  Input  "train" => /opt/ml/input/data/train/
  Input  "val"   => /opt/ml/input/data/val/
  Output         => /opt/ml/model/

Run locally:
    python train_step.py \
        --mlflow-uri http://localhost:5000 \
        --experiment-name credit_risk_pipeline \
        --s3-bucket data-lake \
        --s3-prefix projects/credit_risk_pipeline/sagemaker/pipeline \
        --s3-endpoint http://localstack:4566 \
        --random-state 42
"""

import argparse
import json
import logging
import os

import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger("train_step")


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
TRAIN_PATH = "/opt/ml/input/data/train"
VAL_PATH = "/opt/ml/input/data/val"

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
TARGET = "serious_dlqin2yrs"
SCALE_POS_WEIGHT = 13.9
CV_SPLITS = 5
BASELINE_S3_KEY = "baseline/baseline_results.json"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data(
    train_path: str,
    val_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_parquet(os.path.join(train_path, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(val_path, "val.parquet"))
    feature_cols = [c for c in train_df.columns if c != TARGET]

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET].values

    logger.info(
        f"Train: {len(X_train):,} | Val: {len(X_val):,} | Features: {len(feature_cols)}"
    )
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_models(random_state: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=SCALE_POS_WEIGHT,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=random_state,
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            scale_pos_weight=SCALE_POS_WEIGHT,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            scale_pos_weight=SCALE_POS_WEIGHT,
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=0,
        ),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def _fit_xgboost(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> tuple:
    cv_model = clone(model)
    cv_model.set_params(early_stopping_rounds=None)
    cv_auc_mean, cv_auc_std = compute_cv_score(
        cv_model, X_train, y_train, random_state
    )
    model.set_params(early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, cv_auc_mean, cv_auc_std


def _fit_default(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> tuple:
    cv_auc_mean, cv_auc_std = compute_cv_score(
        model, X_train, y_train, random_state
    )
    model.fit(X_train, y_train)
    return model, cv_auc_mean, cv_auc_std


def _build_fit_dispatchers() -> dict:
    return {
        "xgboost": _fit_xgboost,
    }


def fit_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> tuple:
    fit_fn = _build_fit_dispatchers().get(name, _fit_default)
    return fit_fn(model, X_train, y_train, X_val, y_val, random_state)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_cv_score(
    model,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> tuple[float, float]:
    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    logger.info(
        f"[CV-{CV_SPLITS}] AUC={scores.mean():.4f} ± {scores.std():.4f}"
    )
    return float(scores.mean()), float(scores.std())


def compute_split_metrics(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict[str, float]:
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    ks = ks_statistic(y, y_prob)
    logger.info(f"[{split_name}] AUC={auc:.4f} KS={ks:.4f}")
    return {
        f"{split_name}_auc_roc": round(auc, 4),
        f"{split_name}_ks": round(ks, 4),
        f"{split_name}_gini": round(2 * auc - 1, 4),
        f"{split_name}_pr_auc": round(average_precision_score(y, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_results(
    results: dict,
    model_dir: str,
    s3_bucket: str,
    s3_prefix: str,
    s3_endpoint: str,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    results_path = os.path.join(model_dir, "baseline_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    s3_key = f"{s3_prefix}/{BASELINE_S3_KEY}"
    s3 = boto3.client("s3", endpoint_url=s3_endpoint)
    s3.upload_file(results_path, s3_bucket, s3_key)
    logger.info(f"Uploaded baseline_results.json => s3://{s3_bucket}/{s3_key}")


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def log_baseline_run(
    model_name: str,
    model,
    cv_auc_mean: float,
    cv_auc_std: float,
    train_metrics: dict,
    val_metrics: dict,
    X_train: np.ndarray,
) -> str:
    with mlflow.start_run(run_name=f"baseline_{model_name}") as run:
        mlflow.log_params(
            {
                "step": "baseline_training",
                "model_name": model_name,
                "train_size": len(X_train),
                "scale_pos_weight": SCALE_POS_WEIGHT,
            }
        )
        mlflow.log_metrics(
            {
                "cv_auc_mean": round(cv_auc_mean, 4),
                "cv_auc_std": round(cv_auc_std, 4),
                **train_metrics,
                **val_metrics,
            }
        )
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    otel_provider = setup_otel_logging("sagemaker-pipeline-train")

    logger.info(f"Args: {vars(args)}")
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)

        X_train, y_train, X_val, y_val = load_data(TRAIN_PATH, VAL_PATH)

        models = build_models(args.random_state)
        results = {}

        for model_name, model in models.items():
            logger.info(f"Training baseline: {model_name}")

            model, cv_auc_mean, cv_auc_std = fit_model(
                model_name,
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                args.random_state,
            )
            train_metrics = compute_split_metrics(
                model, X_train, y_train, "train"
            )
            val_metrics = compute_split_metrics(model, X_val, y_val, "val")

            run_id = log_baseline_run(
                model_name,
                model,
                cv_auc_mean,
                cv_auc_std,
                train_metrics,
                val_metrics,
                X_train,
            )
            results[model_name] = {
                "run_id": run_id,
                "val_auc": val_metrics["val_auc_roc"],
                "val_ks": val_metrics["val_ks"],
            }

        logger.info("Baseline summary:")
        for name, r in sorted(
            results.items(), key=lambda x: x[1]["val_auc"], reverse=True
        ):
            logger.info(
                f"  {name}: val_auc={r['val_auc']:.4f} ks={r['val_ks']:.4f}"
            )

        save_results(
            results,
            args.model_dir,
            args.s3_bucket,
            args.s3_prefix,
            args.s3_endpoint,
        )

        logger.info("Baseline training complete.")
    finally:
        otel_provider.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit risk baseline training step"
    )
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="credit_risk_pipeline")
    parser.add_argument("--s3-bucket", default="data-lake")
    parser.add_argument(
        "--s3-prefix",
        default="projects/credit_risk_pipeline/sagemaker/pipeline",
    )
    parser.add_argument("--s3-endpoint", default="http://localstack:4566")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
