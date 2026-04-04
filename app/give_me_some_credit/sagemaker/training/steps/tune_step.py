"""
tune_step.py — SageMaker TrainingStep: Step 3

Reads preprocessed splits + baseline results from S3, runs Optuna tuning
on CatBoost, LightGBM, XGBoost, selects champion, uploads tuning summary.

SageMaker mounts:
  Input  "train" => /opt/ml/input/data/train/
  Input  "val"   => /opt/ml/input/data/val/
  Output         => /opt/ml/model/

Note: HyperparameterTuner is a managed AWS service unavailable in local mode.
      This TrainingStep is used as a drop-in replacement.

Run locally:
    python tune_step.py \
        --mlflow-uri http://localhost:5000 \
        --experiment-name give_me_some_credit \
        --s3-bucket data-lake \
        --s3-prefix projects/give_me_some_credit/sagemaker/pipeline \
        --s3-endpoint http://localstack:4566 \
        --n-trials 10 \
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
import optuna
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
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

logger = logging.getLogger("tune_step")


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


optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# SageMaker I/O paths
# ---------------------------------------------------------------------------
TRAIN_PATH = "/opt/ml/input/data/train"
VAL_PATH = "/opt/ml/input/data/val"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
TARGET = "serious_dlqin2yrs"
SCALE_POS_WEIGHT = 13.9
BASELINE_S3_KEY = "baseline/baseline_results.json"
TUNING_S3_KEY = "tuning/tuning_summary.json"


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


def load_baseline_results(
    s3_bucket: str, s3_prefix: str, s3_endpoint: str
) -> dict:
    s3_key = f"{s3_prefix}/{BASELINE_S3_KEY}"
    s3 = boto3.client("s3", endpoint_url=s3_endpoint)
    response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    results = json.loads(response["Body"].read())
    logger.info(f"Loaded baseline results for: {list(results.keys())}")
    return results


# ---------------------------------------------------------------------------
# Tune
# ---------------------------------------------------------------------------
def _catboost_param_space(trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 200, 600),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.0, 1.0
        ),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
    }


def _lightgbm_param_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def _xgboost_param_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _build_param_spaces() -> dict:
    return {
        "catboost": _catboost_param_space,
        "lightgbm": _lightgbm_param_space,
        "xgboost": _xgboost_param_space,
    }


def _build_model_constructors() -> dict:
    return {
        "catboost": lambda params, random_state: CatBoostClassifier(
            **params,
            scale_pos_weight=SCALE_POS_WEIGHT,
            random_seed=random_state,
            verbose=0,
        ),
        "lightgbm": lambda params, random_state: LGBMClassifier(
            **params,
            scale_pos_weight=SCALE_POS_WEIGHT,
            random_state=random_state,
            verbosity=-1,
        ),
        "xgboost": lambda params, random_state: XGBClassifier(
            **params,
            scale_pos_weight=SCALE_POS_WEIGHT,
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=random_state,
            verbosity=0,
        ),
    }


def _build_objective(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
):
    param_space_fn = _build_param_spaces()[model_name]
    constructor = _build_model_constructors()[model_name]

    def objective(trial):
        params = param_space_fn(trial)
        model = constructor(params, random_state)

        fit_kwargs = (
            {"eval_set": [(X_val, y_val)], "verbose": False}
            if model_name == "xgboost"
            else {}
        )
        model.fit(X_train, y_train, **fit_kwargs)

        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    return objective


def run_study(
    model_name: str,
    n_trials: int,
    random_state: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> optuna.Study:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=f"credit_risk_{model_name}",
    )
    study.optimize(
        _build_objective(
            model_name, X_train, y_train, X_val, y_val, random_state
        ),
        n_trials=n_trials,
    )
    return study


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def refit_and_evaluate(
    model_name: str,
    best_params: dict,
    random_state: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple:
    constructor = _build_model_constructors()[model_name]
    model = constructor(best_params, random_state)

    fit_kwargs = (
        {"eval_set": [(X_val, y_val)], "verbose": False}
        if model_name == "xgboost"
        else {}
    )
    model.fit(X_train, y_train, **fit_kwargs)

    y_prob = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_prob)
    val_ks = ks_statistic(y_val, y_prob)

    return model, round(val_auc, 4), round(val_ks, 4)


def select_champion(tuning_results: dict) -> tuple[str, dict]:
    champion_name = max(
        tuning_results, key=lambda k: tuning_results[k]["val_auc"]
    )
    return champion_name, tuning_results[champion_name]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_tuning_summary(
    summary: dict,
    model_dir: str,
    s3_bucket: str,
    s3_prefix: str,
    s3_endpoint: str,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    summary_path = os.path.join(model_dir, "tuning_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    s3_key = f"{s3_prefix}/{TUNING_S3_KEY}"
    s3 = boto3.client("s3", endpoint_url=s3_endpoint)
    s3.upload_file(summary_path, s3_bucket, s3_key)
    logger.info(f"Uploaded tuning_summary.json => s3://{s3_bucket}/{s3_key}")


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def log_tuning_run(
    model_name: str,
    best_params: dict,
    baseline_auc: float,
    val_auc: float,
    val_ks: float,
    n_trials: int,
    model,
) -> str:
    with mlflow.start_run(run_name=f"tuned_{model_name}") as run:
        mlflow.log_params(
            {
                "step": "tuning",
                "model_name": f"{model_name}_tuned",
                "n_trials": n_trials,
                "baseline_auc": round(baseline_auc, 4),
                **best_params,
            }
        )
        mlflow.log_metrics(
            {
                "val_auc_roc": val_auc,
                "val_ks": val_ks,
                "improvement_vs_baseline": round(val_auc - baseline_auc, 4),
            }
        )
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    otel_provider = setup_otel_logging("sagemaker-pipeline-tune")

    logger.info(f"Args: {vars(args)}")
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)

        X_train, y_train, X_val, y_val = load_data(TRAIN_PATH, VAL_PATH)
        baseline_results = load_baseline_results(
            args.s3_bucket, args.s3_prefix, args.s3_endpoint
        )

        tuning_results = {}

        for model_name in _build_param_spaces():
            baseline_auc = baseline_results[model_name]["val_auc"]
            logger.info(
                f"Tuning {model_name} ({args.n_trials} trials) | baseline AUC={baseline_auc:.4f}"
            )

            study = run_study(
                model_name,
                args.n_trials,
                args.random_state,
                X_train,
                y_train,
                X_val,
                y_val,
            )
            logger.info(
                f"  {model_name}: {baseline_auc:.4f} => {study.best_value:.4f} "
                f"({study.best_value - baseline_auc:+.4f})"
            )

            model, val_auc, val_ks = refit_and_evaluate(
                model_name,
                study.best_params,
                args.random_state,
                X_train,
                y_train,
                X_val,
                y_val,
            )
            run_id = log_tuning_run(
                model_name,
                study.best_params,
                baseline_auc,
                val_auc,
                val_ks,
                args.n_trials,
                model,
            )
            tuning_results[model_name] = {
                "run_id": run_id,
                "val_auc": val_auc,
                "val_ks": val_ks,
                "best_params": study.best_params,
            }

        champion_name, champion_meta = select_champion(tuning_results)
        logger.info(
            f"Champion: {champion_name} val_auc={champion_meta['val_auc']:.4f}"
        )

        save_tuning_summary(
            summary={
                "champion_name": champion_name,
                "champion_run_id": champion_meta["run_id"],
                "val_auc": champion_meta["val_auc"],
                "val_ks": champion_meta["val_ks"],
                "best_params": champion_meta["best_params"],
                "all_results": tuning_results,
            },
            model_dir=args.model_dir,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_endpoint=args.s3_endpoint,
        )

        logger.info("Tuning complete.")

    finally:
        otel_provider.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit risk hyperparameter tuning step"
    )
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="give_me_some_credit")
    parser.add_argument("--s3-bucket", default="data-lake")
    parser.add_argument(
        "--s3-prefix",
        default="projects/give_me_some_credit/sagemaker/pipeline",
    )
    parser.add_argument("--s3-endpoint", default="http://localstack:4566")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
