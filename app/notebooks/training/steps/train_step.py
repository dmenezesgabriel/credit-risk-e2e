"""
train_step.py — SageMaker TrainingStep: Step 2
===============================================
Reads preprocessed splits from Step 1 output.
Trains 4 baseline models, logs to MLflow.
Writes best baseline model + all run IDs to /opt/ml/model/.

SageMaker TrainingStep mounts:
  Input  channel "train" → /opt/ml/input/data/train/
  Input  channel "val"   → /opt/ml/input/data/val/
  Output                 → /opt/ml/model/

Hyperparameters passed as CLI args by SageMaker.
"""

import argparse
import json
import logging
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_step")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
parser.add_argument("--experiment-name", default="credit_risk_pipeline")
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument(
    "--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
)
args = parser.parse_args()

RANDOM_STATE = args.random_state
SPW = 13.9
TARGET = "serious_dlqin2yrs"

# ---------------------------------------------------------------------------
# I/O paths — SageMaker TrainingStep convention
# ---------------------------------------------------------------------------
TRAIN_PATH = "/opt/ml/input/data/train"
VAL_PATH = "/opt/ml/input/data/val"
os.makedirs(args.model_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate(model, X, y, split_name):
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    ks = ks_statistic(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    metrics = {
        f"{split_name}_auc_roc": round(auc, 4),
        f"{split_name}_ks": round(ks, 4),
        f"{split_name}_gini": round(2 * auc - 1, 4),
        f"{split_name}_pr_auc": round(pr_auc, 4),
    }
    mlflow.log_metrics(metrics)
    logger.info(f"[{split_name}] AUC={auc:.4f} KS={ks:.4f}")
    return metrics


def cv_score(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(f"[CV-{n_splits}] AUC={scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean()), float(scores.std())


# ---------------------------------------------------------------------------
# Load splits
# ---------------------------------------------------------------------------
logger.info("Loading preprocessed splits...")
train_file = os.path.join(TRAIN_PATH, "train.parquet")
val_file = os.path.join(VAL_PATH, "val.parquet")

train_df = pd.read_parquet(train_file)
val_df = pd.read_parquet(val_file)

feature_cols = [c for c in train_df.columns if c != TARGET]
X_train = train_df[feature_cols].values
y_train = train_df[TARGET].values
X_val = val_df[feature_cols].values
y_val = val_df[TARGET].values

logger.info(
    f"Train: {len(X_train)} | Val: {len(X_val)} | Features: {len(feature_cols)}"
)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    ),
    "xgboost": XGBClassifier(
        scale_pos_weight=SPW,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        verbosity=0,
    ),
    "lightgbm": LGBMClassifier(
        scale_pos_weight=SPW,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=-1,
    ),
    "catboost": CatBoostClassifier(
        scale_pos_weight=SPW,
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=RANDOM_STATE,
        verbose=0,
    ),
}

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(args.mlflow_uri)
mlflow.set_experiment(args.experiment_name)

baseline_results = {}

for model_name, model in MODELS.items():
    logger.info(f"Training baseline: {model_name}")

    with mlflow.start_run(run_name=f"baseline_{model_name}") as run:
        mlflow.log_param("step", "baseline_training")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("scale_pos_weight", SPW)

        # CV with early-stopping-safe clone for XGBoost
        if model_name == "xgboost":
            cv_model = clone(model)
            cv_model.set_params(early_stopping_rounds=None)
            cv_mean, cv_std = cv_score(cv_model, X_train, y_train)
        else:
            cv_mean, cv_std = cv_score(model, X_train, y_train)

        mlflow.log_metric("cv_auc_mean", round(cv_mean, 4))
        mlflow.log_metric("cv_auc_std", round(cv_std, 4))

        # Final fit
        if model_name == "xgboost":
            model.set_params(early_stopping_rounds=20)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        evaluate(model, X_train, y_train, "train")
        val_m = evaluate(model, X_val, y_val, "val")
        mlflow.sklearn.log_model(model, artifact_path="model")

        baseline_results[model_name] = {
            "run_id": run.info.run_id,
            "val_auc": val_m["val_auc_roc"],
            "val_ks": val_m["val_ks"],
        }

# ---------------------------------------------------------------------------
# Write outputs for TuningStep
# Saves all run IDs + val metrics so tune_step knows which runs to extend.
# ---------------------------------------------------------------------------
results_path = os.path.join(args.model_dir, "baseline_results.json")
with open(results_path, "w") as f:
    json.dump(baseline_results, f, indent=2)

# Log summary
logger.info("Baseline summary:")
for name, r in sorted(
    baseline_results.items(), key=lambda x: x[1]["val_auc"], reverse=True
):
    logger.info(f"  {name}: val_auc={r['val_auc']:.4f} ks={r['val_ks']:.4f}")

logger.info("Baseline training complete.")
