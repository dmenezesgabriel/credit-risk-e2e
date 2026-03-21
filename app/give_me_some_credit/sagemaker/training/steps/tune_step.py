"""
tune_step.py — SageMaker TrainingStep: Step 3
==============================================
Reads preprocessed splits + baseline results from Steps 1 and 2.
Runs Optuna tuning on CatBoost, LightGBM, XGBoost.
Writes tuned model + metadata to /opt/ml/model/.

SageMaker TrainingStep mounts:
  Input channel "train"    → /opt/ml/input/data/train/
  Input channel "val"      → /opt/ml/input/data/val/
  Input channel "baseline" → /opt/ml/input/data/baseline/
  Output                   → /opt/ml/model/
"""

import argparse
import json
import logging
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tune_step")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
parser.add_argument("--experiment-name", default="credit_risk_pipeline")
parser.add_argument("--n-trials", type=int, default=10)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument(
    "--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
)
args = parser.parse_args()

RANDOM_STATE = args.random_state
SPW = 13.9
TARGET = "serious_dlqin2yrs"

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
TRAIN_PATH = "/opt/ml/input/data/train"
VAL_PATH = "/opt/ml/input/data/val"
BASELINE_PATH = "/opt/ml/input/data/baseline"
os.makedirs(args.model_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
logger.info("Loading splits and baseline results...")

train_df = pd.read_parquet(os.path.join(TRAIN_PATH, "train.parquet"))
val_df = pd.read_parquet(os.path.join(VAL_PATH, "val.parquet"))

feature_cols = [c for c in train_df.columns if c != TARGET]
X_train = train_df[feature_cols].values
y_train = train_df[TARGET].values
X_val = val_df[feature_cols].values
y_val = val_df[TARGET].values

with open(os.path.join(BASELINE_PATH, "baseline_results.json")) as f:
    baseline_results = json.load(f)

logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def make_objective(model_name):
    def objective(trial):
        if model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 600),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.15, log=True
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 1.0
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", 0.0, 2.0
                ),
                "border_count": trial.suggest_int("border_count", 32, 128),
                "scale_pos_weight": SPW,
                "random_seed": RANDOM_STATE,
                "verbose": 0,
            }
            m = CatBoostClassifier(**params)
            m.fit(X_train, y_train)
        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.15, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 10, 100
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-4, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-4, 10.0, log=True
                ),
                "scale_pos_weight": SPW,
                "random_state": RANDOM_STATE,
                "verbosity": -1,
            }
            m = LGBMClassifier(**params)
            m.fit(X_train, y_train)
        else:  # xgboost
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.15, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", 5, 50
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-4, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-4, 10.0, log=True
                ),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "scale_pos_weight": SPW,
                "eval_metric": "auc",
                "early_stopping_rounds": 30,
                "random_state": RANDOM_STATE,
                "verbosity": 0,
            }
            m = XGBClassifier(**params)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    return objective


# ---------------------------------------------------------------------------
# Tuning loop
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(args.mlflow_uri)
mlflow.set_experiment(args.experiment_name)

tuning_results = {}

for model_name in ["catboost", "lightgbm", "xgboost"]:
    baseline_auc = baseline_results[model_name]["val_auc"]
    logger.info(
        f"Tuning {model_name} ({args.n_trials} trials) baseline={baseline_auc:.4f}"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=f"credit_risk_{model_name}",
    )
    study.optimize(make_objective(model_name), n_trials=args.n_trials)

    best_params = study.best_params
    best_auc = study.best_value
    improvement = best_auc - baseline_auc
    logger.info(
        f"  {model_name}: {baseline_auc:.4f} → {best_auc:.4f} ({improvement:+.4f})"
    )

    # Retrain final model with best params
    if model_name == "catboost":
        final = CatBoostClassifier(**best_params, verbose=0)
        final.fit(X_train, y_train)
    elif model_name == "lightgbm":
        final = LGBMClassifier(**best_params, verbosity=-1)
        final.fit(X_train, y_train)
    else:
        final = XGBClassifier(**best_params, verbosity=0)
        final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_auc = roc_auc_score(y_val, final.predict_proba(X_val)[:, 1])
    val_ks = ks_statistic(y_val, final.predict_proba(X_val)[:, 1])

    with mlflow.start_run(run_name=f"tuned_{model_name}") as run:
        mlflow.log_param("step", "tuning")
        mlflow.log_param("model_name", f"{model_name}_tuned")
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("baseline_auc", round(baseline_auc, 4))
        mlflow.log_params(best_params)
        mlflow.log_metric("val_auc_roc", round(val_auc, 4))
        mlflow.log_metric("val_ks", round(val_ks, 4))
        mlflow.log_metric("improvement_vs_baseline", round(improvement, 4))
        mlflow.sklearn.log_model(final, artifact_path="model")

        tuning_results[model_name] = {
            "run_id": run.info.run_id,
            "val_auc": round(val_auc, 4),
            "val_ks": round(val_ks, 4),
            "best_params": best_params,
        }

# ---------------------------------------------------------------------------
# Write champion model to model dir
# EvaluateStep reads from here to run final test set evaluation.
# ---------------------------------------------------------------------------
champion_name = max(tuning_results, key=lambda x: tuning_results[x]["val_auc"])
champion_meta = tuning_results[champion_name]

# Refit champion one final time (same params, consistent artifact)
best_params = champion_meta["best_params"]
if champion_name == "catboost":
    champion_model = CatBoostClassifier(**best_params, verbose=0)
    champion_model.fit(X_train, y_train)
elif champion_name == "lightgbm":
    champion_model = LGBMClassifier(**best_params, verbosity=-1)
    champion_model.fit(X_train, y_train)
else:
    champion_model = XGBClassifier(**best_params, verbosity=0)
    champion_model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
    )

model_path = os.path.join(args.model_dir, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(champion_model, f)

tuning_summary = {
    "champion_name": champion_name,
    "champion_run_id": champion_meta["run_id"],
    "val_auc": champion_meta["val_auc"],
    "val_ks": champion_meta["val_ks"],
    "best_params": best_params,
    "all_results": tuning_results,
}
summary_path = os.path.join(args.model_dir, "tuning_summary.json")
with open(summary_path, "w") as f:
    json.dump(tuning_summary, f, indent=2)

logger.info(
    f"Champion: {champion_name} val_auc={champion_meta['val_auc']:.4f}"
)
logger.info("Tuning complete.")
