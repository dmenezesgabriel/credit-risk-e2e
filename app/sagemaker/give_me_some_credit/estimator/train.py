"""
train.py - Credit Risk Training Script
=======================================
Runs inside a SageMaker training container (local mode or real AWS).
Reproduces the full notebook flow:
  1. Read gold parquet from S3
  2. Stratified 70/15/15 split
  3. Fit preprocessor on train only
  4. Train LogReg, XGBoost, LightGBM, CatBoost
  5. Optuna tuning on top-3 models
  6. Evaluate on val + test
  7. Log everything to MLflow
  8. Register champion in MLflow Model Registry
  9. Save model + preprocessor to /opt/ml/model/

SageMaker environment variables used:
  SM_MODEL_DIR         → where to write model artifacts (/opt/ml/model)
  SM_CHANNEL_TRAIN     → not used (we read directly from S3)
  SM_HP_*              → hyperparameters passed from pipeline.py

Called by pipeline.py via sagemaker.sklearn.SKLearn estimator.
"""

import argparse
import io
import json
import logging
import os
import pickle
import time
import warnings

import boto3
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")  # non-interactive backend inside container
import matplotlib.pyplot as plt
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train")

# ---------------------------------------------------------------------------
# SageMaker injects hyperparameters as CLI args - parse them here.
# Defaults match the validated notebook values so the script is runnable
# standalone without pipeline.py.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--s3-endpoint", default="http://localstack:4566")
parser.add_argument("--s3-bucket", default="data-lake")
parser.add_argument("--gold-prefix", default="gold/credit_risk/features/")
parser.add_argument("--ingestion-date", default="2026-03-14")
parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
parser.add_argument("--experiment-name", default="credit_risk_training")
parser.add_argument("--n-trials", type=int, default=50)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--cost-fn", type=int, default=10)
parser.add_argument("--cost-fp", type=int, default=1)
parser.add_argument(
    "--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
)
args = parser.parse_args()

RANDOM_STATE = args.random_state
SPW = 13.9  # scale_pos_weight from EDA (neg/pos = 13.9)
TARGET = "serious_dlqin2yrs"

NUMERIC_FEATURES = [
    "revolving_utilization_of_unsecured_lines",
    "age",
    "number_of_time30_59_days_past_due_not_worse",
    "debt_ratio",
    "monthly_income",
    "number_of_open_credit_lines_and_loans",
    "number_of_times90_days_late",
    "number_real_estate_loans_or_lines",
    "number_of_time60_89_days_past_due_not_worse",
    "number_of_dependents",
    "monthly_income_is_missing",
    "number_of_dependents_is_missing",
    "delinquency_score",
    "debt_to_income_ratio",
    "unsecured_to_total_lines_ratio",
    "has_any_delinquency",
]
CATEGORICAL_FEATURES = ["age_risk_bucket"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
AGE_BUCKET_ORDER = [["young", "middle", "senior", "elderly"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini(auc):
    return 2 * auc - 1


def evaluate(model, X, y, split_name):
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    ks = ks_statistic(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    g = gini(auc)
    metrics = {
        f"{split_name}_auc_roc": round(auc, 4),
        f"{split_name}_ks": round(ks, 4),
        f"{split_name}_gini": round(g, 4),
        f"{split_name}_pr_auc": round(pr_auc, 4),
    }
    mlflow.log_metrics(metrics)
    logger.info(
        f"[{split_name}] AUC={auc:.4f} KS={ks:.4f} Gini={g:.4f} PR-AUC={pr_auc:.4f}"
    )
    return metrics


def cv_score(model, X, y, n_splits=5):
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(
        f"[CV-{n_splits}fold] AUC={scores.mean():.4f} ± {scores.std():.4f}"
    )
    return float(scores.mean()), float(scores.std())


def optimal_threshold(y_true, y_prob, cost_fn, cost_fp):
    """Find threshold minimising expected cost."""
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        costs.append(cost_fn * fn + cost_fp * fp)
    return float(thresholds[np.argmin(costs)])


# ---------------------------------------------------------------------------
# Step 1 - Load gold from S3
# ---------------------------------------------------------------------------
logger.info("Loading gold layer from S3...")
s3 = boto3.client(
    "s3",
    endpoint_url=args.s3_endpoint,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)

prefix = f"{args.gold_prefix}ingestion_date={args.ingestion_date}/"
keys = [
    o["Key"]
    for o in s3.list_objects_v2(Bucket=args.s3_bucket, Prefix=prefix).get(
        "Contents", []
    )
    if o["Key"].endswith(".parquet")
]
if not keys:
    raise FileNotFoundError(
        f"No parquet files found at s3://{args.s3_bucket}/{prefix}"
    )

df = pd.concat(
    [
        pd.read_parquet(
            io.BytesIO(
                s3.get_object(Bucket=args.s3_bucket, Key=k)["Body"].read()
            )
        )
        for k in keys
    ],
    ignore_index=True,
)
df = df.drop(columns=["ingestion_date"], errors="ignore")
logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")


# ---------------------------------------------------------------------------
# Step 2 - Stratified split (70/15/15)
# Split happens here - inside the training job - not in the data pipeline.
# This is the leakage firewall: preprocessor is fit only on X_train below.
# ---------------------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
logger.info(f"Split: train={len(y_train)} val={len(y_val)} test={len(y_test)}")
logger.info(
    f"Default rate - train:{y_train.mean()*100:.2f}% val:{y_val.mean()*100:.2f}% test:{y_test.mean()*100:.2f}%"
)


# ---------------------------------------------------------------------------
# Step 3 - Preprocessing pipeline (fit on train ONLY)
# ---------------------------------------------------------------------------
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_pipeline = Pipeline(
    [
        (
            "encoder",
            OrdinalEncoder(
                categories=AGE_BUCKET_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)
logger.info(
    f"Preprocessor fitted on {len(X_train)} rows. Features: {X_train_proc.shape[1]}"
)


# ---------------------------------------------------------------------------
# Step 4 - Baseline training (all 4 models)
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(args.mlflow_uri)
mlflow.set_experiment(args.experiment_name)
client = mlflow.tracking.MlflowClient()

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

baseline_results = {}
for model_name, model in MODELS.items():
    logger.info(f"Training baseline: {model_name}")
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("scale_pos_weight", SPW)
        mlflow.log_param("random_state", RANDOM_STATE)

        # For CV, XGBoost must not have early_stopping_rounds - CV folds
        # have no eval_set, so the callback raises ValueError immediately.
        # We clone the model with early_stopping disabled for CV only,
        # then re-enable it for the final fit with eval_set.
        if model_name == "xgboost":
            from sklearn.base import clone

            cv_model = clone(model)
            cv_model.set_params(early_stopping_rounds=None)
            cv_mean, cv_std = cv_score(cv_model, X_train_proc, y_train)
        else:
            cv_mean, cv_std = cv_score(model, X_train_proc, y_train)

        mlflow.log_metric("cv_auc_mean", round(cv_mean, 4))
        mlflow.log_metric("cv_auc_std", round(cv_std, 4))

        if model_name == "xgboost":
            model.set_params(early_stopping_rounds=20)
            model.fit(
                X_train_proc,
                y_train,
                eval_set=[(X_val_proc, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train_proc, y_train)

        evaluate(model, X_train_proc, y_train, "train")
        val_m = evaluate(model, X_val_proc, y_val, "val")
        mlflow.sklearn.log_model(model, artifact_path="model")

        baseline_results[model_name] = {
            "run_id": run.info.run_id,
            "val_auc": val_m["val_auc_roc"],
            "val_ks": val_m["val_ks"],
            "model": model,
        }

logger.info("Baseline training complete.")


# ---------------------------------------------------------------------------
# Step 5 - Optuna tuning (XGBoost, LightGBM, CatBoost)
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
            m.fit(
                X_train_proc,
                y_train,
                eval_set=[(X_val_proc, y_val)],
                verbose=False,
            )
            return roc_auc_score(y_val, m.predict_proba(X_val_proc)[:, 1])

        m.fit(X_train_proc, y_train)
        return roc_auc_score(y_val, m.predict_proba(X_val_proc)[:, 1])

    return objective


tuning_results = {}
for model_name in ["catboost", "lightgbm", "xgboost"]:
    logger.info(f"Tuning {model_name} ({args.n_trials} trials)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=f"credit_risk_{model_name}",
    )
    study.optimize(make_objective(model_name), n_trials=args.n_trials)

    best_params = study.best_params
    logger.info(
        f"{model_name} best val AUC: {study.best_value:.4f} params: {best_params}"
    )

    # Retrain on full train with best params
    if model_name == "catboost":
        final_model = CatBoostClassifier(**best_params, verbose=0)
        final_model.fit(X_train_proc, y_train)
    elif model_name == "lightgbm":
        final_model = LGBMClassifier(**best_params, verbosity=-1)
        final_model.fit(X_train_proc, y_train)
    else:
        final_model = XGBClassifier(**best_params, verbosity=0)
        final_model.fit(
            X_train_proc,
            y_train,
            eval_set=[(X_val_proc, y_val)],
            verbose=False,
        )

    with mlflow.start_run(run_name=f"{model_name}_tuned") as tuned_run:
        mlflow.log_param("model_name", f"{model_name}_tuned")
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param(
            "baseline_auc", round(baseline_results[model_name]["val_auc"], 4)
        )
        mlflow.log_params(best_params)

        evaluate(final_model, X_train_proc, y_train, "train")
        val_m = evaluate(final_model, X_val_proc, y_val, "val")
        mlflow.log_metric(
            "improvement_vs_baseline",
            round(
                val_m["val_auc_roc"] - baseline_results[model_name]["val_auc"],
                4,
            ),
        )
        mlflow.sklearn.log_model(final_model, artifact_path="model")

        tuning_results[model_name] = {
            "run_id": tuned_run.info.run_id,
            "val_auc": val_m["val_auc_roc"],
            "val_ks": val_m["val_ks"],
            "model": final_model,
        }


# ---------------------------------------------------------------------------
# Step 6 - Final evaluation on test set
# Test set is used exactly once, here.
# ---------------------------------------------------------------------------
logger.info("=== FINAL TEST SET EVALUATION ===")

tuned_champion_name = max(
    tuning_results, key=lambda x: tuning_results[x]["val_auc"]
)
tuned_champion = tuning_results[tuned_champion_name]

all_final = {
    "logistic_regression": baseline_results["logistic_regression"]["model"],
    "xgboost_tuned": tuning_results["xgboost"]["model"],
    "lightgbm_tuned": tuning_results["lightgbm"]["model"],
    "catboost_tuned": tuning_results["catboost"]["model"],
}

test_metrics_all = {}
for name, model in all_final.items():
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ks = ks_statistic(y_test, y_prob)
    test_metrics_all[name] = {"auc": auc, "ks": ks, "y_prob": y_prob}
    logger.info(f"[test] {name} AUC={auc:.4f} KS={ks:.4f}")

# Log test metrics to champion run
champ_prob = tuned_champion["model"].predict_proba(X_test_proc)[:, 1]
champ_test_auc = roc_auc_score(y_test, champ_prob)
champ_test_ks = ks_statistic(y_test, champ_prob)
opt_thresh = optimal_threshold(y_test, champ_prob, args.cost_fn, args.cost_fp)

with mlflow.start_run(run_id=tuned_champion["run_id"]):
    mlflow.log_metrics(
        {
            "test_auc_roc": round(champ_test_auc, 4),
            "test_ks": round(champ_test_ks, 4),
            "test_gini": round(gini(champ_test_auc), 4),
            "optimal_threshold_cost": round(opt_thresh, 4),
        }
    )


# ---------------------------------------------------------------------------
# Step 7 - SHAP + save artifacts
# ---------------------------------------------------------------------------
logger.info("Computing SHAP values...")
explainer = shap.TreeExplainer(tuned_champion["model"])
shap_values = explainer.shap_values(X_val_proc[:2000])
if isinstance(shap_values, list):
    shap_values = shap_values[1]

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_val_proc[:2000], feature_names=ALL_FEATURES, show=False
)
ax.set_title(f"SHAP - {tuned_champion_name}")
plt.tight_layout()
shap_path = "/tmp/shap_summary.png"
plt.savefig(shap_path, dpi=120, bbox_inches="tight")
plt.close()

with mlflow.start_run(run_id=tuned_champion["run_id"]):
    mlflow.log_artifact(shap_path, artifact_path="explainability")


# ---------------------------------------------------------------------------
# Step 8 - Register champion in MLflow Model Registry
# ---------------------------------------------------------------------------
model_uri = f"runs:/{tuned_champion['run_id']}/model"
mv = client.create_model_version(
    name="credit_risk_champion",
    source=model_uri,
    run_id=tuned_champion["run_id"],
)
for _ in range(30):
    mv = client.get_model_version("credit_risk_champion", mv.version)
    if mv.status == "READY":
        break
    time.sleep(2)

client.transition_model_version_stage(
    name="credit_risk_champion",
    version=mv.version,
    stage="Staging",
)
logger.info(f"Registered credit_risk_champion v{mv.version} → Staging")
logger.info(
    f"Champion: {tuned_champion_name} | Test AUC={champ_test_auc:.4f} | KS={champ_test_ks:.4f}"
)


# ---------------------------------------------------------------------------
# Step 9 - Save model + preprocessor to SM_MODEL_DIR
# SageMaker uploads this directory to S3 after the job completes.
# The inference handler loads from here.
# ---------------------------------------------------------------------------
os.makedirs(args.model_dir, exist_ok=True)

model_path = os.path.join(args.model_dir, "model.pkl")
prep_path = os.path.join(args.model_dir, "preprocessor.pkl")
meta_path = os.path.join(args.model_dir, "metadata.json")

with open(model_path, "wb") as f:
    pickle.dump(tuned_champion["model"], f)

with open(prep_path, "wb") as f:
    pickle.dump(preprocessor, f)

metadata = {
    "champion_model": tuned_champion_name,
    "mlflow_run_id": tuned_champion["run_id"],
    "mlflow_version": mv.version,
    "test_auc": round(champ_test_auc, 4),
    "test_ks": round(champ_test_ks, 4),
    "test_gini": round(gini(champ_test_auc), 4),
    "optimal_threshold": round(opt_thresh, 4),
    "features": ALL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "ingestion_date": args.ingestion_date,
    "scale_pos_weight": SPW,
    "random_state": RANDOM_STATE,
}
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

# Also log preprocessor to MLflow
with mlflow.start_run(run_id=tuned_champion["run_id"]):
    mlflow.log_artifact(prep_path, artifact_path="preprocessor")
    mlflow.log_artifact(meta_path, artifact_path="metadata")
    mlflow.log_dict(metadata, "metadata/training_metadata.json")

logger.info(f"Artifacts saved to {args.model_dir}")
logger.info("Training complete.")
