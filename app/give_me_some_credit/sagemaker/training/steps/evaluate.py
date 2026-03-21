"""
evaluate.py — SageMaker ProcessingStep: Step 4
===============================================
Final test set evaluation + conditional model registration.
Writes evaluation report to output — ConditionStep reads it.

SageMaker ProcessingStep mounts:
  Input channel "test"      => /opt/ml/processing/input/test/
  Input channel "tuning"    => /opt/ml/processing/input/tuning/
  Input channel "prep_meta" => /opt/ml/processing/input/prep_meta/
  Output channel "report"   => /opt/ml/processing/output/report/

The ConditionStep in sm_pipeline.py reads evaluation_report.json
and branches: register if test_auc >= threshold, else fail.
"""

import json
import logging
import os
import time

import mlflow
import mlflow.exceptions
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("evaluate")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "credit_risk_pipeline")
AUC_THRESHOLD = float(os.environ.get("AUC_THRESHOLD", "0.85"))
TARGET = "serious_dlqin2yrs"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


TEST_PATH = "/opt/ml/processing/input/test"
TUNING_DIR = "/opt/ml/processing/input/tuning"
PREP_META_DIR = "/opt/ml/processing/input/prep_meta"
OUTPUT_PATH = "/opt/ml/processing/output/report"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def optimal_threshold(y_true, y_prob, cost_fn=10, cost_fp=1):
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        costs.append(cost_fn * fn + cost_fp * fp)
    return float(thresholds[np.argmin(costs)])


logger.info("Loading test set and champion model...")

test_df = pd.read_parquet(os.path.join(TEST_PATH, "test.parquet"))
feature_cols = [c for c in test_df.columns if c != TARGET]
X_test = test_df[feature_cols].values
y_test = test_df[TARGET].values


with open(os.path.join(TUNING_DIR, "tuning_summary.json")) as f:
    tuning_summary = json.load(f)

champion_run_id = tuning_summary["champion_run_id"]
champion_name = tuning_summary["champion_name"]
logger.info(f"Loading champion: {champion_name} from run {champion_run_id}")
model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")


with open(os.path.join(PREP_META_DIR, "prep_meta.json")) as f:
    prep_meta = json.load(f)

prep_run_id = prep_meta["run_id"]
logger.info(f"Loading preprocessor from run {prep_run_id}")
preprocessor = mlflow.sklearn.load_model(f"runs:/{prep_run_id}/preprocessor")

logger.info(
    f"Test set: {len(y_test)} rows | default rate: {y_test.mean()*100:.2f}%"
)


y_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)
test_ks = ks_statistic(y_test, y_prob)
test_gini = 2 * test_auc - 1
test_pr = average_precision_score(y_test, y_prob)
opt_thresh = optimal_threshold(y_test, y_prob)

logger.info(f"TEST AUC={test_auc:.4f} KS={test_ks:.4f} Gini={test_gini:.4f}")
logger.info(f"Optimal threshold (cost): {opt_thresh:.4f}")


with mlflow.start_run(run_id=champion_run_id):
    mlflow.log_metrics(
        {
            "test_auc_roc": round(test_auc, 4),
            "test_ks": round(test_ks, 4),
            "test_gini": round(test_gini, 4),
            "test_pr_auc": round(test_pr, 4),
            "optimal_threshold_cost": round(opt_thresh, 4),
        }
    )


passes_threshold = bool(test_auc >= AUC_THRESHOLD)
logger.info(
    f"AUC {test_auc:.4f} >= threshold {AUC_THRESHOLD}: {passes_threshold}"
)

if passes_threshold:
    logger.info("Registering model in MLflow Model Registry...")
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{champion_run_id}/model"

    try:
        client.create_registered_model("credit_risk_champion")
        logger.info("Created new registered model: credit_risk_champion")
    except mlflow.exceptions.MlflowException:
        logger.info("Registered model already exists, creating new version")

    mv = client.create_model_version(
        name="credit_risk_champion",
        source=model_uri,
        run_id=champion_run_id,
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
    registered_version = mv.version
    logger.info(f"Registered: credit_risk_champion v{mv.version} => Staging")
else:
    registered_version = None
    logger.warning(
        f"Model did NOT meet AUC threshold ({test_auc:.4f} < {AUC_THRESHOLD}). Not registered."
    )


report = {
    "champion_name": tuning_summary["champion_name"],
    "champion_run_id": champion_run_id,
    "val_auc": tuning_summary["val_auc"],
    "test_auc": round(test_auc, 4),
    "test_ks": round(test_ks, 4),
    "test_gini": round(test_gini, 4),
    "test_pr_auc": round(test_pr, 4),
    "optimal_threshold": round(opt_thresh, 4),
    "auc_threshold": AUC_THRESHOLD,
    "passes_threshold": passes_threshold,
    "registered_version": registered_version,
}

report_path = os.path.join(OUTPUT_PATH, "evaluation_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

logger.info(f"Evaluation report written: {report_path}")
logger.info("Evaluation complete.")
