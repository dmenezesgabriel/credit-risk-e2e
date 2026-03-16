"""
evaluate.py — SageMaker ProcessingStep: Step 4
===============================================
Final test set evaluation + conditional model registration.
Writes evaluation report to output — ConditionStep reads it.

SageMaker ProcessingStep mounts:
  Input channel "test"   → /opt/ml/processing/input/test/
  Input channel "model"  → /opt/ml/processing/input/model/
  Input channel "prep"   → /opt/ml/processing/input/preprocessor/
  Output channel "report"→ /opt/ml/processing/output/report/

The ConditionStep in sm_pipeline.py reads evaluation_report.json
and branches: register if test_auc >= threshold, else fail.
"""

import json
import logging
import os
import pickle
import time

import mlflow
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_PATH = "/opt/ml/processing/input/test"
MODEL_PATH = "/opt/ml/processing/input/model"
PREP_PATH = "/opt/ml/processing/input/preprocessor"
OUTPUT_PATH = "/opt/ml/processing/output/report"
os.makedirs(OUTPUT_PATH, exist_ok=True)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "credit_risk_pipeline")
AUC_THRESHOLD = float(os.environ.get("AUC_THRESHOLD", "0.85"))
TARGET = "serious_dlqin2yrs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Load test set + model
# ---------------------------------------------------------------------------
logger.info("Loading test set and champion model...")

test_df = pd.read_parquet(os.path.join(TEST_PATH, "test.parquet"))
feature_cols = [c for c in test_df.columns if c != TARGET]
X_test = test_df[feature_cols].values
y_test = test_df[TARGET].values

model_file = os.path.join(MODEL_PATH, "model.pkl")
with open(model_file, "rb") as f:
    model = pickle.load(f)

tuning_summary_file = os.path.join(MODEL_PATH, "tuning_summary.json")
with open(tuning_summary_file) as f:
    tuning_summary = json.load(f)

prep_file = os.path.join(PREP_PATH, "preprocessor.pkl")
with open(prep_file, "rb") as f:
    preprocessor = pickle.load(f)

config_file = os.path.join(PREP_PATH, "feature_config.json")
with open(config_file) as f:
    feature_config = json.load(f)

logger.info(
    f"Champion: {tuning_summary['champion_name']} | val_auc={tuning_summary['val_auc']:.4f}"
)
logger.info(f"Test set: {len(y_test)} rows | default rate: {y_test.mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# Final test set evaluation — used exactly once
# ---------------------------------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)
test_ks = ks_statistic(y_test, y_prob)
test_gini = 2 * test_auc - 1
test_pr = average_precision_score(y_test, y_prob)
opt_thresh = optimal_threshold(y_test, y_prob)

logger.info(f"TEST AUC={test_auc:.4f} KS={test_ks:.4f} Gini={test_gini:.4f}")
logger.info(f"Optimal threshold (cost): {opt_thresh:.4f}")

# ---------------------------------------------------------------------------
# Log to MLflow under champion run
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

champion_run_id = tuning_summary["champion_run_id"]
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
    mlflow.log_artifact(prep_file, artifact_path="preprocessor")

# ---------------------------------------------------------------------------
# Register if AUC passes threshold
# ConditionStep reads evaluation_report.json to make the branch decision.
# ---------------------------------------------------------------------------
passes_threshold = bool(test_auc >= AUC_THRESHOLD)
logger.info(f"AUC {test_auc:.4f} >= threshold {AUC_THRESHOLD}: {passes_threshold}")

if passes_threshold:
    logger.info("Registering model in MLflow Model Registry...")
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{champion_run_id}/model"
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
    logger.info(f"Registered: credit_risk_champion v{mv.version} → Staging")
else:
    registered_version = None
    logger.warning(
        f"Model did NOT meet AUC threshold ({test_auc:.4f} < {AUC_THRESHOLD}). Not registered."
    )

# ---------------------------------------------------------------------------
# Write evaluation report — ConditionStep reads this
# ---------------------------------------------------------------------------
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
