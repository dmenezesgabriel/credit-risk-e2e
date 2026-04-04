"""
predictor.py — Flask inference server for credit risk batch transform.

SageMaker Batch Transform contract:
  GET  /ping        → 200 if healthy, 404 otherwise
  POST /invocations → predictions for a batch of records

The model artifacts are extracted by SageMaker from model.tar.gz into
/opt/ml/model/ before the container starts.  Expected layout:

  /opt/ml/model/
  ├── champion/          # MLflow sklearn model directory
  ├── preprocessor/      # MLflow sklearn preprocessor directory
  └── evaluation_report.json
"""

import io
import json
import logging
import os
import traceback

import flask
import mlflow.sklearn
import numpy as np
import pandas as pd

logger = logging.getLogger("predictor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

MODEL_DIR = "/opt/ml/model"

# Gold feature schema (must match training preprocessing exactly)
FEATURE_COLUMNS = [
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
    "age_risk_bucket",
]


class ScoringService:
    """Lazy-loaded singleton that holds the model, preprocessor, and threshold."""

    model = None
    preprocessor = None
    optimal_threshold = 0.5

    @classmethod
    def load(cls) -> bool:
        if cls.model is not None:
            return True

        try:
            cls.model = mlflow.sklearn.load_model(
                os.path.join(MODEL_DIR, "champion")
            )
            cls.preprocessor = mlflow.sklearn.load_model(
                os.path.join(MODEL_DIR, "preprocessor")
            )

            report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
            with open(report_path) as f:
                report = json.load(f)
            cls.optimal_threshold = report["optimal_threshold"]

            logger.info(
                f"Model loaded: champion={report.get('champion_name', '?')}, "
                f"threshold={cls.optimal_threshold:.4f}"
            )
            return True
        except Exception:
            logger.error(f"Failed to load model:\n{traceback.format_exc()}")
            cls.model = None
            return False

    @classmethod
    def predict(cls, df: pd.DataFrame) -> pd.DataFrame:
        row_ids = df["row_id"].values
        features = df[FEATURE_COLUMNS]

        X = cls.preprocessor.transform(features)
        probabilities = cls.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= cls.optimal_threshold).astype(int)

        return pd.DataFrame(
            {
                "row_id": row_ids,
                "probability": np.round(probabilities, 6),
                "prediction": predictions,
            }
        )


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    healthy = ScoringService.load()
    status = 200 if healthy else 404
    return flask.Response(
        response="\n", status=status, mimetype="application/json"
    )


@app.route("/invocations", methods=["POST"])
def invocations():
    if not ScoringService.load():
        return flask.Response(
            response="Model not loaded", status=503, mimetype="text/plain"
        )

    content_type = flask.request.content_type or ""
    if "application/x-parquet" not in content_type:
        return flask.Response(
            response=f"Unsupported content type: {content_type}. Use application/x-parquet.",
            status=415,
            mimetype="text/plain",
        )

    try:
        df = pd.read_parquet(io.BytesIO(flask.request.data))
    except Exception as e:
        return flask.Response(
            response=f"Failed to read parquet input: {e}",
            status=400,
            mimetype="text/plain",
        )

    logger.info(f"Invoked with {len(df):,} records")

    result_df = ScoringService.predict(df)

    buf = io.BytesIO()
    result_df.to_parquet(buf, index=False)
    buf.seek(0)

    return flask.Response(
        response=buf.getvalue(),
        status=200,
        mimetype="application/x-parquet",
    )
