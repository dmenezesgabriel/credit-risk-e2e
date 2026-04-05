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
import threading
import traceback
from dataclasses import dataclass, field

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


class ScoringService:
    """Thread-safe lazy-loaded singleton for model, preprocessor, and threshold."""

    model = None
    preprocessor = None
    optimal_threshold = 0.5
    feature_columns = None
    _lock = threading.Lock()

    @classmethod
    def load(cls) -> bool:
        if cls.model is not None:
            return True

        with cls._lock:
            # Double-check after acquiring lock
            if cls.model is not None:
                return True

            try:
                model = mlflow.sklearn.load_model(
                    os.path.join(MODEL_DIR, "champion")
                )
                preprocessor = mlflow.sklearn.load_model(
                    os.path.join(MODEL_DIR, "preprocessor")
                )

                report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
                with open(report_path) as f:
                    report = json.load(f)

                cls.optimal_threshold = report["optimal_threshold"]
                cls.feature_columns = report["feature_columns"]
                cls.preprocessor = preprocessor
                # Set model last — it gates the "is loaded" check
                cls.model = model

                logger.info(
                    f"Model loaded: champion={report.get('champion_name', '?')}, "
                    f"threshold={cls.optimal_threshold:.4f}, "
                    f"features={len(cls.feature_columns)}"
                )
                return True
            except Exception:
                logger.error(
                    f"Failed to load model:\n{traceback.format_exc()}"
                )
                cls.model = None
                return False

    @classmethod
    def predict(cls, df: pd.DataFrame) -> pd.DataFrame:
        row_ids = df["row_id"].values
        features = df[cls.feature_columns]

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


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
MAX_NULL_RATE = 0.5


@dataclass
class ValidationResult:
    is_valid: bool
    missing_columns: list[str] = field(default_factory=list)
    high_null_columns: list[str] = field(default_factory=list)
    row_count: int = 0


def validate_input(
    df: pd.DataFrame,
    expected_features: list[str],
) -> ValidationResult:
    """Lightweight schema and null-rate validation. Early return on failure."""
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        return ValidationResult(
            is_valid=False, missing_columns=missing, row_count=len(df)
        )

    high_null = [
        col
        for col in expected_features
        if df[col].isna().mean() > MAX_NULL_RATE
    ]

    return ValidationResult(
        is_valid=len(high_null) == 0,
        high_null_columns=high_null,
        row_count=len(df),
    )


# ---------------------------------------------------------------------------
# Score distribution logging
# ---------------------------------------------------------------------------
def compute_score_summary(
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Compute score distribution summary for output drift visibility."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "mean_probability": round(float(probabilities.mean()), 6),
        "std_probability": round(float(probabilities.std()), 6),
        "p25": round(float(np.percentile(probabilities, 25)), 6),
        "p50": round(float(np.percentile(probabilities, 50)), 6),
        "p75": round(float(np.percentile(probabilities, 75)), 6),
        "pct_predicted_default": round(float(predictions.mean()), 6),
        "record_count": len(probabilities),
    }


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    healthy = ScoringService.load()
    status = 200 if healthy else 404
    return flask.Response(
        response="\n", status=status, mimetype="application/json"
    )


@app.route("/execution-parameters", methods=["GET"])
def execution_parameters():
    return flask.jsonify(
        {
            "MaxConcurrentTransforms": 1,
            "BatchStrategy": "SINGLE_RECORD",
            "MaxPayloadInMB": 6,
        }
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

    validation = validate_input(df, ScoringService.feature_columns)
    if not validation.is_valid:
        error_body = {
            "error": "input_validation_failed",
            "missing_columns": validation.missing_columns,
            "high_null_columns": validation.high_null_columns,
            "row_count": validation.row_count,
        }
        logger.warning(f"Validation failed: {json.dumps(error_body)}")
        return flask.Response(
            response=json.dumps(error_body),
            status=422,
            mimetype="application/json",
        )

    result_df = ScoringService.predict(df)

    score_summary = compute_score_summary(
        result_df["probability"].values, ScoringService.optimal_threshold
    )
    logger.info(f"Score summary: {json.dumps(score_summary)}")

    buf = io.BytesIO()
    result_df.to_parquet(buf, index=False)
    buf.seek(0)

    return flask.Response(
        response=buf.getvalue(),
        status=200,
        mimetype="application/x-parquet",
    )
