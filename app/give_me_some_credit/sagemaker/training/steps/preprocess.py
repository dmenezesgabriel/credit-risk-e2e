"""
preprocess.py — SageMaker ProcessingStep: Step 1

Reads gold parquet, produces train/val/test splits + fitted preprocessor.

SageMaker mounts:
  Input  "gold"         => /opt/ml/processing/input/gold/
  Output "train"        => /opt/ml/processing/output/train/
  Output "val"          => /opt/ml/processing/output/val/
  Output "test"         => /opt/ml/processing/output/test/
  Output "preprocessor" => /opt/ml/processing/output/preprocessor/

Run locally:
    python preprocess.py \
        --mlflow-uri http://localhost:5000 \
        --experiment-name credit_risk_pipeline \
        --random-state 42
"""

import argparse
import json
import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocess")

# ---------------------------------------------------------------------------
# SageMaker I/O paths
# ---------------------------------------------------------------------------
INPUT_PATH = "/opt/ml/processing/input/gold"
OUTPUT_TRAIN = "/opt/ml/processing/output/train"
OUTPUT_VAL = "/opt/ml/processing/output/val"
OUTPUT_TEST = "/opt/ml/processing/output/test"
OUTPUT_PREP = "/opt/ml/processing/output/preprocessor"

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
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
AGE_BUCKET_ORDER = [["young", "middle", "senior", "elderly"]]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Split ratios
# ---------------------------------------------------------------------------
VAL_TEST_RATIO = 0.30
TEST_FROM_TEMP_RATIO = 0.50


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data(input_path: str) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VAL_TEST_RATIO,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=TEST_FROM_TEMP_RATIO,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def fit_and_transform(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    preprocessor.fit(X_train)
    logger.info(f"Preprocessor fitted on {len(X_train):,} rows")
    return (
        preprocessor.transform(X_train),
        preprocessor.transform(X_val),
        preprocessor.transform(X_test),
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_splits(
    splits: dict[str, tuple],
    output_paths: dict[str, str],
) -> None:
    for name, (X_proc, y_series) in splits.items():
        out_path = output_paths[name]
        os.makedirs(out_path, exist_ok=True)

        df_out = pd.DataFrame(X_proc, columns=ALL_FEATURES)
        df_out[TARGET] = y_series.values

        out_file = os.path.join(out_path, f"{name}.parquet")
        pq.write_table(
            pa.Table.from_pandas(df_out, preserve_index=False), out_file
        )
        logger.info(f"Saved {name}: {len(df_out):,} rows => {out_file}")


def save_metadata(
    run_id: str,
    random_state: int,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    os.makedirs(OUTPUT_PREP, exist_ok=True)

    with open(os.path.join(OUTPUT_PREP, "prep_meta.json"), "w") as f:
        json.dump({"run_id": run_id}, f)

    with open(os.path.join(OUTPUT_PREP, "feature_config.json"), "w") as f:
        json.dump(
            {
                "features": ALL_FEATURES,
                "numeric": NUMERIC_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
                "target": TARGET,
                "random_state": random_state,
                "train_rows": len(y_train),
                "val_rows": len(y_val),
                "test_rows": len(y_test),
                "default_rate_train": float(y_train.mean()),
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def log_to_mlflow(
    preprocessor: ColumnTransformer,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> str:
    with mlflow.start_run(run_name="preprocessing") as run:
        mlflow.log_params(
            {
                "step": "preprocessing",
                "random_state": random_state,
                "train_size": len(y_train),
                "val_size": len(y_val),
                "test_size": len(y_test),
            }
        )
        mlflow.sklearn.log_model(preprocessor, artifact_path="preprocessor")
        return run.info.run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    logger.info(f"Args: {vars(args)}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    df = load_data(INPUT_PATH)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, args.random_state
    )

    for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
        logger.info(
            f"  {name}: {len(ys):,} rows | default rate: {ys.mean() * 100:.2f}%"
        )

    preprocessor = build_preprocessor()
    X_train_proc, X_val_proc, X_test_proc = fit_and_transform(
        preprocessor, X_train, X_val, X_test
    )

    save_splits(
        splits={
            "train": (X_train_proc, y_train),
            "val": (X_val_proc, y_val),
            "test": (X_test_proc, y_test),
        },
        output_paths={
            "train": OUTPUT_TRAIN,
            "val": OUTPUT_VAL,
            "test": OUTPUT_TEST,
        },
    )

    run_id = log_to_mlflow(
        preprocessor, y_train, y_val, y_test, args.random_state
    )
    logger.info(f"Preprocessor logged to MLflow run: {run_id}")

    save_metadata(run_id, args.random_state, y_train, y_val, y_test)

    logger.info("Preprocessing complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit risk preprocessing step"
    )
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="credit_risk_pipeline")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
