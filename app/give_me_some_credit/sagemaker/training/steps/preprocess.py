"""
preprocess.py — SageMaker ProcessingStep: Step 1
=================================================
Reads gold parquet from S3, runs the full preprocessing pipeline,
writes train/val/test splits + fitted preprocessor to output paths.

SageMaker ProcessingStep mounts:
  Input  channel "gold"  => /opt/ml/processing/input/gold/
  Output channel "train" => /opt/ml/processing/output/train/
  Output channel "val"   => /opt/ml/processing/output/val/
  Output channel "test"  => /opt/ml/processing/output/test/
  Output channel "model" => /opt/ml/processing/output/preprocessor/

All paths are standard SageMaker ProcessingStep conventions.
"""

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

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
)
mlflow.set_experiment(
    os.environ.get("EXPERIMENT_NAME", "credit_risk_pipeline")
)


INPUT_PATH = "/opt/ml/processing/input/gold"
OUTPUT_TRAIN = "/opt/ml/processing/output/train"
OUTPUT_VAL = "/opt/ml/processing/output/val"
OUTPUT_TEST = "/opt/ml/processing/output/test"
OUTPUT_PREP = "/opt/ml/processing/output/preprocessor"

for path in [OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST, OUTPUT_PREP]:
    os.makedirs(path, exist_ok=True)

TARGET = "serious_dlqin2yrs"
RANDOM_STATE = 42

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


logger.info(f"Reading gold from: {INPUT_PATH}")
parquet_files = [
    os.path.join(INPUT_PATH, f)
    for f in os.listdir(INPUT_PATH)
    if f.endswith(".parquet")
]
if not parquet_files:
    raise FileNotFoundError(f"No parquet files in {INPUT_PATH}")

df = pd.concat(
    [pd.read_parquet(f) for f in parquet_files],
    ignore_index=True,
)
df = df.drop(columns=["ingestion_date"], errors="ignore")
logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")


X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
    logger.info(
        f"  {name}: {len(ys)} rows | default rate: {ys.mean()*100:.2f}%"
    )


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
logger.info(f"Preprocessor fitted on {len(X_train)} rows")

X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)


def write_split(X_proc, y_series, out_path, split_name):
    df_out = pd.DataFrame(X_proc, columns=ALL_FEATURES)
    df_out[TARGET] = y_series.values
    table = pa.Table.from_pandas(df_out, preserve_index=False)
    out_file = os.path.join(out_path, f"{split_name}.parquet")
    pq.write_table(table, out_file)
    logger.info(f"Written: {out_file} ({len(df_out)} rows)")


write_split(X_train_proc, y_train, OUTPUT_TRAIN, "train")
write_split(X_val_proc, y_val, OUTPUT_VAL, "val")
write_split(X_test_proc, y_test, OUTPUT_TEST, "test")

with mlflow.start_run(run_name="preprocessing") as run:
    mlflow.log_param("step", "preprocessing")
    mlflow.log_param("train_size", len(y_train))
    mlflow.log_param("val_size", len(y_val))
    mlflow.log_param("test_size", len(y_test))
    mlflow.sklearn.log_model(preprocessor, artifact_path="preprocessor")
    prep_run_id = run.info.run_id

prep_meta = {"run_id": prep_run_id}
meta_path = os.path.join(OUTPUT_PREP, "prep_meta.json")
with open(meta_path, "w") as f:
    json.dump(prep_meta, f)
logger.info(f"Preprocessor logged to MLflow run {prep_run_id}")

config = {
    "features": ALL_FEATURES,
    "numeric": NUMERIC_FEATURES,
    "categorical": CATEGORICAL_FEATURES,
    "target": TARGET,
    "random_state": RANDOM_STATE,
    "train_rows": len(y_train),
    "val_rows": len(y_val),
    "test_rows": len(y_test),
    "default_rate_train": float(y_train.mean()),
}
config_path = os.path.join(OUTPUT_PREP, "feature_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
logger.info(f"Feature config saved: {config_path}")

logger.info("Preprocessing complete.")
