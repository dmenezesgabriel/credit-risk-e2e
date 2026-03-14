"""
Feast feature repository — Credit Risk
======================================
Offline store  : S3 parquet on localstack (gold layer written by Glue job)
Online store   : DynamoDB on localstack — populated by feast materialize

All endpoint config lives in feature_store.yaml — this file stays clean
and environment-agnostic. Swapping localstack for real AWS requires only
removing the two endpoint lines from feature_store.yaml.
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

# ---------------------------------------------------------------------------
# Source
# s3_endpoint_override is inherited from offline_store config in
# feature_store.yaml — no need to repeat it here.
# ---------------------------------------------------------------------------
credit_source = FileSource(
    path="s3://data-lake/gold/credit_risk/features/",
    timestamp_field="event_timestamp",
    s3_endpoint_override="http://localstack:4566",
)

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Credit applicant. Surrogate key from Glue job row index.",
)

# ---------------------------------------------------------------------------
# FeatureView — original features (capped, nulls preserved)
# ---------------------------------------------------------------------------
credit_original_features = FeatureView(
    name="credit_original_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="revolving_utilization_of_unsecured_lines", dtype=Float32),
        Field(name="age", dtype=Int32),
        Field(name="number_of_time30_59_days_past_due_not_worse", dtype=Int32),
        Field(name="debt_ratio", dtype=Float32),
        Field(name="monthly_income", dtype=Float32),
        Field(name="number_of_open_credit_lines_and_loans", dtype=Int32),
        Field(name="number_of_times90_days_late", dtype=Int32),
        Field(name="number_real_estate_loans_or_lines", dtype=Int32),
        Field(name="number_of_time60_89_days_past_due_not_worse", dtype=Int32),
        Field(name="number_of_dependents", dtype=Float32),
    ],
    source=credit_source,
    description="Original silver features after outlier capping. Nulls preserved.",
)

# ---------------------------------------------------------------------------
# FeatureView — engineered features (stateless transforms)
# ---------------------------------------------------------------------------
credit_engineered_features = FeatureView(
    name="credit_engineered_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="monthly_income_is_missing", dtype=Int32),
        Field(name="number_of_dependents_is_missing", dtype=Int32),
        Field(name="delinquency_score", dtype=Float32),
        Field(name="debt_to_income_ratio", dtype=Float32),
        Field(name="unsecured_to_total_lines_ratio", dtype=Float32),
        Field(name="age_risk_bucket", dtype=String),
        Field(name="has_any_delinquency", dtype=Int32),
    ],
    source=credit_source,
    description="Engineered features. Stateless — reproducible at inference time.",
)

# ---------------------------------------------------------------------------
# FeatureService — versioned bundle for training + inference
# ---------------------------------------------------------------------------
credit_risk_v1 = FeatureService(
    name="credit_risk_v1",
    features=[
        credit_original_features,
        credit_engineered_features,
    ],
    description=(
        "Full feature set for credit risk model v1. "
        "Target (serious_dlqin2yrs) fetched separately as label."
    ),
)
