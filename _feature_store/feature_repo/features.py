"""
Feast feature repository — Credit Risk
======================================
Offline store  : S3 parquet on localstack (gold layer written by Glue job)
Online store   : DynamoDB on localstack — populated by feast materialize

All endpoint config lives in feature_store.yaml — this file stays clean
and environment-agnostic. Swapping localstack for real AWS requires only
removing the two endpoint lines from feature_store.yaml.
"""

# from datetime import timedelta

# from feast import Entity, FeatureService, FeatureView, Field, FileSource
# from feast.types import Float32, Int32, String

# # ---------------------------------------------------------------------------
# # Source
# # s3_endpoint_override is inherited from offline_store config in
# # feature_store.yaml — no need to repeat it here.
# # ---------------------------------------------------------------------------
# source = FileSource(
#     path="s3://data-lake/gold/<my_key>/features/",
#     timestamp_field="event_timestamp",
#     s3_endpoint_override="http://localstack:4566",
# )
