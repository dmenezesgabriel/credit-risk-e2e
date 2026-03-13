import os
import boto3
from sagemaker.local import LocalSession


def get_local_session(
    bucket_name="local-mock-bucket",
    localstack_url="http://localstack:4566"
):
    """
    Creates a SageMaker LocalSession patched to use LocalStack for S3 and STS.
    """
    # Ensure dummy credentials exist for boto3
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    boto_session = boto3.Session()

    # Initialize LocalSession
    sagemaker_session = LocalSession(boto_session=boto_session)

    # Redirect S3 and STS to LocalStack
    ls_s3_client = boto_session.client("s3", endpoint_url=localstack_url)
    ls_sts_client = boto_session.client("sts", endpoint_url=localstack_url)

    sagemaker_session._s3_client = ls_s3_client
    sagemaker_session._sts_client = ls_sts_client

    # Ensure the bucket exists
    try:
        ls_s3_client.create_bucket(Bucket=bucket_name)
    except ls_s3_client.exceptions.BucketAlreadyOwnedByYou:
        pass

    # Patch the bucket name method
    sagemaker_session.default_bucket = lambda: bucket_name

    return sagemaker_session