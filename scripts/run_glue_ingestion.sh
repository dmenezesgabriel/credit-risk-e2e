docker run --rm \
  --network credit-risk-e2e_credit-risk-net \
  -e AWS_ACCESS_KEY_ID=test \
  -e AWS_SECRET_ACCESS_KEY=test \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
  -e KAGGLE_KEY=$KAGGLE_KEY \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  public.ecr.aws/glue/aws-glue-libs:5 \
  spark-submit /workspace/glue_jobs/bronze_ingestion.py
