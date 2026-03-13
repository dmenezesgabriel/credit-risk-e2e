# Credit Risk Modeling E2E

```sh
make up
```

Go to airflow web server and run the bronze ingestion DAG

```sh
aws --endpoint-url=http://localhost:4566 s3 ls s3://data-lake/ --recursive
```

```sh
gh repo view --web
```

## References

- https://developers.cyberagent.co.jp/blog/archives/58870/
- https://github.com/aws-samples/amazon-sagemaker-local-mode/blob/main/scikit_learn_bring_your_own_container_local_processing/scikit_learn_bring_your_own_container_local_processing.py
- https://github.com/aws-samples/amazon-sagemaker-local-mode/tree/main/scikit_learn_bring_your_own_container_and_own_model_local_serving
