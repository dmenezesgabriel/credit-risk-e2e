# Credit Risk Modeling E2E

```sh
make up
```

Go to airflow web server and run the bronze ingestion DAG

```sh
aws --endpoint-url=http://localhost:4566 s3 ls s3://data-lake/ --recursive
```
