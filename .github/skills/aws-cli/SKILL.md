---
name: aws-cli
description: Use when need to gather aws data or programatic interactions
---

Since we are running locally with `localstack` always use the argument `--endpoint-url=http://localhost:4566` and the environment variables `AWS_ACCESS_KEY_ID=test`, `AWS_SECRET_ACCESS_KEY=test` and `AWS_DEFAULT_REGION=us-east-1`

- **Verify available commands**:

```sh
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1 aws help --endpoint-url=http://localhost:4566
```

- **List s3 buckets**:

```sh
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1 aws s3 ls --endpoint-url=http://localhost:4566
```
