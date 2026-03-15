.PHONY: up down down-v

up:
	DOCKER_GID=$$(stat -c '%g' /var/run/docker.sock) docker compose up -d

down:
	@echo "Syncing S3 data to volume before shutdown..."
	docker exec localstack mkdir -p /var/lib/localstack/s3_backups/data-lake
	docker exec localstack mkdir -p /var/lib/localstack/s3_backups/mlflow-artifacts
	docker exec localstack awslocal s3 sync s3://data-lake /var/lib/localstack/s3_backups/data-lake
	docker exec localstack awslocal s3 sync s3://mlflow-artifacts /var/lib/localstack/s3_backups/mlflow-artifacts
	DOCKER_GID=$$(stat -c '%g' /var/run/docker.sock) docker compose down

tear-down:
	DOCKER_GID=$$(stat -c '%g' /var/run/docker.sock) docker compose down -v
