.PHONY: up down tear-down

UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_GID := $(shell stat -c '%g' /var/run/docker.sock)
DOCKER_ENV := UID=$(UID) GID=$(GID) DOCKER_GID=$(DOCKER_GID)
COMPOSE := $(DOCKER_ENV) docker compose

up: generate-oidc-key
	$(COMPOSE) up -d

down:
	@echo "Syncing S3 data to volume before shutdown..."
	docker exec localstack mkdir -p /var/lib/localstack/s3_backups/data-lake
	docker exec localstack mkdir -p /var/lib/localstack/s3_backups/mlflow-artifacts
	docker exec localstack awslocal s3 sync s3://data-lake /var/lib/localstack/s3_backups/data-lake
	docker exec localstack awslocal s3 sync s3://mlflow-artifacts /var/lib/localstack/s3_backups/mlflow-artifacts
	$(COMPOSE) down

tear-down:
	$(COMPOSE) down -v


up-%:
	$(COMPOSE) up -d $(*)

down-%:
	$(COMPOSE) down $(*)

generate-oidc-key: ## Generate RSA key for Authelia OIDC (run once)
	@bash scripts/generate_oidc_key.sh

setup-sso: generate-oidc-key ## Full SSO setup: generate key + start services
	$(COMPOSE) build airflow-webserver
	$(COMPOSE) up -d authelia caddy portainer portainer-init airflow-webserver grafana homepage
	@echo ""
	@echo "SSO setup complete! Access URLs:"
	@echo "  Auth:      https://auth.app.localhost"
	@echo "  Grafana:   https://grafana.app.localhost"
	@echo "  Airflow:   https://airflow.app.localhost"
	@echo "  Portainer: https://portainer.app.localhost"
	@echo "  Homepage:  https://home.app.localhost"
