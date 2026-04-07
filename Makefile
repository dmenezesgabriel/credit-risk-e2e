.PHONY: up down tear-down up-llm up-adminer up-airflow up-lineage build-lambdas tf-apply

UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_GID := $(shell stat -c '%g' /var/run/docker.sock)
DOCKER_ENV := UID=$(UID) GID=$(GID) DOCKER_GID=$(DOCKER_GID)
COMPOSE := $(DOCKER_ENV) docker compose
COMPOSE_ALL := $(COMPOSE) --profile llm --profile tools --profile airflow --profile lineage

setup-hosts: ## Ensure *.app.localhost resolves to 127.0.0.1
	@bash scripts/setup_hosts.sh

up: generate-oidc-key setup-hosts
	$(COMPOSE) up -d

up-llm: ## Start llamacpp LLM server
	$(COMPOSE) --profile llm up -d llamacpp

up-adminer: ## Start Adminer DB UI
	$(COMPOSE) --profile tools up -d adminer

up-airflow: generate-oidc-key setup-hosts ## Start full stack with Airflow
	$(COMPOSE) --profile airflow up -d

up-lineage: ## Start Marquez lineage service
	$(COMPOSE) --profile lineage up -d

down:
	@echo "Syncing S3 data to volume before shutdown..."
	docker exec localstack mkdir -p /var/lib/localstack/s3_backups/data-lake
	docker exec localstack mkdir -p /var/lib/localstack/s3_backups/mlflow-artifacts
	docker exec localstack awslocal s3 sync s3://data-lake /var/lib/localstack/s3_backups/data-lake
	docker exec localstack awslocal s3 sync s3://mlflow-artifacts /var/lib/localstack/s3_backups/mlflow-artifacts
	$(COMPOSE_ALL) down

tear-down:
	$(COMPOSE_ALL) down -v

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

build-lambdas: ## Install Lambda deps into .build/<name>/ (run before terraform apply)
	cd app/nyc_taxi_trip && uv run scripts/build_lambdas.py

tf-apply: ## Plan and apply Terraform for nyc_taxi_trip (requires build-lambdas first)
	cd app/nyc_taxi_trip/terraform && \
	  terraform init -upgrade -input=false && \
	  terraform apply -var="aws_endpoint=http://localhost:4566" -auto-approve
