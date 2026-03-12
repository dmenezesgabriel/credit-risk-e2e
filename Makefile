.PHONY: up down

up:
	DOCKER_GID=$$(stat -c '%g' /var/run/docker.sock) docker compose up -d

down:
	docker compose down