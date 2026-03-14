.PHONY: up down

up:
	DOCKER_GID=$$(stat -c '%g' /var/run/docker.sock) docker compose up -d

down:
	DOCKER_GID=$$(stat -c '%g' /var/run/docker.sock) docker compose down

down-v:
	DOCKER_GID=$$(stat -c '%g' /var/run/docker.sock) docker compose down -v
