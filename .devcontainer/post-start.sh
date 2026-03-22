#!/usr/bin/env bash

set -euo pipefail

echo "==> Starting MLOps Lab stack..."

UID_VAL=$(id -u)
GID_VAL=$(id -g)

if [ -S /var/run/docker.sock ]; then
  DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
else
  DOCKER_GID="$GID_VAL"
  echo "  WARNING: /var/run/docker.sock not found — DOCKER_GID defaulting to $GID_VAL"
fi

echo "  UID=$UID_VAL  GID=$GID_VAL  DOCKER_GID=$DOCKER_GID"

UID=$UID_VAL GID=$GID_VAL DOCKER_GID=$DOCKER_GID \
  docker compose up -d --build

echo "Startup ok"
