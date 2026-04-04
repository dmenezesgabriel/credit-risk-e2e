---
name: mlflow
description: Use when need to gather mlflow data or programatic interactions
---

Use `uv/astral` tools functionality to verify available mlflow cli options, gather data or interact with tracking server.

Since `mlflow` is being runned locally with docker compose tracing server need to be passed an environment variable. `MLFLOW_TRACKING_URI=http://localhost:5000`

Below some example commands:

- **Verify available MLflow Commands**:

```sh
uv run --with mlflow mlflow experiments --help
```

- **List experiments**:

```sh
MLFLOW_TRACKING_URI=http://localhost:5000 uv run --with mlflow mlflow experiments search
```
