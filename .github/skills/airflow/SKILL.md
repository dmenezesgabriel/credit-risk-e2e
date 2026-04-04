---
name: airflow
description: Use when need to interact with, debug, or gather information from Airflow and its DAG executions
---

Airflow runs locally via Docker Compose with **Airflow 3.x** (LocalExecutor, PostgreSQL backend).

**Prefer running the CLI outside the container** using `uv run` pointed at the shared PostgreSQL database:

```sh
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow uv run --with "apache-airflow[postgres]" airflow <command>
```

This works for all DB-backed commands (`dags list`, `dags trigger`, `dags reserialize`, etc.). For commands that read container-local state (e.g. `config get-value`, container logs, task log files), use `docker exec` instead.

## Key Airflow 3.x differences

- Use `schedule=None` instead of `schedule_interval=None` (removed in Airflow 3).
- Template variables `{{ ds }}`, `{{ logical_date }}` are **only available when the DagRun has a `logical_date`** (i.e. scheduled runs). For `schedule=None` manual triggers, use `{{ macros.datetime.now().strftime('%Y-%m-%d') }}` or access `dag_run.run_id`.
- The Execution API architecture requires the scheduler to reach the webserver at `AIRFLOW__CORE__EXECUTION_API_SERVER_URL` (e.g. `http://airflow-webserver:8080/execution/`).
- Scheduler and webserver must share the same `AIRFLOW__API_AUTH__JWT_SECRET` when running in separate containers.

## CLI commands

Commands can be run either outside the container (preferred) or via `docker exec`.

- **List DAGs**:

```sh
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow uv run --with "apache-airflow[postgres]" airflow dags list
```

- **Trigger a DAG**:

```sh
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow uv run --with "apache-airflow[postgres]" airflow dags trigger <dag_id>
```

- **Reserialize DAGs** (force reload after editing DAG files):

```sh
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow uv run --with "apache-airflow[postgres]" airflow dags reserialize
```

- **Check a config value** (must use `docker exec` — reads container-local config):

```sh
docker exec airflow-webserver airflow config get-value <section> <key>
```

Example: `docker exec airflow-webserver airflow config get-value api_auth jwt_secret`

## Reading logs

- **Scheduler container logs** (for scheduling, task dispatch, and DAG run state):

```sh
docker logs airflow-scheduler --since 60s 2>&1 | grep -iE "error|fail|success"
```

- **Task instance logs** (stored on the shared `airflow_logs` volume):

```sh
docker exec airflow-scheduler cat "/opt/airflow/logs/dag_id=<dag_id>/run_id=<run_id>/task_id=<task_id>/attempt=1.log"
```

- **Find recent task log files**:

```sh
docker exec airflow-scheduler find /opt/airflow/logs -type f -name "*.log" | sort | tail -10
```

## Querying state via Python (in-container)

When the CLI lacks a subcommand (e.g. `list-runs` was removed in Airflow 3.x), query the DB directly:

- **List DAG runs for a DAG**:

```sh
docker exec airflow-webserver python3 -c "
from airflow.models.dagrun import DagRun
from airflow.utils.session import create_session
with create_session() as session:
    runs = session.query(DagRun).filter(DagRun.dag_id == '<dag_id>').all()
    for r in runs:
        print(f'run_id={r.run_id} state={r.state} start_date={r.start_date}')
"
```

- **List task instances for a run**:

```sh
docker exec airflow-webserver python3 -c "
from airflow.models.taskinstance import TaskInstance
from airflow.utils.session import create_session
with create_session() as session:
    tis = session.query(TaskInstance).filter(
        TaskInstance.dag_id == '<dag_id>',
        TaskInstance.run_id == '<run_id>'
    ).all()
    for ti in tis:
        print(f'task={ti.task_id} state={ti.state} duration={ti.duration}')
"
```

- **Check or set DAG paused state**:

```sh
docker exec airflow-webserver python3 -c "
from airflow.models import DagModel
from airflow.utils.session import create_session
with create_session() as session:
    dm = session.query(DagModel).filter(DagModel.dag_id == '<dag_id>').first()
    print(f'is_paused={dm.is_paused}')
"
```

## Common debugging patterns

1. **DAGs not showing in UI**: Check for import errors — run `docker logs airflow-scheduler 2>&1 | grep -i "error"` and look for `TypeError` or `ModuleNotFoundError` on DAG files.
2. **Tasks stuck in queued**: Check scheduler logs for `Connection refused` (execution API unreachable) or `Signature verification failed` (JWT secret mismatch).
3. **Template rendering errors**: Check task logs for `UndefinedError`. Remember that `ds`, `logical_date`, and derived variables are undefined for manual triggers on `schedule=None` DAGs in Airflow 3.x.
4. **Log URL errors** (`No host supplied`): Ensure scheduler and webserver share the `airflow_logs` volume and the scheduler has a `hostname` set in docker-compose.

## Important notes

- Always use `make up` (not raw `docker compose up`) to start services — the Makefile injects required `UID`, `GID`, and `DOCKER_GID` variables.
- After editing DAG files, run `airflow dags reserialize` to force the scheduler to pick up changes immediately.
- The SDK `DagRun` datamodel in Airflow 3.x has these fields: `dag_id`, `run_id`, `logical_date`, `data_interval_start`, `data_interval_end`, `run_after`, `start_date`, `end_date`, `clear_number`, `run_type`, `state`, `conf`, `triggering_user_name`, `consumed_asset_events`.
