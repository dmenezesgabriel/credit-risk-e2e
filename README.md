# MLOps Lab

```mermaid
flowchart TD
    %% Node Definitions
    A1["Ingestion<br/>Raw lake, point-in-time"]
    A2["Split first<br/>Train / val / test"]
    A3["Preprocessing<br/>Impute, encode, scale"]
    A4["Feature engineering<br/>Feature store"]
    A5["Training<br/>Imbalance, CV, algos"]
    A6["Tuning<br/>Optuna, Bayesian"]
    A7["Evaluation<br/>AUC, KS, Gini"]
    A8["Model registry<br/>MLflow, versioning"]
    A9["Deployment<br/>API, shadow mode"]
    A10["Monitoring<br/>Drift, PSI, alerts"]

    %% Connections with Animation Syntax
    A1 e1@--> A2
    A2 e2@--> A3
    A3 e3@--> A4
    A4 e4@--> A5
    A5 e5@--> A6
    A6 e6@--> A7
    A7 e7@--> A8
    A8 e8@--> A9
    A9 e9@--> A10

    %% Updated Retrain Trigger to Preprocessing
    A10 e10@-. retrain trigger .-> A5

    %% Animation Definitions
    e1@{ animation: fast }
    e2@{ animation: fast }
    e3@{ animation: fast }
    e4@{ animation: fast }
    e5@{ animation: fast }
    e6@{ animation: fast }
    e7@{ animation: fast }
    e8@{ animation: fast }
    e9@{ animation: fast }
    e10@{ animation: slow }

    %% Styling
    style A1 fill:#4d4d4d,color:#fff,stroke:#fff,stroke-width:1px
    style A2 fill:#3b328a,color:#fff,stroke:#fff,stroke-width:1px
    style A3 fill:#0a4d3c,color:#fff,stroke:#fff,stroke-width:1px
    style A4 fill:#0a4d3c,color:#fff,stroke:#fff,stroke-width:1px
    style A5 fill:#6e2b14,color:#fff,stroke:#fff,stroke-width:1px
    style A6 fill:#6e2b14,color:#fff,stroke:#fff,stroke-width:1px
    style A7 fill:#73460a,color:#fff,stroke:#fff,stroke-width:1px
    style A8 fill:#73460a,color:#fff,stroke:#fff,stroke-width:1px
    style A9 fill:#0a3b6e,color:#fff,stroke:#fff,stroke-width:1px
    style A10 fill:#0a3b6e,color:#fff,stroke:#fff,stroke-width:1px
```

## Services Summary

| Category               | Service                          | Purpose                                                     |
| ---------------------- | -------------------------------- | ----------------------------------------------------------- |
| **Compute / Dev**      | Jupyter                          | Interactive data science & experimentation environment      |
| **Cloud Mock**         | LocalStack                       | Simulates AWS services locally (S3, Lambda, DynamoDB, etc.) |
| **Database**           | Postgres                         | Metadata & application database (Airflow, MLflow, etc.)     |
| **DB UI**              | Adminer                          | Web UI to manage Postgres                                   |
| **ML Tracking**        | MLflow                           | Experiment tracking, model registry, artifact storage       |
| **Orchestration**      | Airflow (web + scheduler + init) | DAG-based pipeline orchestration                            |
| **Container Mgmt**     | Portainer                        | Docker UI for managing containers                           |
| **Homepage UI**        | Homepage                         | Central dashboard for services                              |
| **Metrics**            | Prometheus                       | Time-series metrics storage & scraping                      |
| **Metrics Source**     | cAdvisor                         | Collects Docker container metrics                           |
| **Visualization**      | Grafana                          | Dashboards for metrics, logs, traces                        |
| **Logs (agent)**       | Promtail                         | Collects and ships logs                                     |
| **Logs (storage)**     | Loki                             | Log aggregation & storage                                   |
| **Tracing (pipeline)** | OTel Collector                   | Collects and routes telemetry data                          |
| **Tracing (storage)**  | Tempo                            | Distributed tracing backend                                 |
| **SSO/Auth**           | Authelia                         | Authentication & authorization (SSO)                        |
| **Reverse Proxy**      | Caddy                            | Routing + TLS + SSO integration                             |
| **LLM Inference**      | llama.cpp                        | Local LLM server for inference                              |
| **Init Jobs**          | Portainer-init                   | Automates initial setup (e.g., OAuth config)                |

### Observability

| Type    | Stack                  |
| ------- | ---------------------- |
| Metrics | Prometheus + cAdvisor  |
| Logs    | Promtail + Loki        |
| Traces  | OTel Collector + Tempo |
| UI      | Grafana                |

## Running Locally

1. Run the containers:

```sh
make up
```

## Useful commands

- Check repository in _github_ url

```sh
gh repo view --web
```

- Docker containers resource consumption:

```sh
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

## References

- [Sagemaker Pipeline Local Mode](https://developers.cyberagent.co.jp/blog/archives/58870/)
- [scikit_learn_bring_your_own_container_local_processing](https://github.com/aws-samples/amazon-sagemaker-local-mode/blob/main/scikit_learn_bring_your_own_container_local_processing/scikit_learn_bring_your_own_container_local_processing.py)
- [scikit_learn_bring_your_own_container_and_own_model_local_serving](https://github.com/aws-samples/amazon-sagemaker-local-mode/tree/main/scikit_learn_bring_your_own_container_and_own_model_local_serving)
- [feast.dev docs](https://rtd.feast.dev/en/latest/#feast.repo_config.RepoConfig)
- [sagemaker-pipelines-bridging-the-gap-between-local-testing-and-deployment](https://aws.plainenglish.io/sagemaker-pipelines-bridging-the-gap-between-local-testing-and-deployment-b5fcd8694b80)
- [tensorflow_bring_your_own_california_housing_local_training_and_batch_transform/](https://github.com/aws-samples/amazon-sagemaker-local-mode/blob/main/tensorflow_bring_your_own_california_housing_local_training_and_batch_transform)
- [automating-feature-engineering-workflows-with-amazon-managed-workflows-for-apache-airflow-mwaa](https://medium.com/@TransformationalLeader_Surjeet/automating-feature-engineering-workflows-with-amazon-managed-workflows-for-apache-airflow-mwaa-b557aa46b1f5)
- [integrating-opentelemetry-for-logging-in-python-a-practical-guide](https://medium.com/@lakinduboteju/integrating-opentelemetry-for-logging-in-python-a-practical-guide-fe52bff61edc)
- [Qwen2.5-Coder-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF)
- [LFM2.5-350M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)
- [mastering-aws-cli](https://github.com/SpillwaveSolutions/mastering-aws-cli/blob/main/references/s3.md)
- [sagemaker-ml-workflow-with-apache-airflow](https://github.com/aws-samples/sagemaker-ml-workflow-with-apache-airflow/blob/master/README.md)
