# Credit Risk Modeling E2E

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

### 1. Ingestion

Raw data arrives from source systems and then are landed in a data lake (S3) as immutable raw snapshots, partitioned by date. Never Overwrite data.

- **Temporal integrity**: You must know exact _point-in-time_ at which each feature was observable. If a feature uses information from the future (flag set after loan is approved), that's leakage before modeling start.

### 2. Split data

Before touching a single null value, you must split your data into train/validation/test. In credit risk the split must be **temporal** (not random).

If you impute missing values (e.g., fill median income) using the full dataset before splitting, the test set leaks information about itself into the training imputer. Same with scalers, encoders, and any aggregation. Fit all transformers on train only, then transform val and test.

### 3. Preprocessing

Missing values can be handled, fit imputers must be applied on train dataset only.

- For numeric features, median imputation is the default (robust for outliers)
- For categorical features, a separate _"missing"_ category often outperforms dropping or mode-filling because tree models can actually learn from _"this value was absent"_
- Scaling (StandardScaler, MinMaxScaler) is only strictly needed for Logistic Regression
- Tree ensembles (XGBoost, LightGBM, CatBoost, Random Forest) are scale-invariant

## Running Locally

1. Run the containers:

```sh
make up
```

2. Go to airflow web server and run the `credit_risk_pipeline_dag.py` ingestion DAG

3. Verify if files were created:

```sh
aws --endpoint-url=http://localhost:4566 s3 ls s3://data-lake/ --recursive
```

## Useful commands

- Check repository in _github_ url

```sh
gh repo view --web
```

## References

- [Sagemaker Pipeline Local Mode](https://developers.cyberagent.co.jp/blog/archives/58870/)
- [scikit_learn_bring_your_own_container_local_processing](https://github.com/aws-samples/amazon-sagemaker-local-mode/blob/main/scikit_learn_bring_your_own_container_local_processing/scikit_learn_bring_your_own_container_local_processing.py)
- [scikit_learn_bring_your_own_container_and_own_model_local_serving](https://github.com/aws-samples/amazon-sagemaker-local-mode/tree/main/scikit_learn_bring_your_own_container_and_own_model_local_serving)
