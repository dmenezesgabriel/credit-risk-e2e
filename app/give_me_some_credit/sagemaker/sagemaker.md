# SageMaker Inference Patterns — Reference Guide

A comparison of the three main inference deployment patterns in the SageMaker Python SDK,
covering Batch Transform and real-time endpoints.

---

## Pattern 1 — Framework Containers with Script Hooks

AWS provides pre-built Docker images for sklearn, XGBoost, PyTorch, TensorFlow, etc.
You supply a Python script with lifecycle hooks. The AWS inference toolkit handles
the HTTP layer (Flask-like server, serialization, concurrency) internally.

### Hooks

| Hook         | Signature                                | Required?            | Default behaviour                  |
| ------------ | ---------------------------------------- | -------------------- | ---------------------------------- |
| `model_fn`   | `model_fn(model_dir) -> model`           | Yes (or use default) | Loads `model.joblib` / `model.pkl` |
| `input_fn`   | `input_fn(body, content_type) -> object` | Optional             | Deserializes numpy/json/csv        |
| `predict_fn` | `predict_fn(input, model) -> output`     | Optional             | Calls `model.predict(input)`       |
| `output_fn`  | `output_fn(prediction, accept) -> bytes` | Optional             | Serializes numpy/json/csv          |

### SDK Classes

- **Training**: `SKLearn`, `XGBoost`, `PyTorch`, `TensorFlow` (Estimator subclasses)
- **Inference**: `SKLearnModel`, `XGBoostModel`, `PyTorchModel` — wrap a `model.tar.gz` and
  an entry-point script
- **SDK v3 replacement**: `ModelBuilder` (replaces all framework-specific Model classes)

### Code Snippet

```python
# inference_script.py — only the hooks you need to override
import joblib, os

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type):
    if content_type == "application/json":
        import json, numpy as np
        return np.array(json.loads(request_body))
    raise ValueError(f"Unsupported: {content_type}")

def predict_fn(input_data, model):
    return model.predict_proba(input_data)[:, 1]

def output_fn(prediction, accept):
    import json
    return json.dumps(prediction.tolist()), "application/json"
```

```python
# Orchestration
from sagemaker.sklearn.model import SKLearnModel

model = SKLearnModel(
    model_data="s3://bucket/model.tar.gz",
    entry_point="inference_script.py",
    framework_version="1.2-1",
    role=role,
)
# Batch Transform
transformer = model.transformer(instance_count=1, instance_type="ml.m5.xlarge")
transformer.transform("s3://bucket/input/", content_type="text/csv")

# Real-time endpoint
predictor = model.deploy(instance_type="ml.m5.xlarge", initial_instance_count=1)
```

### Tradeoffs

| Pros                                          | Cons                                                          |
| --------------------------------------------- | ------------------------------------------------------------- |
| Minimal boilerplate                           | Locked to AWS framework versions                              |
| AWS manages Flask, gunicorn, concurrency      | No custom OS packages                                         |
| Works out-of-the-box for numpy/csv/json       | Awkward for binary formats (parquet, protobuf)                |
| Easy local testing with `local` instance type | Multi-model loading requires workarounds                      |
| Automatic content-type negotiation            | Hooks called sequentially — no control over batching strategy |

### Real-World Use Cases

- Simple sklearn/XGBoost models with tabular CSV or JSON input
- Rapid prototyping where you want to avoid writing a server
- Teams without Docker expertise
- Models with a single artifact and standard content types

---

## Pattern 2 — BYOC (Bring Your Own Container) with Flask

You build and own your Docker image. SageMaker only enforces a minimal HTTP contract.
The container runs any server you choose (nginx + gunicorn + Flask is the AWS reference).

### Required Endpoints

| Endpoint                | Method | Contract                                                              |
| ----------------------- | ------ | --------------------------------------------------------------------- |
| `/ping`                 | GET    | Return `200` when model is loaded and healthy, `404` otherwise        |
| `/invocations`          | POST   | Accept request bytes, return prediction bytes                         |
| `/execution-parameters` | GET    | **Optional** — tells Batch Transform about concurrency/payload limits |

### The `/execution-parameters` Response

```json
{
  "MaxConcurrentTransforms": 1,
  "BatchStrategy": "SINGLE_RECORD",
  "MaxPayloadInMB": 6
}
```

### SDK Classes

- `Model` (generic, no framework prefix) — no hook protocol, just image + model data
- `Transformer` — wraps a `Model` for Batch Transform jobs
- `Predictor` — wraps an endpoint for real-time inference

### AWS Reference Architecture

```
SageMaker container
└── serve (shell script)
    ├── nginx  (reverse proxy, port 8080)
    └── gunicorn (WSGI, gevent workers, unix socket)
        └── Flask app (predictor.py)
            ├── GET  /ping
            ├── POST /invocations
            └── GET  /execution-parameters
```

### Code Snippet

```python
# predictor.py — you own the entire serving stack
import flask, io, json, os, threading
import mlflow.sklearn, pandas as pd

app = flask.Flask(__name__)

class ScoringService:
    model = None
    _lock = threading.Lock()

    @classmethod
    def load(cls):
        if cls.model is not None:
            return True
        with cls._lock:
            if cls.model is not None:  # double-check after acquiring lock
                return True
            cls.model = mlflow.sklearn.load_model("/opt/ml/model/champion")
            with open("/opt/ml/model/evaluation_report.json") as f:
                report = json.load(f)
            cls.threshold = report["optimal_threshold"]
            cls.features = report["feature_columns"]
        return True

@app.route("/ping")
def ping():
    return flask.Response("\n", status=200 if ScoringService.load() else 404)

@app.route("/execution-parameters")
def execution_parameters():
    return flask.jsonify({
        "MaxConcurrentTransforms": 1,
        "BatchStrategy": "SINGLE_RECORD",
        "MaxPayloadInMB": 6,
    })

@app.route("/invocations", methods=["POST"])
def invocations():
    ScoringService.load()
    df = pd.read_parquet(io.BytesIO(flask.request.data))
    proba = ScoringService.model.predict_proba(
        df[ScoringService.features])[:, 1]
    result = pd.DataFrame({"probability": proba})
    buf = io.BytesIO()
    result.to_parquet(buf, index=False)
    buf.seek(0)
    return flask.Response(buf.getvalue(), mimetype="application/x-parquet")
```

```python
# Orchestration
from sagemaker.model import Model

model = Model(
    image_uri="my-ecr-repo/credit-risk-inference:latest",
    model_data="s3://bucket/model.tar.gz",
    role=role,
    sagemaker_session=session,
)
transformer = model.transformer(
    instance_count=1,
    instance_type="ml.m5.xlarge",
    strategy="SingleRecord",
    accept="application/x-parquet",
)
transformer.transform("s3://bucket/input/", content_type="application/x-parquet")
```

### Tradeoffs

| Pros                                                          | Cons                                        |
| ------------------------------------------------------------- | ------------------------------------------- |
| Full control: Python version, OS packages, frameworks         | Must maintain Dockerfile + server code      |
| Any content type (parquet, protobuf, msgpack…)                | More boilerplate                            |
| Multiple model artifacts (champion + preprocessor + metadata) | Thread safety is your responsibility        |
| Custom preprocessing pipelines inside the container           | Local testing requires Docker socket access |
| Decoupled from SageMaker framework release cadence            | Image build/push adds CI overhead           |

### Real-World Use Cases

- Multi-step pipelines (preprocess → score → postprocess) inside a single container
- Binary or custom serialization formats (parquet, Arrow, protobuf)
- Models loaded from MLflow, BentoML, or other registries (not native SageMaker artifacts)
- Teams needing specific Python/CUDA/OS versions
- Organisations with strict security requirements over container contents
- Models combining multiple artifacts (ensemble, preprocessor, threshold, feature config)

---

## Pattern 3 — Estimator → `.transformer()` (Tight Training-Inference Coupling)

The training Estimator object directly creates the inference job. Internally it just
constructs the equivalent framework `Model` (Pattern 1) with the training output S3 URI
auto-wired as `model_data`. The hook protocol is identical to Pattern 1.

### SDK Classes

- `SKLearn`, `XGBoost`, `PyTorch` (Estimator subclasses)
- **SDK v3 replacement**: `ModelTrainer` for training, `ModelBuilder` for inference

### Code Snippet

```python
from sagemaker.sklearn import SKLearn

# Train
estimator = SKLearn(
    entry_point="train_and_infer.py",   # same file defines model_fn / predict_fn
    instance_type="ml.m5.xlarge",
    framework_version="1.2-1",
    role=role,
    hyperparameters={"n-estimators": 100},
)
estimator.fit({"train": "s3://bucket/train/"})

# Batch Transform — no separate model object needed
transformer = estimator.transformer(
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://bucket/predictions/",
)
transformer.transform("s3://bucket/input/", content_type="text/csv")

# Or real-time
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
)
```

### Tradeoffs

| Pros                                           | Cons                                     |
| ---------------------------------------------- | ---------------------------------------- |
| Zero S3 URI wiring — training output auto-used | Couples training and inference versions  |
| Fewer lines of orchestration code              | No control over container image          |
| Good for notebooks / experiments               | Hard to retrain and deploy independently |
| Model registry not required                    | Same constraints as Pattern 1            |

### Real-World Use Cases

- Notebook-driven experimentation where you iterate on train + infer together
- Simple pipelines where the training job output feeds directly into a transform
- Proof-of-concept work before separating training and inference concerns
- Small teams that don't need independent deploy cadences

---

## SDK v3 Note (2025+)

The SageMaker Python SDK v3 introduces breaking changes. The framework-specific classes
are replaced by unified alternatives:

| v2                                                              | v3                  |
| --------------------------------------------------------------- | ------------------- |
| `SKLearnEstimator`, `XGBoost`, `PyTorch`                        | `ModelTrainer`      |
| `SKLearnModel`, `XGBoostModel`, `PyTorchModel`, generic `Model` | `ModelBuilder`      |
| `Predictor`                                                     | `endpoint.invoke()` |

BYOC with generic `Model` / `Transformer` (Pattern 2) is unaffected — `ModelBuilder` wraps
it but the container contract (`/ping`, `/invocations`) is unchanged.

---

## Decision Guide

```
Are you using a custom content type (parquet, protobuf, msgpack)?
  YES → Pattern 2 (BYOC)
  NO  → Do you need custom OS packages / Python version?
          YES → Pattern 2 (BYOC)
          NO  → Are training and inference always deployed together?
                  YES → Pattern 3 (Estimator flow)
                  NO  → Do you need multiple artifacts or MLflow integration?
                          YES → Pattern 2 (BYOC)
                          NO  → Pattern 1 (Framework container + hooks)
```

### Summary Table

|                              | Pattern 1 — Framework Hooks                          | Pattern 2 — BYOC Flask   | Pattern 3 — Estimator Flow        |
| ---------------------------- | ---------------------------------------------------- | ------------------------ | --------------------------------- |
| **Container**                | AWS-managed                                          | Self-managed             | AWS-managed                       |
| **Serving protocol**         | `model_fn` / `input_fn` / `predict_fn` / `output_fn` | `/ping` + `/invocations` | Same as Pattern 1                 |
| **SDK class**                | `SKLearnModel`, `XGBoostModel`…                      | Generic `Model`          | `SKLearn`, `XGBoost`… (Estimator) |
| **Content types**            | numpy/csv/json (default)                             | Anything                 | numpy/csv/json (default)          |
| **Flexibility**              | Medium                                               | High                     | Low                               |
| **Boilerplate**              | Low                                                  | Medium                   | Very low                          |
| **MLflow / custom registry** | Awkward                                              | Natural                  | Awkward                           |
| **This project uses**        | —                                                    | ✅ Yes                   | —                                 |
