FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

COPY . /opt/ml/code/

# SAGEMAKER_PROGRAM=train_step.py (SageMaker prepends /opt/ml/code/)
COPY steps/train_step.py /opt/ml/code/train_step.py
COPY steps/tune_step.py  /opt/ml/code/tune_step.py

WORKDIR /opt/ml/code

# train shim: reads SAGEMAKER_PROGRAM, falls back to train.py
RUN echo '#!/bin/bash' > /usr/local/bin/train && \
    echo 'SCRIPT="${SAGEMAKER_PROGRAM:-train.py}"' >> /usr/local/bin/train && \
    echo 'exec python /opt/ml/code/$SCRIPT "$@"' >> /usr/local/bin/train && \
    chmod +x /usr/local/bin/train
