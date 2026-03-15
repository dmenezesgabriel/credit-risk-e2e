FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

COPY . /opt/ml/code/

WORKDIR /opt/ml/code

# SageMaker local mode calls the container with the argument "train".
# It does NOT call "train.py" — it looks for an executable named "train"
# in PATH or in the working directory.
# Solution: create a shell script named "train" (no extension) that
# delegates to train.py, then make it executable and put it in PATH.
RUN echo '#!/bin/bash\npython /opt/ml/code/train.py "$@"' > /usr/local/bin/train \
    && chmod +x /usr/local/bin/train

# No ENTRYPOINT — let SageMaker call "train" directly as the command.
# SageMaker local compose file sets: command: train