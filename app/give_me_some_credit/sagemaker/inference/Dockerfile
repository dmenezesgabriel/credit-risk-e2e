FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

COPY nginx.conf predictor.py serve wsgi.py /opt/program/
RUN chmod +x /opt/program/serve

WORKDIR /opt/program
ENV PATH="/opt/program:${PATH}"
