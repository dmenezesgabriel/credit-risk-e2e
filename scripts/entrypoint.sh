#!/usr/bin/env bash
set -e

python scripts/setup_uv_kernel.py

exec "$@"
