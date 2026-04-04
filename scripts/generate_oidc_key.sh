#!/bin/bash
# Generate RSA 2048 private key for Authelia OIDC JWKS
# Run once: make generate-oidc-key

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_DIR="${SCRIPT_DIR}/../config/authelia/secrets"
KEY_FILE="${KEY_DIR}/oidc-jwks-rsa.pem"

mkdir -p "${KEY_DIR}"

if [ -f "${KEY_FILE}" ]; then
    echo "OIDC RSA key already exists at ${KEY_FILE}"
    echo "Delete it first if you want to regenerate."
    exit 0
fi

openssl genrsa -out "${KEY_FILE}" 2048
chmod 600 "${KEY_FILE}"
echo "Generated OIDC RSA key at ${KEY_FILE}"
