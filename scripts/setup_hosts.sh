#!/usr/bin/env bash
# Ensure *.app.localhost subdomains resolve to 127.0.0.1.
# Modern Linux with systemd-resolved handles this automatically (RFC 6761).
# This script only modifies /etc/hosts on systems that need it (WSL2, Crostini,
# or those without systemd-resolved).
set -euo pipefail

HOSTS_FILE="/etc/hosts"
MARKER="# mlops-lab app.localhost entries"
TEST_DOMAIN="auth.app.localhost"

DOMAINS=(
  auth.app.localhost
  home.app.localhost
  grafana.app.localhost
  airflow.app.localhost
  portainer.app.localhost
)

# Skip if the system already resolves *.app.localhost (systemd-resolved, etc.)
if getent ahosts "$TEST_DOMAIN" 2>/dev/null | grep -q '127\.0\.0\.1'; then
  echo "[hosts] $TEST_DOMAIN already resolves to 127.0.0.1 — no changes needed."
  exit 0
fi

# Check if entries already exist in /etc/hosts
if grep -qF "$MARKER" "$HOSTS_FILE" 2>/dev/null; then
  echo "[hosts] *.app.localhost entries already present in $HOSTS_FILE — skipping."
  exit 0
fi

# Build the entry line
ENTRY="127.0.0.1  ${DOMAINS[*]}  $MARKER"

echo "[hosts] System does not resolve *.app.localhost automatically."
echo "[hosts] Adding entries to $HOSTS_FILE (requires sudo)..."
echo "$ENTRY" | sudo tee -a "$HOSTS_FILE" > /dev/null
echo "[hosts] Done. Added: ${DOMAINS[*]}"
