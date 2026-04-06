#!/usr/bin/env bash
# Ensure *.app.localhost subdomains resolve to 127.0.0.1.
# Some Linux systems (e.g. Crostini, WSL, or those without systemd-resolved)
# do not resolve *.localhost wildcards automatically.
set -euo pipefail

HOSTS_FILE="/etc/hosts"
MARKER="# mlops-lab app.localhost entries"

DOMAINS=(
  auth.app.localhost
  home.app.localhost
  grafana.app.localhost
  airflow.app.localhost
  portainer.app.localhost
)

# Check if entries already exist
if grep -qF "$MARKER" "$HOSTS_FILE" 2>/dev/null; then
  echo "[hosts] *.app.localhost entries already present in $HOSTS_FILE — skipping."
  exit 0
fi

# Build the entry line
ENTRY="127.0.0.1  ${DOMAINS[*]}  $MARKER"

echo "[hosts] Adding *.app.localhost entries to $HOSTS_FILE (requires sudo)..."
echo "$ENTRY" | sudo tee -a "$HOSTS_FILE" > /dev/null
echo "[hosts] Done. Added: ${DOMAINS[*]}"
