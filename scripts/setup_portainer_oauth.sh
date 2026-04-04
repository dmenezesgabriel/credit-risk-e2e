#!/bin/sh
# Configure Portainer CE OAuth via API to use Authelia OIDC
# Runs as portainer-init Docker service or manually via: sh scripts/setup_portainer_oauth.sh

set -eu

# Install curl (BusyBox wget doesn't support PUT method needed by Portainer API)
apk add --no-cache curl >/dev/null 2>&1

PORTAINER_URL="${PORTAINER_URL:-http://portainer:9000}"
ADMIN_USER="${PORTAINER_ADMIN_USER:-admin}"
ADMIN_PASS="${PORTAINER_ADMIN_PASS:-superpassword}"
AUTHELIA_INTERNAL_URL="${AUTHELIA_INTERNAL_URL:-http://authelia:9091}"
AUTHELIA_EXTERNAL_URL="${AUTHELIA_EXTERNAL_URL:-https://auth.app.localhost}"
PORTAINER_EXTERNAL_URL="${PORTAINER_EXTERNAL_URL:-https://portainer.app.localhost}"

MAX_RETRIES=30
RETRY_INTERVAL=5

echo "Waiting for Portainer API..."
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "${PORTAINER_URL}/api/system/status" >/dev/null 2>&1; then
        break
    fi
    if [ "$i" = "$MAX_RETRIES" ]; then
        echo "ERROR: Portainer not reachable after ${MAX_RETRIES} retries"
        exit 1
    fi
    sleep $RETRY_INTERVAL
done

echo "Authenticating with Portainer..."
AUTH_RESPONSE=$(curl -sf -H "Content-Type: application/json" \
    -d "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASS}\"}" \
    "${PORTAINER_URL}/api/auth") || {
    echo "ERROR: Failed to authenticate with Portainer"
    exit 1
}

# Extract JWT token (shell string manipulation — no jq needed in alpine)
TOKEN=$(echo "${AUTH_RESPONSE}" | sed 's/.*"jwt":"\([^"]*\)".*/\1/')

if [ -z "${TOKEN}" ] || [ "${TOKEN}" = "${AUTH_RESPONSE}" ]; then
    echo "ERROR: Failed to parse auth token"
    exit 1
fi

echo "Fetching current settings..."
SETTINGS=$(curl -sf -H "Authorization: Bearer ${TOKEN}" \
    "${PORTAINER_URL}/api/settings")

# Check if OAuth is already configured (AuthenticationMethod 3 = OAuth)
CURRENT_AUTH=$(echo "${SETTINGS}" | sed 's/.*"AuthenticationMethod":\([0-9]*\).*/\1/')
if [ "${CURRENT_AUTH}" = "3" ]; then
    echo "Portainer OAuth already configured (AuthenticationMethod=3). Skipping."
    exit 0
fi

echo "Configuring Portainer OAuth with Authelia OIDC..."

# Uses internal Authelia URL for server-to-server calls (token, userinfo)
# and external URL for browser redirects (authorization, logout)
OAUTH_PAYLOAD=$(cat <<EOF
{
    "AuthenticationMethod": 3,
    "OAuthSettings": {
        "ClientID": "portainer",
        "ClientSecret": "portainer-secret",
        "AccessTokenURI": "${AUTHELIA_INTERNAL_URL}/api/oidc/token",
        "AuthorizationURI": "${AUTHELIA_EXTERNAL_URL}/api/oidc/authorization",
        "ResourceURI": "${AUTHELIA_INTERNAL_URL}/api/oidc/userinfo",
        "RedirectURI": "${PORTAINER_EXTERNAL_URL}",
        "UserIdentifier": "preferred_username",
        "Scopes": "openid profile email groups",
        "SSO": true,
        "LogoutURI": "${AUTHELIA_EXTERNAL_URL}/api/oidc/end-session",
        "OAuthAutoCreateUsers": true,
        "DefaultTeamID": 0
    }
}
EOF
)

curl -sf -X PUT \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "${OAUTH_PAYLOAD}" \
    "${PORTAINER_URL}/api/settings" >/dev/null && {
    echo "Portainer OAuth configured successfully!"
    echo "  SSO enabled (auto-redirect to Authelia)"
    echo "  Auth URL: ${AUTHELIA_EXTERNAL_URL}/api/oidc/authorization"
    echo "  Token URL: ${AUTHELIA_INTERNAL_URL}/api/oidc/token"
} || {
    echo "WARNING: OAuth configuration may have failed. Check Portainer logs."
    exit 1
}
