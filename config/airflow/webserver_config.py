"""
Airflow FAB Auth Manager — Authelia OIDC Configuration

Docs: https://airflow.apache.org/docs/apache-airflow-providers-fab/stable/auth-manager/webserver-authentication.html
"""

import logging
import os
from typing import Any

from airflow.providers.fab.auth_manager.security_manager.override import (
    FabAirflowSecurityManagerOverride,
)
from flask_appbuilder.security.manager import AUTH_OAUTH

log = logging.getLogger(__name__)

# --- Auth type ---
AUTH_TYPE = AUTH_OAUTH
AUTH_USER_REGISTRATION = True
AUTH_USER_REGISTRATION_ROLE = "Viewer"
AUTH_ROLES_SYNC_AT_LOGIN = True

# --- Role mapping: Authelia group name -> Airflow role(s) ---
AUTH_ROLES_MAPPING = {
    "admins": ["Admin"],
    "airflow-op": ["Op"],
    "airflow-user": ["User"],
    "airflow-viewer": ["Viewer"],
}

# --- OIDC endpoints ---
# Browser redirects use the external URL; server-to-server calls use the
# internal Docker network URL to avoid TLS trust issues.
_AUTHELIA_EXTERNAL = os.getenv(
    "AUTHELIA_EXTERNAL_URL", "https://auth.app.localhost"
)
_AUTHELIA_INTERNAL = os.getenv("AUTHELIA_INTERNAL_URL", "http://authelia:9091")

OAUTH_PROVIDERS = [
    {
        "name": "authelia",
        "icon": "fa-key",
        "token_key": "access_token",
        "remote_app": {
            "client_id": "airflow",
            "client_secret": "airflow-secret",
            # Do NOT use server_metadata_url: the internal URL returns
            # issuer=http://authelia:9091 but the ID-token iss claim is
            # https://auth.app.localhost, causing an iss mismatch.
            "api_base_url": f"{_AUTHELIA_INTERNAL}/api/oidc/",
            "access_token_url": f"{_AUTHELIA_INTERNAL}/api/oidc/token",
            "authorize_url": f"{_AUTHELIA_EXTERNAL}/api/oidc/authorization",
            "jwks_uri": f"{_AUTHELIA_INTERNAL}/jwks.json",
            "userinfo_endpoint": f"{_AUTHELIA_INTERNAL}/api/oidc/userinfo",
            "client_kwargs": {
                "scope": "openid profile email groups",
            },
        },
    },
]


# --- Custom security manager: extract user info from Authelia OIDC token ---
class AutheliaSecurityManager(FabAirflowSecurityManagerOverride):
    def get_oauth_user_info(
        self, provider: str, resp: Any
    ) -> dict[str, str | list[str]]:
        if provider != "authelia":
            return {}

        me = self.appbuilder.sm.oauth_remotes[provider].get("userinfo")
        user_data = me.json() if me else {}

        groups = user_data.get("groups", [])
        if not groups:
            groups = ["airflow-viewer"]

        name = user_data.get("name", "")
        parts = name.split(" ", 1) if name else ["", ""]

        log.info(
            "Authelia OIDC login: user=%s groups=%s",
            user_data.get("preferred_username"),
            groups,
        )

        return {
            "username": user_data.get("preferred_username", "unknown"),
            "email": user_data.get("email", ""),
            "first_name": parts[0],
            "last_name": parts[1] if len(parts) > 1 else "",
            "role_keys": groups,
        }


SECURITY_MANAGER_CLASS = AutheliaSecurityManager

# --- Flask session key (reuse Airflow's secret) ---
SECRET_KEY = os.getenv(
    "AIRFLOW__API_SERVER__SECRET_KEY", "your-super-secret-key-change-me"
)
