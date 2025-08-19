import configparser
import logging
import os
from pathlib import Path
import time
from typing import Dict, Optional  # Added Dict

import requests

logger = logging.getLogger(__name__)

FIREWORKS_CONFIG_DIR = Path.home() / ".fireworks"
AUTH_INI_FILE = FIREWORKS_CONFIG_DIR / "auth.ini"


def _parse_simple_auth_file(file_path: Path) -> Dict[str, str]:
    """
    Parses an auth file with simple key=value lines.
    Handles comments starting with # or ;.
    Strips whitespace and basic quotes from values.
    """
    creds = {}
    if not file_path.exists():
        return creds
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove surrounding quotes if present
                    if value and (
                        (value.startswith('"') and value.endswith('"'))
                        or (value.startswith("'") and value.endswith("'"))
                    ):
                        value = value[1:-1]

                    if key in [
                        "api_key",
                        "account_id",
                        "api_base",
                        # OAuth2-related keys
                        "issuer",
                        "client_id",
                        "access_token",
                        "refresh_token",
                        "expires_at",
                        "scope",
                        "token_type",
                    ] and value:
                        creds[key] = value
    except Exception as e:
        logger.warning(f"Error during simple parsing of {file_path}: {e}")
    return creds


def _get_credential_from_config_file(key_name: str) -> Optional[str]:
    """
    Helper to get a specific credential (api_key or account_id) from auth.ini.
    Tries simple parsing first, then configparser.
    """
    if not AUTH_INI_FILE.exists():
        return None

    # 1. Try simple key-value parsing first
    simple_creds = _parse_simple_auth_file(AUTH_INI_FILE)
    if key_name in simple_creds:
        logger.debug(f"Using {key_name} from simple key-value parsing of {AUTH_INI_FILE}.")
        return simple_creds[key_name]

    # 2. Fallback to configparser if not found via simple parsing or if simple parsing failed
    #    This path will also generate the "no section headers" warning if applicable,
    #    but only if simple parsing didn't yield the key.
    try:
        config = configparser.ConfigParser()
        config.read(AUTH_INI_FILE)

        # Try [fireworks] section
        if "fireworks" in config and config.has_option("fireworks", key_name):
            value_from_file = config.get("fireworks", key_name)
            if value_from_file:
                logger.debug(f"Using {key_name} from [fireworks] section in {AUTH_INI_FILE}.")
                return value_from_file

        # Try default section (configparser might place items without section header here)
        if config.has_option(config.default_section, key_name):
            value_from_default = config.get(config.default_section, key_name)
            if value_from_default:
                logger.debug(f"Using {key_name} from default section [{config.default_section}] in {AUTH_INI_FILE}.")
                return value_from_default

    except configparser.MissingSectionHeaderError:
        # This error implies the file is purely key-value, which simple parsing should have handled.
        # If simple parsing failed to get the key, then it's likely not there or malformed.
        logger.debug(f"{AUTH_INI_FILE} has no section headers, and simple parsing did not find {key_name}.")
    except configparser.Error as e_config:
        logger.warning(f"Configparser error reading {AUTH_INI_FILE} for {key_name}: {e_config}")
    except Exception as e_general:
        logger.warning(f"Unexpected error reading {AUTH_INI_FILE} for {key_name}: {e_general}")

    return None


def get_fireworks_api_key() -> Optional[str]:
    """
    Retrieves the Fireworks API key.

    The key is sourced in the following order:
    1. FIREWORKS_API_KEY environment variable.
    2. 'api_key' from the [fireworks] section of ~/.fireworks/auth.ini.

    Returns:
        The API key if found, otherwise None.
    """
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if api_key:
        logger.debug("Using FIREWORKS_API_KEY from environment variable.")
        return api_key

    api_key_from_file = _get_credential_from_config_file("api_key")
    if api_key_from_file:
        return api_key_from_file

    logger.debug("Fireworks API key not found in environment variables or auth.ini.")
    return None


def get_fireworks_account_id() -> Optional[str]:
    """
    Retrieves the Fireworks Account ID.

    The Account ID is sourced in the following order:
    1. FIREWORKS_ACCOUNT_ID environment variable.
    2. 'account_id' from the [fireworks] section of ~/.fireworks/auth.ini.

    Returns:
        The Account ID if found, otherwise None.
    """
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if account_id:
        logger.debug("Using FIREWORKS_ACCOUNT_ID from environment variable.")
        return account_id

    account_id_from_file = _get_credential_from_config_file("account_id")
    if account_id_from_file:
        return account_id_from_file

    logger.debug("Fireworks Account ID not found in environment variables or auth.ini.")
    return None


def get_fireworks_api_base() -> str:
    """
    Retrieves the Fireworks API base URL.

    The base URL is sourced in the following order:
    1. FIREWORKS_API_BASE environment variable.
    2. 'api_base' from the [fireworks] section of ~/.fireworks/auth.ini (or simple key=val).
    3. Defaults to "https://api.fireworks.ai".

    Returns:
        The API base URL.
    """
    env_api_base = os.environ.get("FIREWORKS_API_BASE")
    if env_api_base:
        logger.debug("Using FIREWORKS_API_BASE from environment variable.")
        return env_api_base

    file_api_base = _get_credential_from_config_file("api_base")
    if file_api_base:
        logger.debug("Using api_base from auth.ini configuration.")
        return file_api_base

    default_base = "https://api.fireworks.ai"
    logger.debug(f"FIREWORKS_API_BASE not set; defaulting to {default_base}.")
    return default_base


def _get_from_env_or_file(key_name: str) -> Optional[str]:
    # 1. Check env
    env_val = os.environ.get(key_name.upper())
    if env_val:
        return env_val
    # 2. Check config file
    return _get_credential_from_config_file(key_name.lower())


def _write_auth_config(updates: Dict[str, str]) -> None:
    """Merge-write simple key=value pairs into AUTH_INI_FILE preserving existing values."""
    FIREWORKS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = _parse_simple_auth_file(AUTH_INI_FILE)
    existing.update({k: v for k, v in updates.items() if v is not None})
    lines = [f"{k}={v}" for k, v in existing.items()]
    AUTH_INI_FILE.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(AUTH_INI_FILE, 0o600)
    except Exception:
        pass


def _discover_oidc(issuer: str) -> Dict[str, str]:
    """Fetch OIDC discovery doc. Returns empty dict on failure."""
    try:
        url = issuer.rstrip("/") + "/.well-known/openid-configuration"
        resp = requests.get(url, timeout=10)
        if resp.ok:
            return resp.json()
    except Exception:
        return {}
    return {}


def _refresh_oauth_token_if_needed() -> Optional[str]:
    """Refresh OAuth access token if expired and refresh token available. Returns current/new token or None."""
    cfg = _parse_simple_auth_file(AUTH_INI_FILE)
    access_token = cfg.get("access_token")
    refresh_token = cfg.get("refresh_token")
    expires_at_str = cfg.get("expires_at")
    issuer = cfg.get("issuer") or os.environ.get("FIREWORKS_OIDC_ISSUER")
    client_id = cfg.get("client_id") or os.environ.get("FIREWORKS_OAUTH_CLIENT_ID")

    # If we have no expiry, just return access token (best effort)
    if not refresh_token or not issuer or not client_id:
        return access_token

    now = int(time.time())
    try:
        expires_at = int(expires_at_str) if expires_at_str else None
    except ValueError:
        expires_at = None

    # If not expired (with 60s buffer), return current token
    if access_token and expires_at and expires_at - 60 > now:
        return access_token

    # Attempt refresh
    discovery = _discover_oidc(issuer)
    token_endpoint = discovery.get("token_endpoint") or issuer.rstrip("/") + "/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    try:
        resp = requests.post(token_endpoint, data=data, timeout=15)
        if not resp.ok:
            logger.warning(f"OAuth token refresh failed: {resp.status_code} {resp.text[:200]}")
            return access_token
        tok = resp.json()
        new_access = tok.get("access_token")
        new_refresh = tok.get("refresh_token") or refresh_token
        expires_in = tok.get("expires_in")
        new_expires_at = str(now + int(expires_in)) if expires_in else expires_at_str
        _write_auth_config(
            {
                "access_token": new_access,
                "refresh_token": new_refresh,
                "expires_at": new_expires_at,
                "token_type": tok.get("token_type") or cfg.get("token_type") or "Bearer",
                "scope": tok.get("scope") or cfg.get("scope") or "",
            }
        )
        return new_access or access_token
    except Exception as e:
        logger.debug(f"Exception during oauth refresh: {e}")
        return access_token


def get_auth_bearer() -> Optional[str]:
    """Return a bearer token to use in Authorization.

    Priority:
    1. FIREWORKS_ACCESS_TOKEN env
    2. FIREWORKS_API_KEY env
    3. Refreshed OAuth access_token from auth.ini (if present)
    4. api_key from auth.ini
    """
    env_access = os.environ.get("FIREWORKS_ACCESS_TOKEN")
    if env_access:
        return env_access
    env_key = os.environ.get("FIREWORKS_API_KEY")
    if env_key:
        return env_key
    refreshed = _refresh_oauth_token_if_needed()
    if refreshed:
        return refreshed
    return _get_credential_from_config_file("api_key")
