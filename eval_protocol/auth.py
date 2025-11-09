import configparser
import logging
import os
from pathlib import Path
from typing import Dict, Optional  # Added Dict

import requests

logger = logging.getLogger(__name__)

# Default locations (used for tests and as fallback). Actual resolution is dynamic via _get_auth_ini_file().
FIREWORKS_CONFIG_DIR = Path.home() / ".fireworks"
AUTH_INI_FILE = FIREWORKS_CONFIG_DIR / "auth.ini"


def _get_profile_base_dir() -> Path:
    """
    Resolve the Fireworks configuration base directory following firectl behavior:
    - Default: ~/.fireworks
    - If FIREWORKS_PROFILE is set and non-empty: ~/.fireworks/profiles/<profile>
    """
    profile_name = os.environ.get("FIREWORKS_PROFILE", "").strip()
    base_dir = Path.home() / ".fireworks"
    if profile_name:
        base_dir = base_dir / "profiles" / profile_name
    return base_dir


def _get_auth_ini_file() -> Path:
    """
    Determine the auth.ini file path.
    Priority:
      1) FIREWORKS_AUTH_FILE env var when set
      2) ~/.fireworks[/profiles/<profile>]/auth.ini (profile driven)
    """
    auth_file_env = os.environ.get("FIREWORKS_AUTH_FILE")
    if auth_file_env:
        return Path(auth_file_env)
    return _get_profile_base_dir() / "auth.ini"


def _is_profile_active() -> bool:
    """
    Returns True if a specific profile or explicit auth file is active.
    In this case, profile-based credentials should take precedence over env vars.
    """
    if os.environ.get("FIREWORKS_AUTH_FILE"):
        return True
    prof = os.environ.get("FIREWORKS_PROFILE", "").strip()
    return bool(prof)


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
        with open(file_path, "r", encoding="utf-8") as f:
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

                    if key in ["api_key", "account_id"] and value:
                        creds[key] = value
    except Exception as e:
        logger.warning("Error during simple parsing of %s: %s", str(file_path), e)
    return creds


def _get_credential_from_config_file(key_name: str) -> Optional[str]:
    """
    Helper to get a specific credential (api_key or account_id) from auth.ini.
    Tries simple parsing first, then configparser.
    """
    auth_ini_path = _get_auth_ini_file()
    if not auth_ini_path.exists():
        return None

    # 1. Try simple key-value parsing first
    simple_creds = _parse_simple_auth_file(auth_ini_path)
    if key_name in simple_creds:
        logger.debug("Using %s from simple key-value parsing of %s.", key_name, str(auth_ini_path))
        return simple_creds[key_name]

    # 2. Fallback to configparser if not found via simple parsing or if simple parsing failed
    #    This path will also generate the "no section headers" warning if applicable,
    #    but only if simple parsing didn't yield the key.
    try:
        config = configparser.ConfigParser()
        config.read(auth_ini_path)

        # Try [fireworks] section
        if "fireworks" in config and config.has_option("fireworks", key_name):
            value_from_file = config.get("fireworks", key_name)
            if value_from_file:
                logger.debug("Using %s from [fireworks] section in %s.", key_name, str(auth_ini_path))
                return value_from_file

        # Try default section (configparser might place items without section header here)
        if config.has_option(config.default_section, key_name):
            value_from_default = config.get(config.default_section, key_name)
            if value_from_default:
                logger.debug(
                    "Using %s from default section [%s] in %s.",
                    key_name,
                    config.default_section,
                    str(auth_ini_path),
                )
                return value_from_default

    except configparser.MissingSectionHeaderError:
        # This error implies the file is purely key-value, which simple parsing should have handled.
        # If simple parsing failed to get the key, then it's likely not there or malformed.
        logger.debug("%s has no section headers, and simple parsing did not find %s.", str(auth_ini_path), key_name)
    except configparser.Error as e_config:
        logger.warning("Configparser error reading %s for %s: %s", str(auth_ini_path), key_name, e_config)
    except Exception as e_general:
        logger.warning("Unexpected error reading %s for %s: %s", str(auth_ini_path), key_name, e_general)

    return None


def _get_credentials_from_config_file() -> Dict[str, Optional[str]]:
    """
    Retrieve both api_key and account_id from auth.ini with a single read/parse.
    Tries simple parsing first for both keys, then falls back to configparser for any missing ones.
    Returns a dict with up to two keys: 'api_key' and 'account_id'.
    """
    results: Dict[str, Optional[str]] = {}
    auth_ini_path = _get_auth_ini_file()
    if not auth_ini_path.exists():
        return results

    # 1) Simple key=value parsing
    try:
        simple_creds = _parse_simple_auth_file(auth_ini_path)
        if "api_key" in simple_creds and simple_creds["api_key"]:
            results["api_key"] = simple_creds["api_key"]
        if "account_id" in simple_creds and simple_creds["account_id"]:
            results["account_id"] = simple_creds["account_id"]
        if "api_key" in results and "account_id" in results:
            return results
    except Exception as e:
        logger.warning("Error during simple parsing of %s: %s", str(auth_ini_path), e)

    # 2) ConfigParser for any missing keys
    try:
        config = configparser.ConfigParser()
        config.read(auth_ini_path)
        for key_name in ("api_key", "account_id"):
            if key_name in results and results[key_name]:
                continue
            if "fireworks" in config and config.has_option("fireworks", key_name):
                value_from_file = config.get("fireworks", key_name)
                if value_from_file:
                    results[key_name] = value_from_file
                    continue
            if config.has_option(config.default_section, key_name):
                value_from_default = config.get(config.default_section, key_name)
                if value_from_default:
                    results[key_name] = value_from_default
    except configparser.MissingSectionHeaderError:
        # Purely key=value file without section headers; simple parsing should have handled it already.
        logger.debug("%s has no section headers; falling back to simple parsing results.", str(auth_ini_path))
    except configparser.Error as e_config:
        logger.warning("Configparser error reading %s: %s", str(auth_ini_path), e_config)
    except Exception as e_general:
        logger.warning("Unexpected error reading %s: %s", str(auth_ini_path), e_general)

    return results


def get_fireworks_api_key() -> Optional[str]:
    """
    Retrieves the Fireworks API key.

    The key is sourced in the following order:
    1. FIREWORKS_API_KEY environment variable.
    2. 'api_key' from the [fireworks] section of ~/.fireworks/auth.ini.

    Returns:
        The API key if found, otherwise None.
    """
    # If a profile is active, prefer profile file first, then env
    if _is_profile_active():
        api_key_from_file = _get_credential_from_config_file("api_key")
        if api_key_from_file:
            return api_key_from_file
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if api_key:
            logger.debug("Using FIREWORKS_API_KEY from environment variable (profile active but file missing).")
            return api_key
    else:
        # Default behavior: env overrides file
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
    3. If an API key is available (env or auth.ini), resolve via verifyApiKey.

    Returns:
        The Account ID if found, otherwise None.
    """
    # If a profile is active, prefer profile file first, then env
    if _is_profile_active():
        creds = _get_credentials_from_config_file()
        account_id_from_file = creds.get("account_id")
        if account_id_from_file:
            return account_id_from_file
        account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
        if account_id:
            logger.debug("Using FIREWORKS_ACCOUNT_ID from environment variable (profile active but file missing).")
            return account_id
    else:
        # Default behavior: env overrides file
        account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
        if account_id:
            logger.debug("Using FIREWORKS_ACCOUNT_ID from environment variable.")
            return account_id
        creds = _get_credentials_from_config_file()
        account_id_from_file = creds.get("account_id")
        if account_id_from_file:
            return account_id_from_file

    # 3) Fallback: if API key is present, attempt to resolve via verifyApiKey (env or auth.ini)
    try:
        # Intentionally use get_fireworks_api_key to centralize precedence (env vs file)
        api_key_for_verify = get_fireworks_api_key()
        if api_key_for_verify:
            resolved = verify_api_key_and_get_account_id(api_key=api_key_for_verify, api_base=get_fireworks_api_base())
            if resolved:
                logger.debug("Using FIREWORKS_ACCOUNT_ID resolved via verifyApiKey: %s", resolved)
                return resolved
    except Exception as e:
        logger.debug("Failed to resolve FIREWORKS_ACCOUNT_ID via verifyApiKey: %s", e)

    logger.debug("Fireworks Account ID not found in environment variables, auth.ini, or via verifyApiKey.")
    return None


def get_fireworks_api_base() -> str:
    """
    Retrieves the Fireworks API base URL.

    The base URL is sourced from the FIREWORKS_API_BASE environment variable.
    If not set, it defaults to "https://api.fireworks.ai".

    Returns:
        The API base URL.
    """
    api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    if os.environ.get("FIREWORKS_API_BASE"):
        logger.debug("Using FIREWORKS_API_BASE from environment variable.")
    else:
        logger.debug("FIREWORKS_API_BASE not set in environment, defaulting to %s.", api_base)
    return api_base


def verify_api_key_and_get_account_id(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[str]:
    """
    Calls the Fireworks API verify endpoint to validate the API key and returns the
    account id from response headers when available.

    Args:
        api_key: Optional explicit API key. When None, resolves via get_fireworks_api_key().
        api_base: Optional explicit API base. When None, resolves via get_fireworks_api_base().

    Returns:
        The resolved account id if verification succeeds and the header is present; otherwise None.
    """
    try:
        resolved_key = api_key or get_fireworks_api_key()
        if not resolved_key:
            return None
        resolved_base = api_base or get_fireworks_api_base()

        from .common_utils import get_user_agent

        url = f"{resolved_base.rstrip('/')}/verifyApiKey"
        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "User-Agent": get_user_agent(),
        }
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            logger.debug("verifyApiKey returned status %s", resp.status_code)
            return None
        # Header keys could vary in case; requests provides case-insensitive dict
        account_id = resp.headers.get("x-fireworks-account-id") or resp.headers.get("X-Fireworks-Account-Id")
        if account_id and account_id.strip():
            logger.debug("Resolved FIREWORKS_ACCOUNT_ID via verifyApiKey: %s", account_id)
            return account_id.strip()
        return None
    except Exception as e:
        logger.debug("Failed to verify API key for account id resolution: %s", e)
        return None
