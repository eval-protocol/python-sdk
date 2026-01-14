import logging
import os
from typing import Dict, Optional

import requests
from dotenv import dotenv_values, find_dotenv, load_dotenv

logger = logging.getLogger(__name__)


def find_dotenv_path(search_path: Optional[str] = None) -> Optional[str]:
    """
    Find the .env file path, searching .env.dev first, then .env.

    Args:
        search_path: Directory to search from. If None, uses current working directory.

    Returns:
        Path to the .env file if found, otherwise None.
    """
    # If a specific search path is provided, look there first
    if search_path:
        env_dev_path = os.path.join(search_path, ".env.dev")
        if os.path.isfile(env_dev_path):
            return env_dev_path
        env_path = os.path.join(search_path, ".env")
        if os.path.isfile(env_path):
            return env_path
        return None

    # Otherwise use find_dotenv to search up the directory tree
    env_dev_path = find_dotenv(filename=".env.dev", raise_error_if_not_found=False, usecwd=True)
    if env_dev_path:
        return env_dev_path
    env_path = find_dotenv(filename=".env", raise_error_if_not_found=False, usecwd=True)
    if env_path:
        return env_path
    return None


def get_dotenv_values(search_path: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Get all key-value pairs from the .env file.

    Args:
        search_path: Directory to search from. If None, uses current working directory.

    Returns:
        Dictionary of environment variable names to values.
    """
    dotenv_path = find_dotenv_path(search_path)
    if dotenv_path:
        return dotenv_values(dotenv_path)
    return {}


# --- Load .env files ---
# Attempt to load .env.dev first, then .env as a fallback.
# This happens when the module is imported.
# We use override=False (default) so that existing environment variables
# (e.g., set in the shell) are NOT overridden by .env files.
_DOTENV_PATH = find_dotenv_path()
if _DOTENV_PATH:
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
    logger.debug(f"eval_protocol.auth: Loaded environment variables from: {_DOTENV_PATH}")
else:
    logger.debug(
        "eval_protocol.auth: No .env.dev or .env file found. Relying on shell/existing environment variables."
    )
# --- End .env loading ---


def get_fireworks_api_key() -> Optional[str]:
    """
    Retrieves the Fireworks API key.

    Returns:
        The API key if found, otherwise None.
    """
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if api_key and api_key.strip():
        logger.debug("Using FIREWORKS_API_KEY from environment variable.")
        return api_key.strip()
    logger.debug("Fireworks API key not found in environment variables.")
    return None


def get_fireworks_account_id() -> Optional[str]:
    """
    Retrieves the Fireworks Account ID.

    Returns:
        The Account ID if found, otherwise None.
    """
    # Account id is derived from the API key (single source of truth).
    try:
        api_key_for_verify = get_fireworks_api_key()
        if api_key_for_verify:
            resolved = verify_api_key_and_get_account_id(api_key=api_key_for_verify, api_base=get_fireworks_api_base())
            if resolved:
                logger.debug("Resolved account id via verifyApiKey: %s", resolved)
                return resolved
    except Exception as e:
        logger.debug("Failed to resolve account id via verifyApiKey: %s", e)

    logger.debug("Fireworks Account ID not found via verifyApiKey.")
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
            If api_base is api.fireworks.ai, it is used directly. Otherwise, defaults to
            dev.api.fireworks.ai for the verification call.

    Returns:
        The resolved account id if verification succeeds and the header is present; otherwise None.
    """
    try:
        resolved_key = api_key or get_fireworks_api_key()
        if not resolved_key:
            return None
        provided_base = api_base or get_fireworks_api_base()
        # Use api.fireworks.ai if explicitly provided, otherwise fall back to dev
        if "api.fireworks.ai" in provided_base:
            resolved_base = provided_base
        else:
            resolved_base = "https://dev.api.fireworks.ai"

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
            logger.debug("Resolved account id via verifyApiKey: %s", account_id)
            return account_id.strip()
        return None
    except Exception as e:
        logger.debug("Failed to verify API key for account id resolution: %s", e)
        return None
