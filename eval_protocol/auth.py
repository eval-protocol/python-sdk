import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


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
            logger.debug("Resolved account id via verifyApiKey: %s", account_id)
            return account_id.strip()
        return None
    except Exception as e:
        logger.debug("Failed to verify API key for account id resolution: %s", e)
        return None
