import configparser
import logging
import os
from pathlib import Path
from typing import Dict, Optional  # Added Dict

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

                    if key in ["api_key", "account_id"] and value:
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
        logger.debug(
            f"Using {key_name} from simple key-value parsing of {AUTH_INI_FILE}."
        )
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
                logger.debug(
                    f"Using {key_name} from [fireworks] section in {AUTH_INI_FILE}."
                )
                return value_from_file

        # Try default section (configparser might place items without section header here)
        if config.has_option(config.default_section, key_name):
            value_from_default = config.get(config.default_section, key_name)
            if value_from_default:
                logger.debug(
                    f"Using {key_name} from default section [{config.default_section}] in {AUTH_INI_FILE}."
                )
                return value_from_default

    except configparser.MissingSectionHeaderError:
        # This error implies the file is purely key-value, which simple parsing should have handled.
        # If simple parsing failed to get the key, then it's likely not there or malformed.
        logger.debug(
            f"{AUTH_INI_FILE} has no section headers, and simple parsing did not find {key_name}."
        )
    except configparser.Error as e_config:
        logger.warning(
            f"Configparser error reading {AUTH_INI_FILE} for {key_name}: {e_config}"
        )
    except Exception as e_general:
        logger.warning(
            f"Unexpected error reading {AUTH_INI_FILE} for {key_name}: {e_general}"
        )

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

    The base URL is sourced from the FIREWORKS_API_BASE environment variable.
    If not set, it defaults to "https://api.fireworks.ai".

    Returns:
        The API base URL.
    """
    api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    if os.environ.get("FIREWORKS_API_BASE"):
        logger.debug("Using FIREWORKS_API_BASE from environment variable.")
    else:
        logger.debug(
            f"FIREWORKS_API_BASE not set in environment, defaulting to {api_base}."
        )
    return api_base
