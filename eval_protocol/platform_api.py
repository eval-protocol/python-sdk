# eval_protocol/platform_api.py
import logging
import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv

from eval_protocol.auth import (
    get_fireworks_account_id,
    get_fireworks_api_base,
    get_fireworks_api_key,
)
from fireworks.types import Secret
from fireworks import Fireworks, FireworksError, NotFoundError, InternalServerError

logger = logging.getLogger(__name__)

# --- Load .env files ---
# Attempt to load .env.dev first, then .env as a fallback.
# This happens when the module is imported.
# We use override=False (default) so that existing environment variables
# (e.g., set in the shell) are NOT overridden by .env files.
ENV_DEV_PATH = find_dotenv(filename=".env.dev", raise_error_if_not_found=False, usecwd=True)
if ENV_DEV_PATH:
    load_dotenv(dotenv_path=ENV_DEV_PATH, override=False)
    logger.info(f"eval_protocol.platform_api: Loaded environment variables from: {ENV_DEV_PATH}")
else:
    ENV_PATH = find_dotenv(filename=".env", raise_error_if_not_found=False, usecwd=True)
    if ENV_PATH:
        load_dotenv(dotenv_path=ENV_PATH, override=False)
        logger.info(f"eval_protocol.platform_api: Loaded environment variables from: {ENV_PATH}")
    else:
        logger.info(
            "eval_protocol.platform_api: No .env.dev or .env file found. "
            "Relying on shell/existing environment variables."
        )
# --- End .env loading ---


class PlatformAPIError(Exception):
    """Custom exception for platform API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self) -> str:
        return f"{super().__str__()} (Status: {self.status_code}, Response: {self.response_text or 'N/A'})"


def _normalize_secret_resource_id(key_name: str) -> str:
    """
    Normalize a secret's resource ID for Fireworks paths:
    - Lowercase
    - Replace underscores with hyphens
    - Leave other characters as-is (server enforces allowed set)
    """
    return key_name.lower().replace("_", "-")


def create_or_update_fireworks_secret(
    account_id: str,
    key_name: str,  # This is the identifier for the secret, e.g., "my-eval-api-key"
    secret_value: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> bool:
    """
    Creates a new secret on the Fireworks AI platform or updates it if it already exists.
    The 'name' of the secret in Fireworks API terms is the resource path, while 'keyName' is the identifier.

    Args:
        account_id: Fireworks Account ID.
        key_name: The identifier for the secret (e.g., "WOLFRAM_ALPHA_API_KEY", "my_eval_shim_key").
        secret_value: The actual secret string.
        api_key: Fireworks API key for authenticating this request. Resolves from env/config if None.
        api_base: Fireworks API base URL. Resolves from env/config if None.

    Returns:
        True if successful, False otherwise.
    """
    resolved_api_key = api_key or get_fireworks_api_key()
    resolved_api_base = api_base or get_fireworks_api_base()
    resolved_account_id = account_id  # Must be provided
    client = Fireworks(api_key=resolved_api_key, account_id=resolved_account_id, base_url=resolved_api_base)

    if not all([resolved_api_key, resolved_api_base, resolved_account_id]):
        logger.error("Missing Fireworks API key, base URL, or account ID for creating/updating secret.")
        return False

    # Check if secret exists using GET (path uses normalized resource id)
    resource_id = _normalize_secret_resource_id(key_name)
    secret_exists = False
    try:
        secret = client.secrets.get(resource_id)
        if secret:
            secret_exists = True
            logger.info(f"Secret '{key_name}' already exists. Will attempt to update.")
    except NotFoundError:
        # Secret doesn't exist, proceed with creation
        secret_exists = False
    except InternalServerError as e:
        # As per user feedback, 500 on GET might mean not found, treat as not found
        logger.warning(
            f"Received 500 error when checking for secret '{key_name}'. Assuming it doesn't exist. Response: {e}"
        )
        secret_exists = False
    except FireworksError as e:
        logger.error(f"Error checking for secret '{key_name}': {e}")
        return False

    if secret_exists:
        # Update existing secret (PATCH)
        # Body for PATCH requires 'keyName' and 'value'.
        # Transform key_name for payload: uppercase and underscores
        payload_key_name = key_name.upper().replace("-", "_")
        # Ensure it starts with an uppercase letter (though .upper() should handle it)
        if not payload_key_name or not payload_key_name[0].isupper():
            # This case should be rare if key_name is not empty and contains letters
            logger.warning(
                f"Could not transform key_name '{key_name}' to valid starting uppercase for payload. Using default 'EP_SECRET.'"
            )
            payload_key_name = "EP_SECRET"  # Fallback, though unlikely needed with .upper()

        try:
            logger.debug(f"PATCH payload for '{key_name}': key_name={payload_key_name}")
            client.secrets.update(resource_id, key_name=payload_key_name, value=secret_value)
            logger.info(f"Successfully updated secret '{key_name}' on Fireworks platform.")
            return True
        except FireworksError as e:
            logger.error(f"Error updating secret '{key_name}': {e}")
            return False
    else:
        # Transform key_name for payload "keyName" field: uppercase and underscores
        payload_key_name = key_name.upper().replace("-", "_")
        if not payload_key_name or not payload_key_name[0].isupper():
            logger.warning(
                f"Could not transform key_name '{key_name}' to valid starting uppercase for payload. Using default 'EP_SECRET.'"
            )
            payload_key_name = "EP_SECRET"

        try:
            logger.debug(f"POST payload for '{key_name}': {payload_key_name}")
            client.secrets.create(key_name=payload_key_name, value=secret_value, name=resource_id)
            return True
        except FireworksError as e:
            logger.error(f"Error creating secret '{key_name}': {e}")
            return False


def get_fireworks_secret(
    account_id: str,
    key_name: str,  # This is the identifier for the secret
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[Secret]:
    """
    Retrieves a secret from the Fireworks AI platform by its keyName.
    Note: This typically does not return the secret's actual value for security reasons,
          but rather its metadata.
    """
    resolved_api_key = api_key or get_fireworks_api_key()
    resolved_api_base = api_base or get_fireworks_api_base()
    resolved_account_id = account_id

    if not all([resolved_api_key, resolved_api_base, resolved_account_id]):
        logger.error("Missing Fireworks API key, base URL, or account ID for getting secret.")
        return None

    client = Fireworks(api_key=resolved_api_key, account_id=resolved_account_id, base_url=resolved_api_base)
    resource_id = _normalize_secret_resource_id(key_name)

    try:
        secret = client.secrets.get(resource_id)
        if secret:
            logger.info(f"Successfully retrieved secret '{key_name}'.")
            return secret
        else:
            logger.warning(f"Secret '{key_name}' lookup succeeded but returned empty/falsy value.")
            return None
    except NotFoundError:
        logger.info(f"Secret '{key_name}' not found.")
        return None
    except InternalServerError as e:
        # As per user feedback, 500 on GET might mean not found
        logger.warning(
            f"Received 500 error when getting secret '{key_name}'. Assuming it doesn't exist. Response: {e}"
        )
        return None
    except FireworksError as e:
        logger.error(f"Error getting secret '{key_name}': {e}")
        return None


def delete_fireworks_secret(
    account_id: str,
    key_name: str,  # This is the identifier for the secret
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> bool:
    """
    Deletes a secret from the Fireworks AI platform by its keyName.
    """
    resolved_api_key = api_key or get_fireworks_api_key()
    resolved_api_base = api_base or get_fireworks_api_base()
    resolved_account_id = account_id

    if not all([resolved_api_key, resolved_api_base, resolved_account_id]):
        logger.error("Missing Fireworks API key, base URL, or account ID for deleting secret.")
        return False

    client = Fireworks(api_key=resolved_api_key, account_id=resolved_account_id, base_url=resolved_api_base)
    resource_id = _normalize_secret_resource_id(key_name)

    try:
        client.secrets.delete(resource_id, account_id=resolved_account_id)
        logger.info(f"Successfully deleted secret '{key_name}'.")
        return True
    except NotFoundError:
        logger.info(f"Secret '{key_name}' not found, nothing to delete.")
        return True
    except InternalServerError as e:
        # As per user feedback, 500 on GET might mean not found, apply same logic for DELETE
        logger.warning(
            f"Received 500 error when deleting secret '{key_name}'. Assuming it might not have existed. Response: {e}"
        )
        return True  # Consider deletion successful if it results in non-existence
    except FireworksError as e:
        logger.error(f"Error deleting secret '{key_name}': {e}")
        return False


if __name__ == "__main__":
    # Example usage for manual testing of secret management
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Note: .env file loading is now handled at the module level when platform_api.py is imported.
    # The section that was here for loading .env files specifically for __main__ has been removed
    # to rely on the module-level loading.

    # These should be set in your .env.dev, .env file (or shell environment) for this test to run
    # FIREWORKS_API_KEY="your_fireworks_api_key"
    # FIREWORKS_API_BASE="https://api.fireworks.ai" # or your dev/staging endpoint

    test_account_id = get_fireworks_account_id()
    test_api_key = get_fireworks_api_key()  # Not passed directly, functions will resolve
    test_api_base = get_fireworks_api_base()

    logger.info("Attempting to use the following configuration for testing Fireworks secrets API:")
    logger.info(f"  Resolved account id (derived from API key): {test_account_id}")
    logger.info(f"  Resolved FIREWORKS_API_BASE: {test_api_base}")
    logger.info(
        f"  Resolved FIREWORKS_API_KEY: {'********' + test_api_key[-4:] if test_api_key and len(test_api_key) > 4 else 'Not set or too short'}"
    )

    if not test_account_id or not test_api_key or not test_api_base:
        logger.error(
            "CRITICAL: FIREWORKS_API_KEY and FIREWORKS_API_BASE must be correctly set in environment or .env file to run this test."
        )
        sys.exit(1)

    test_secret_key_name = "rewardkit-test-secret-delete-me"  # Changed to be valid
    test_secret_value = "test_secret_value_12345"
    updated_secret_value = "updated_secret_value_67890"

    logger.info(f"--- Testing Fireworks Secret Management for account: {test_account_id} ---")

    # 1. Ensure it doesn't exist initially (or delete if it does from a previous failed run)
    logger.info(f"\n[Test Step 0] Attempting to delete '{test_secret_key_name}' if it exists (cleanup)...")
    delete_fireworks_secret(account_id=test_account_id, key_name=test_secret_key_name)
    retrieved = get_fireworks_secret(test_account_id, test_secret_key_name)
    if retrieved is None:
        logger.info(f"Confirmed secret '{test_secret_key_name}' does not exist before creation test.")
    else:
        logger.error(f"Secret '{test_secret_key_name}' still exists after cleanup attempt. Manual check needed.")
        # sys.exit(1) # Optional: make it fatal

    # 2. Create secret
    logger.info(f"\n[Test Step 1] Creating secret '{test_secret_key_name}' with value '{test_secret_value}'...")
    success_create: bool = create_or_update_fireworks_secret(test_account_id, test_secret_key_name, test_secret_value)
    logger.info(f"Create operation success: {success_create}")

    # 3. Get secret (to verify creation, though value won't be returned)
    logger.info(f"\n[Test Step 2] Getting secret '{test_secret_key_name}'...")
    retrieved_after_create = get_fireworks_secret(test_account_id, test_secret_key_name)
    if retrieved_after_create:
        logger.info(f"Retrieved secret metadata: {retrieved_after_create}")
        # Assert against the transformed keyName that's expected in the payload/response body
        expected_payload_key_name = test_secret_key_name.upper().replace("-", "_")
        assert retrieved_after_create.key_name == expected_payload_key_name
        # Note: value is typically not returned in GET responses for security reasons
        # The value field will be None or empty string, so we don't assert on it
        if retrieved_after_create.value:
            logger.info(f"Note: Secret value was returned (unusual): {retrieved_after_create.value[:10]}...")
    else:
        logger.error(f"Failed to retrieve secret '{test_secret_key_name}' after creation.")

    # 4. Update secret
    logger.info(f"\n[Test Step 3] Updating secret '{test_secret_key_name}' with value '{updated_secret_value}'...")
    success_update = create_or_update_fireworks_secret(test_account_id, test_secret_key_name, updated_secret_value)
    logger.info(f"Update operation success: {success_update}")
    # (Getting again won't show the value, so we assume PATCH worked if it returned True)

    # 5. Delete secret
    logger.info(f"\n[Test Step 4] Deleting secret '{test_secret_key_name}'...")
    success_delete = delete_fireworks_secret(test_account_id, test_secret_key_name)
    logger.info(f"Delete operation success: {success_delete}")

    # 6. Get secret (to verify deletion)
    logger.info(f"\n[Test Step 5] Getting secret '{test_secret_key_name}' again to confirm deletion...")
    retrieved_after_delete = get_fireworks_secret(test_account_id, test_secret_key_name)
    if retrieved_after_delete is None:
        logger.info(f"Secret '{test_secret_key_name}' successfully confirmed as deleted.")
    else:
        logger.error(f"Secret '{test_secret_key_name}' still exists after delete operation: {retrieved_after_delete}")

    logger.info("\n--- Test script finished ---")
