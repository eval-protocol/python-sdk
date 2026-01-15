"""
Secret handling module for ep CLI commands.

This module provides reusable functions for loading, selecting, and uploading
secrets to Fireworks. Used by 'ep upload', 'ep create rft', and 'ep create evj'.
"""

import os
import sys
from typing import Dict

from dotenv import dotenv_values

from eval_protocol.auth import get_fireworks_api_key
from eval_protocol.platform_api import create_or_update_fireworks_secret, get_fireworks_secret
from .utils import _ensure_account_id, _get_questionary_style


def load_secrets_from_env_file(env_file_path: str) -> Dict[str, str]:
    """
    Load secrets from a .env file that should be uploaded to Fireworks.

    Uses python-dotenv's dotenv_values() for proper parsing of .env files,
    which correctly handles:
    - End-of-line comments (e.g., KEY=value # comment)
    - Quoted values (single and double quotes)
    - Escape sequences
    - Multi-line values
    """
    if not os.path.exists(env_file_path):
        return {}

    # Use dotenv_values for proper .env parsing (handles comments, quotes, etc.)
    parsed = dotenv_values(env_file_path)
    # Filter out None values and convert to Dict[str, str]
    return {k: v for k, v in parsed.items() if v is not None}


def mask_secret_value(value: str | None) -> str:
    """
    Return a masked representation of a secret showing only a small prefix/suffix.
    Example: fw_3Z*******Xgnk
    """
    try:
        if not isinstance(value, str) or not value:
            return "<empty>"
        prefix_len = 6
        suffix_len = 4
        if len(value) <= prefix_len + suffix_len:
            return value[0] + "***" + value[-1]
        return f"{value[:prefix_len]}***{value[-suffix_len:]}"
    except Exception:
        return "<masked>"


def check_existing_secrets(
    secrets: Dict[str, str],
    account_id: str,
) -> set[str]:
    """
    Check which secrets already exist on Fireworks.
    Returns a set of secret names that already exist.
    """
    existing = set()
    for secret_name in secrets.keys():
        secret = get_fireworks_secret(account_id=account_id, key_name=secret_name)
        if secret is not None:
            existing.add(secret_name)
    return existing


def prompt_select_secrets(
    secrets: Dict[str, str],
    secrets_from_env_file: Dict[str, str],
    existing_secrets: set[str],
    non_interactive: bool,
) -> Dict[str, str]:
    """
    Prompt user to select which environment variables to upload as secrets.
    Existing secrets are unchecked by default and marked with [exists].
    Returns the selected secrets.
    """
    if not secrets:
        return {}

    if non_interactive:
        # In non-interactive mode, only return new secrets (skip existing ones)
        return {k: v for k, v in secrets.items() if k not in existing_secrets}

    # Check if running in a non-TTY environment (e.g., CI/CD)
    if not sys.stdin.isatty():
        # In non-TTY, only return new secrets (skip existing ones)
        return {k: v for k, v in secrets.items() if k not in existing_secrets}

    try:
        import questionary

        custom_style = _get_questionary_style()

        # Build choices with source info and masked values
        # Existing secrets are unchecked by default
        choices = []
        for key, value in secrets.items():
            source = ".env" if key in secrets_from_env_file else "env"
            masked = mask_secret_value(value)
            is_existing = key in existing_secrets
            exists_marker = " [exists]" if is_existing else ""
            label = f"{key}{exists_marker} ({source}: {masked})"
            # Uncheck existing secrets by default
            choices.append(questionary.Choice(title=label, value=key, checked=not is_existing))

        if len(choices) == 0:
            return {}

        print("\nFound environment variables to upload as Fireworks secrets:")
        if existing_secrets:
            print("   (Secrets marked [exists] are unchecked - selecting them will override)")
        selected_keys = questionary.checkbox(
            "Select secrets to upload:",
            choices=choices,
            style=custom_style,
            pointer=">",
            instruction="(↑↓ move, space select, enter confirm)",
        ).ask()

        if selected_keys is None:
            # User cancelled with Ctrl+C
            print("\nSecret upload cancelled.")
            return {}

        return {k: v for k, v in secrets.items() if k in selected_keys}

    except ImportError:
        # Fallback to simple text-based selection
        return _prompt_select_secrets_fallback(secrets, secrets_from_env_file, existing_secrets)
    except KeyboardInterrupt:
        print("\n\nSecret upload cancelled.")
        return {}


def _prompt_select_secrets_fallback(
    secrets: Dict[str, str],
    secrets_from_env_file: Dict[str, str],
    existing_secrets: set[str],
) -> Dict[str, str]:
    """Fallback prompt selection for when questionary is not available."""
    print("\n" + "=" * 60)
    print("Found environment variables to upload as Fireworks secrets:")
    print("=" * 60)
    print("\nTip: Install questionary for better UX: pip install questionary\n")

    secret_list = list(secrets.items())
    new_indices = []
    for idx, (key, value) in enumerate(secret_list, 1):
        source = ".env" if key in secrets_from_env_file else "env"
        masked = mask_secret_value(value)
        is_existing = key in existing_secrets
        exists_marker = " [exists]" if is_existing else ""
        print(f"  [{idx}] {key}{exists_marker} ({source}: {masked})")
        if not is_existing:
            new_indices.append(idx)

    print("\n" + "=" * 60)
    if existing_secrets:
        print("Note: Secrets marked [exists] will be overridden if selected.")
    default_selection = ",".join(str(i) for i in new_indices) if new_indices else "none"
    print("Enter numbers (comma-separated), 'all' for all, or 'none' to skip.")
    print(f"Default (new secrets only): {default_selection}")

    try:
        choice = input("Selection: ").strip().lower()
    except KeyboardInterrupt:
        print("\nSecret upload cancelled.")
        return {}

    if not choice:
        # Default: only new secrets
        choice = default_selection

    if choice == "none":
        return {}

    if choice == "all":
        return secrets

    try:
        indices = [int(x.strip()) for x in choice.split(",")]
        selected = {}
        for idx in indices:
            if 1 <= idx <= len(secret_list):
                key, value = secret_list[idx - 1]
                selected[key] = value
        return selected
    except ValueError:
        print("Invalid input. Skipping secret upload.")
        return {}


def confirm_override_secrets(
    secrets_to_override: set[str],
    non_interactive: bool,
) -> set[str]:
    """
    Prompt user to confirm overriding existing secrets (double verification).
    Returns the set of secrets confirmed for override.
    """
    if not secrets_to_override:
        return set()

    if non_interactive:
        # In non-interactive mode, skip overriding existing secrets by default
        print(f"\n⚠️  Skipping {len(secrets_to_override)} existing secret(s) in non-interactive mode.")
        print("   Use interactive mode to confirm overriding existing secrets.")
        return set()

    # Check if running in a non-TTY environment
    if not sys.stdin.isatty():
        print(f"\n⚠️  Skipping {len(secrets_to_override)} existing secret(s) (non-TTY environment).")
        return set()

    print(f"\n⚠️  The following {len(secrets_to_override)} secret(s) already exist on Fireworks:")
    for name in sorted(secrets_to_override):
        print(f"   • {name}")

    try:
        import questionary

        custom_style = _get_questionary_style()

        # First confirmation
        confirm1 = questionary.confirm(
            "Do you want to override these existing secrets?",
            default=False,
            style=custom_style,
        ).ask()

        if confirm1 is None or not confirm1:
            print("Override cancelled. Existing secrets will be skipped.")
            return set()

        # Second confirmation (double verification)
        confirm2 = questionary.confirm(
            "⚠️  Are you SURE? This will permanently overwrite the existing secret values.",
            default=False,
            style=custom_style,
        ).ask()

        if confirm2 is None or not confirm2:
            print("Override cancelled. Existing secrets will be skipped.")
            return set()

        return secrets_to_override

    except ImportError:
        # Fallback to simple text-based confirmation
        return _confirm_override_secrets_fallback(secrets_to_override)
    except KeyboardInterrupt:
        print("\n\nOverride cancelled.")
        return set()


def _confirm_override_secrets_fallback(secrets_to_override: set[str]) -> set[str]:
    """Fallback confirmation for when questionary is not available."""
    print("\n" + "=" * 60)
    print("WARNING: Confirm override of existing secrets")
    print("=" * 60)

    try:
        # First confirmation
        response1 = input("Do you want to override these existing secrets? (yes/no): ").strip().lower()
        if response1 not in ("yes", "y"):
            print("Override cancelled. Existing secrets will be skipped.")
            return set()

        # Second confirmation (double verification)
        response2 = input("⚠️  Are you SURE? Type 'override' to confirm: ").strip().lower()
        if response2 != "override":
            print("Override cancelled. Existing secrets will be skipped.")
            return set()

        return secrets_to_override
    except KeyboardInterrupt:
        print("\n\nOverride cancelled.")
        return set()


def handle_secrets_upload(
    project_root: str,
    env_file: str | None,
    non_interactive: bool,
) -> None:
    """
    Main entry point for handling secrets upload flow.

    This function:
    1. Loads secrets from .env file and environment
    2. Checks which secrets already exist on Fireworks
    3. Prompts user to select secrets (existing ones unchecked by default)
    4. Requires double confirmation for overriding existing secrets
    5. Uploads the selected/confirmed secrets

    Args:
            project_root: Path to the project root directory.
            env_file: Optional path to a specific .env file (overrides default).
            non_interactive: If True, skip prompts and only upload new secrets.
    """
    try:
        fw_account_id = _ensure_account_id()

        # Determine .env file path
        if env_file:
            env_file_path = env_file
        else:
            env_file_path = os.path.join(project_root, ".env")

        # Load secrets from .env file
        secrets_from_file = load_secrets_from_env_file(env_file_path)
        secrets_from_env_file = secrets_from_file.copy()  # Track what came from .env file

        # Also consider FIREWORKS_API_KEY from environment, but prefer .env value
        fw_api_key_value = get_fireworks_api_key()
        if fw_api_key_value and "FIREWORKS_API_KEY" not in secrets_from_file:
            secrets_from_file["FIREWORKS_API_KEY"] = fw_api_key_value

        if fw_account_id and secrets_from_file:
            if secrets_from_env_file and os.path.exists(env_file_path):
                print(f"Loading secrets from: {env_file_path}")

            # Check which secrets already exist on Fireworks BEFORE prompting
            print("Checking for existing secrets on Fireworks...")
            existing_secrets = check_existing_secrets(secrets_from_file, fw_account_id)

            # Prompt user to select which secrets to upload
            # Existing secrets are unchecked by default
            selected_secrets = prompt_select_secrets(
                secrets_from_file,
                secrets_from_env_file,
                existing_secrets,
                non_interactive,
            )

            if selected_secrets:
                # Separate new secrets from existing ones that user explicitly selected
                new_secrets = {k: v for k, v in selected_secrets.items() if k not in existing_secrets}
                secrets_needing_override = {k: v for k, v in selected_secrets.items() if k in existing_secrets}

                # Confirm override for existing secrets (double verification)
                confirmed_overrides: set[str] = set()
                if secrets_needing_override:
                    confirmed_overrides = confirm_override_secrets(
                        set(secrets_needing_override.keys()),
                        non_interactive,
                    )

                # Build final list of secrets to upload
                secrets_to_upload = dict(new_secrets)
                for name in confirmed_overrides:
                    secrets_to_upload[name] = secrets_needing_override[name]

                if not secrets_to_upload:
                    print("No secrets to upload (existing secrets were not confirmed for override).")
                else:
                    print(f"\nUploading {len(secrets_to_upload)} secret(s) to Fireworks...")
                    for secret_name, secret_value in secrets_to_upload.items():
                        source = ".env" if secret_name in secrets_from_env_file else "environment"
                        action = "Overriding" if secret_name in confirmed_overrides else "Creating"
                        print(f"{action} {secret_name} on Fireworks... ({source}: {mask_secret_value(secret_value)})")
                        if create_or_update_fireworks_secret(
                            account_id=fw_account_id,
                            key_name=secret_name,
                            secret_value=secret_value,
                        ):
                            print(
                                f"✓ {secret_name} secret {'updated' if secret_name in confirmed_overrides else 'created'} on Fireworks."
                            )
                        else:
                            print(
                                f"Warning: Failed to {'update' if secret_name in confirmed_overrides else 'create'} {secret_name} secret on Fireworks."
                            )
            else:
                print("No secrets selected for upload.")
        else:
            if not fw_account_id:
                print(
                    "Warning: Could not resolve Fireworks account id from FIREWORKS_API_KEY; cannot register secrets."
                )
            if not secrets_from_file:
                print("Warning: No API keys found in environment or .env file; no secrets to register.")
    except Exception as e:
        print(f"Warning: Skipped Fireworks secret registration due to error: {e}")
