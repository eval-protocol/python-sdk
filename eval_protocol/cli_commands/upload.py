import argparse
from eval_protocol.cli_commands.utils import DiscoveredTest
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

from eval_protocol.auth import get_fireworks_api_key
from eval_protocol.platform_api import create_or_update_fireworks_secret

from eval_protocol.evaluation import create_evaluation
from .utils import (
    _build_entry_point,
    _build_evaluator_dashboard_url,
    _discover_and_select_tests,
    _discover_tests,
    _ensure_account_id,
    _get_questionary_style,
    load_module_from_file_path,
    _normalize_evaluator_id,
    _prompt_select,
)


def _to_pyargs_nodeid(file_path: str, func_name: str) -> str | None:
    """Attempt to build a pytest nodeid suitable for `pytest <nodeid>`.

    Preference order:
    1) Dotted package module path with double-colon: pkg.subpkg.module::func
    2) Filesystem path with double-colon: path/to/module.py::func

    Returns dotted form when package root can be inferred (directory chain with __init__.py
    leading up to a directory contained in sys.path). Returns None if no reasonable
    nodeid can be created (should be rare).
    """
    try:
        abs_path = os.path.abspath(file_path)
        dir_path, filename = os.path.split(abs_path)
        module_base, ext = os.path.splitext(filename)
        if ext != ".py":
            # Not a python file
            return None

        # Walk up while packages have __init__.py
        segments: list[str] = [module_base]
        current = dir_path
        package_root = None
        while True:
            if os.path.isfile(os.path.join(current, "__init__.py")):
                segments.insert(0, os.path.basename(current))
                parent = os.path.dirname(current)
                # Stop if parent is not within current sys.path import roots
                if parent == current:
                    break
                current = parent
            else:
                package_root = current
                break

        # If we found a package chain, check that the package_root is importable (in sys.path)
        if package_root and any(
            os.path.abspath(sp).rstrip(os.sep) == os.path.abspath(package_root).rstrip(os.sep) for sp in sys.path
        ):
            dotted = ".".join(segments)
            return f"{dotted}::{func_name}"

        # Do not emit a dotted top-level module for non-packages; prefer path-based nodeid

        # Fallback to relative path (if under cwd) or absolute path
        cwd = os.getcwd()
        try:
            rel = os.path.relpath(abs_path, cwd)
        except Exception:
            rel = abs_path
        return f"{rel}::{func_name}"
    except Exception:
        return None


def _parse_entry(entry: str, cwd: str) -> tuple[str, str]:
    # Accept module::function, path::function, or legacy module:function
    entry = entry.strip()
    if "::" in entry:
        target, func = entry.split("::", 1)
        # Determine if target looks like a filesystem path; otherwise treat as module path
        looks_like_path = (
            "/" in target or "\\" in target or target.endswith(".py") or os.path.exists(os.path.join(cwd, target))
        )
        if looks_like_path:
            abs_path = os.path.abspath(os.path.join(cwd, target))
            return abs_path, func
        else:
            # Treat as module path for --pyargs style
            return target, func
    elif ":" in entry:
        # Legacy support: module:function → convert to module path + function
        module, func = entry.split(":", 1)
        return module, func
    else:
        raise ValueError("--entry must be in 'module::function', 'path::function', or 'module:function' format")


def _resolve_entry_to_qual_and_source(entry: str, cwd: str) -> tuple[str, str]:
    target, func = _parse_entry(entry, cwd)

    # Determine the file path to load
    if "/" in target or "\\" in target or os.path.exists(target):
        # It's a file path - convert to absolute
        if not os.path.isabs(target):
            target = os.path.abspath(os.path.join(cwd, target))
        if not target.endswith(".py"):
            target = target + ".py"
        if not os.path.isfile(target):
            raise ValueError(f"File not found: {target}")
        source_file_path = target
    else:
        # Treat dotted name as a file path
        dotted_as_path = target.replace(".", "/") + ".py"
        source_file_path = os.path.join(cwd, dotted_as_path)

    # Load the module from the file path
    module = load_module_from_file_path(source_file_path)
    module_name = getattr(module, "__name__", Path(source_file_path).stem)

    if not hasattr(module, func):
        raise ValueError(f"Function '{func}' not found in module '{module_name}'")

    qualname = f"{module_name}.{func}"
    return qualname, os.path.abspath(source_file_path) if source_file_path else ""


def _load_secrets_from_env_file(env_file_path: str) -> Dict[str, str]:
    """
    Load secrets from a .env file that should be uploaded to Fireworks.
    """
    if not os.path.exists(env_file_path):
        return {}

    # Load the .env file into a temporary environment
    env_vars = {}
    with open(env_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")  # Remove quotes
                env_vars[key] = value
    return env_vars


def _mask_secret_value(value: str) -> str:
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


def _prompt_select_secrets(
    secrets: Dict[str, str],
    secrets_from_env_file: Dict[str, str],
    non_interactive: bool,
) -> Dict[str, str]:
    """
    Prompt user to select which environment variables to upload as secrets.
    Returns the selected secrets.
    """
    if not secrets:
        return {}

    if non_interactive:
        return secrets

    # Check if running in a non-TTY environment (e.g., CI/CD)
    if not sys.stdin.isatty():
        return secrets

    try:
        import questionary

        custom_style = _get_questionary_style()

        # Build choices with source info and masked values
        choices = []
        for key, value in secrets.items():
            source = ".env" if key in secrets_from_env_file else "env"
            masked = _mask_secret_value(value)
            label = f"{key} ({source}: {masked})"
            choices.append(questionary.Choice(title=label, value=key, checked=True))

        if len(choices) == 0:
            return {}

        print("\nFound environment variables to upload as Fireworks secrets:")
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
        return _prompt_select_secrets_fallback(secrets, secrets_from_env_file)
    except KeyboardInterrupt:
        print("\n\nSecret upload cancelled.")
        return {}


def _prompt_select_secrets_fallback(
    secrets: Dict[str, str],
    secrets_from_env_file: Dict[str, str],
) -> Dict[str, str]:
    """Fallback prompt selection for when questionary is not available."""
    print("\n" + "=" * 60)
    print("Found environment variables to upload as Fireworks secrets:")
    print("=" * 60)
    print("\nTip: Install questionary for better UX: pip install questionary\n")

    secret_list = list(secrets.items())
    for idx, (key, value) in enumerate(secret_list, 1):
        source = ".env" if key in secrets_from_env_file else "env"
        masked = _mask_secret_value(value)
        print(f"  [{idx}] {key} ({source}: {masked})")

    print("\n" + "=" * 60)
    print("Enter numbers to select (comma-separated), 'all' for all, or 'none' to skip:")

    try:
        choice = input("Selection: ").strip().lower()
    except KeyboardInterrupt:
        print("\nSecret upload cancelled.")
        return {}

    if not choice or choice == "none":
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


def upload_command(args: argparse.Namespace) -> int:
    root = os.path.abspath(getattr(args, "path", "."))
    entries_arg = getattr(args, "entry", None)
    non_interactive: bool = bool(getattr(args, "yes", False))
    if entries_arg:
        entries = [e.strip() for e in re.split(r"[,\s]+", entries_arg) if e.strip()]
        selected_specs: list[tuple[str, str]] = []
        for e in entries:
            qualname, resolved_path = _resolve_entry_to_qual_and_source(e, root)
            selected_specs.append((qualname, resolved_path))
    else:
        selected_tests: list[DiscoveredTest] | None = _discover_and_select_tests(root, non_interactive=non_interactive)
        if not selected_tests:
            return 1
        selected_specs = [(t.qualname, t.file_path) for t in selected_tests]

    base_id = getattr(args, "id", None)
    display_name = getattr(args, "display_name", None)
    description = getattr(args, "description", None)
    force = bool(getattr(args, "force", False))
    env_file = getattr(args, "env_file", None)

    # Load secrets from .env file and ensure they're available on Fireworks
    try:
        fw_account_id = _ensure_account_id()

        # Determine .env file path
        if env_file:
            env_file_path = env_file
        else:
            env_file_path = os.path.join(root, ".env")

        # Load secrets from .env file
        secrets_from_file = _load_secrets_from_env_file(env_file_path)
        secrets_from_env_file = secrets_from_file.copy()  # Track what came from .env file

        # Also consider FIREWORKS_API_KEY from environment, but prefer .env value
        fw_api_key_value = get_fireworks_api_key()
        if fw_api_key_value and "FIREWORKS_API_KEY" not in secrets_from_file:
            secrets_from_file["FIREWORKS_API_KEY"] = fw_api_key_value

        if fw_account_id and secrets_from_file:
            if secrets_from_env_file and os.path.exists(env_file_path):
                print(f"Loading secrets from: {env_file_path}")

            # Prompt user to select which secrets to upload
            selected_secrets = _prompt_select_secrets(
                secrets_from_file,
                secrets_from_env_file,
                non_interactive,
            )

            if selected_secrets:
                print(f"\nUploading {len(selected_secrets)} selected secret(s) to Fireworks...")
                for secret_name, secret_value in selected_secrets.items():
                    source = ".env" if secret_name in secrets_from_env_file else "environment"
                    print(
                        f"Ensuring {secret_name} is registered as a secret on Fireworks for rollout... "
                        f"({source}: {_mask_secret_value(secret_value)})"
                    )
                    if create_or_update_fireworks_secret(
                        account_id=fw_account_id,
                        key_name=secret_name,
                        secret_value=secret_value,
                    ):
                        print(f"✓ {secret_name} secret created/updated on Fireworks.")
                    else:
                        print(f"Warning: Failed to create/update {secret_name} secret on Fireworks.")
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

    exit_code = 0
    for i, (qualname, source_file_path) in enumerate(selected_specs):
        # Generate a short default ID from just the test function name
        if base_id:
            evaluator_id = base_id
            if len(selected_specs) > 1:
                evaluator_id = f"{base_id}-{i + 1}"
        else:
            # Extract just the test function name from qualname
            test_func_name = qualname.split(".")[-1]
            # Extract source file name (e.g., "test_gpqa.py" -> "test_gpqa")
            if source_file_path:
                source_file_name = Path(source_file_path).stem
            else:
                source_file_name = "eval"
            # Create a shorter ID: filename-testname
            evaluator_id = f"{source_file_name}-{test_func_name}"

        # Normalize the evaluator ID to meet Fireworks requirements
        evaluator_id = _normalize_evaluator_id(evaluator_id)

        # Compute entry point metadata for backend as a pytest nodeid usable with `pytest <entrypoint>`
        # Always prefer a path-based nodeid to work in plain pytest environments (server may not use --pyargs)
        func_name = qualname.split(".")[-1]
        entry_point = _build_entry_point(root, source_file_path, func_name)

        print(f"\nUploading evaluator '{evaluator_id}' for {qualname.split('.')[-1]}...")
        try:
            result = create_evaluation(
                evaluator_id=evaluator_id,
                display_name=display_name or evaluator_id,
                description=description or f"Evaluator for {qualname}",
                force=force,
                entry_point=entry_point,
            )
            name = result.get("name", evaluator_id) if isinstance(result, dict) else evaluator_id

            # Print success message with Fireworks dashboard link
            print(f"\n✅ Successfully uploaded evaluator: {evaluator_id}")
            print("📊 View in Fireworks Dashboard:")
            dashboard_url = _build_evaluator_dashboard_url(evaluator_id)
            print(f"   {dashboard_url}\n")
        except Exception as e:
            print(f"Failed to upload {qualname}: {e}")
            exit_code = 2

    return exit_code
