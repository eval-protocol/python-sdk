import argparse
import importlib.util
import inspect
import json
import os
import pkgutil
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pytest
from eval_protocol.auth import (
    get_fireworks_account_id,
    get_fireworks_api_key,
    get_fireworks_api_base,
    verify_api_key_and_get_account_id,
)
from eval_protocol.platform_api import create_or_update_fireworks_secret

from eval_protocol.evaluation import create_evaluation


@dataclass
class DiscoveredTest:
    module_path: str
    module_name: str
    qualname: str
    file_path: str
    lineno: int | None
    has_parametrize: bool
    param_count: int
    nodeids: list[str]


def _iter_python_files(root: str) -> Iterable[str]:
    # Don't follow symlinks to avoid infinite loops
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Skip common virtualenv and node paths
        if any(
            skip in dirpath
            for skip in [
                "/.venv",
                "/venv",
                "/node_modules",
                "/.git",
                "/dist",
                "/build",
                "/__pycache__",
                ".egg-info",
                "/vendor",
            ]
        ):
            continue
        # Also skip specific directories by modifying dirnames in-place
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in ["venv", "node_modules", "__pycache__", "dist", "build", "vendor"]
        ]

        for name in filenames:
            # Skip setup files, test discovery scripts, __init__, and hidden files
            if (
                name.endswith(".py")
                and not name.startswith(".")
                and not name.startswith("test_discovery")
                and name not in ["setup.py", "versioneer.py", "conf.py", "__main__.py"]
            ):
                yield os.path.join(dirpath, name)


def _is_eval_protocol_test(obj: Any) -> bool:
    # evaluation_test decorator returns a dual_mode_wrapper with _origin_func and pytest marks
    if not callable(obj):
        return False
    origin = getattr(obj, "_origin_func", None)
    if origin is None:
        return False
    # Must have pytest marks from evaluation_test
    marks = getattr(obj, "pytestmark", [])
    return len(marks) > 0


def _extract_param_info_from_marks(obj: Any) -> tuple[bool, int, list[str]]:
    """Extract parametrization info from pytest marks.

    Returns:
        (has_parametrize, param_count, param_ids)
    """
    marks = getattr(obj, "pytestmark", [])
    has_parametrize = False
    total_combinations = 0
    all_param_ids: list[str] = []

    for m in marks:
        if getattr(m, "name", "") == "parametrize":
            has_parametrize = True
            # The data is in kwargs for eval_protocol's parametrization
            kwargs = getattr(m, "kwargs", {})
            argnames = kwargs.get("argnames", m.args[0] if m.args else "")
            argvalues = kwargs.get("argvalues", m.args[1] if len(m.args) > 1 else [])
            ids = kwargs.get("ids", [])

            # Count this dimension of parameters
            if isinstance(argvalues, (list, tuple)):
                count = len(argvalues)
                total_combinations = count  # For now, just use the count from this mark

                # Use provided IDs
                if ids and isinstance(ids, (list, tuple)):
                    all_param_ids = list(ids[:count])
                else:
                    # Generate IDs based on argnames
                    if isinstance(argnames, str) and "," not in argnames:
                        # Single parameter
                        all_param_ids = [f"{argnames}={i}" for i in range(count)]
                    else:
                        # Multiple parameters
                        all_param_ids = [f"variant_{i}" for i in range(count)]

    return has_parametrize, total_combinations, all_param_ids


def _discover_tests(root: str) -> list[DiscoveredTest]:
    abs_root = os.path.abspath(root)
    if abs_root not in sys.path:
        sys.path.insert(0, abs_root)

    discovered: list[DiscoveredTest] = []

    # Collect all test functions from Python files
    for file_path in _iter_python_files(root):
        try:
            unique_name = "ep_upload_" + re.sub(r"[^a-zA-Z0-9_]", "_", os.path.abspath(file_path))
            spec = importlib.util.spec_from_file_location(unique_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
            else:
                continue
        except Exception:
            continue

        for name, obj in inspect.getmembers(module):
            if _is_eval_protocol_test(obj):
                origin = getattr(obj, "_origin_func", obj)
                try:
                    src_file = inspect.getsourcefile(origin) or file_path
                    _, lineno = inspect.getsourcelines(origin)
                except Exception:
                    src_file, lineno = file_path, None

                # Extract parametrization info from marks
                has_parametrize, param_count, param_ids = _extract_param_info_from_marks(obj)

                # Generate synthetic nodeids for display
                base_nodeid = f"{os.path.basename(file_path)}::{name}"
                if has_parametrize and param_ids:
                    nodeids = [f"{base_nodeid}[{pid}]" for pid in param_ids]
                else:
                    nodeids = [base_nodeid]

                discovered.append(
                    DiscoveredTest(
                        module_path=module.__name__,
                        module_name=module.__name__,
                        qualname=f"{module.__name__}.{name}",
                        file_path=os.path.abspath(src_file),
                        lineno=lineno,
                        has_parametrize=has_parametrize,
                        param_count=param_count,
                        nodeids=nodeids,
                    )
                )

    # Deduplicate by qualname (in case same test appears multiple times)
    by_qual: dict[str, DiscoveredTest] = {}
    for t in discovered:
        existing = by_qual.get(t.qualname)
        if not existing or t.param_count > existing.param_count:
            by_qual[t.qualname] = t
    return sorted(by_qual.values(), key=lambda x: (x.file_path, x.lineno or 0))


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

    # Check if target looks like a file path
    if "/" in target or "\\" in target or os.path.exists(target):
        # It's a file path - convert to absolute and load as module
        if not os.path.isabs(target):
            target = os.path.abspath(os.path.join(cwd, target))

        if not target.endswith(".py"):
            target = target + ".py"

        if not os.path.isfile(target):
            raise ValueError(f"File not found: {target}")

        # Import module from file path
        spec = importlib.util.spec_from_file_location(Path(target).stem, target)
        if not spec or not spec.loader:
            raise ValueError(f"Unable to load module from path: {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        module_name = spec.name
        source_file_path = target
    else:
        # Treat as module path (e.g., "my_package.my_module")
        module_name = target
        module = importlib.import_module(module_name)
        source_file_path = getattr(module, "__file__", "") or ""

    if not hasattr(module, func):
        raise ValueError(f"Function '{func}' not found in module '{module_name}'")

    qualname = f"{module_name}.{func}"
    return qualname, os.path.abspath(source_file_path) if source_file_path else ""


def _generate_ts_mode_code(test: DiscoveredTest) -> tuple[str, str]:
    # Deprecated: we no longer generate a shim; keep stub for import compatibility
    return ("", "main.py")


def _normalize_evaluator_id(evaluator_id: str) -> str:
    """
    Normalize evaluator ID to meet Fireworks requirements:
    - Only lowercase a-z, 0-9, and hyphen (-)
    - Maximum 63 characters
    """
    # Convert to lowercase
    normalized = evaluator_id.lower()

    # Replace underscores with hyphens
    normalized = normalized.replace("_", "-")

    # Remove any characters that aren't alphanumeric or hyphen
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)

    # Remove consecutive hyphens
    normalized = re.sub(r"-+", "-", normalized)

    # Remove leading/trailing hyphens
    normalized = normalized.strip("-")

    # Ensure it starts with a letter (Fireworks requirement)
    if normalized and not normalized[0].isalpha():
        normalized = "eval-" + normalized

    # Truncate to 63 characters
    if len(normalized) > 63:
        normalized = normalized[:63].rstrip("-")

    return normalized


def _format_test_choice(test: DiscoveredTest, idx: int) -> str:
    """Format a test as a choice string for display."""
    # Shorten the qualname for display
    name = test.qualname.split(".")[-1]
    location = f"{Path(test.file_path).name}:{test.lineno}" if test.lineno else Path(test.file_path).name

    if test.has_parametrize and test.param_count > 1:
        return f"{name} ({test.param_count} variants) - {location}"
    else:
        return f"{name} - {location}"


def _prompt_select_interactive(tests: list[DiscoveredTest]) -> list[DiscoveredTest]:
    """Interactive selection with arrow keys using questionary."""
    try:
        import questionary
        from questionary import Style

        # Custom style similar to Vercel CLI
        custom_style = Style(
            [
                ("qmark", "fg:#673ab7 bold"),
                ("question", "bold"),
                ("answer", "fg:#f44336 bold"),
                ("pointer", "fg:#673ab7 bold"),
                ("highlighted", "fg:#673ab7 bold"),
                ("selected", "fg:#cc5454"),
                ("separator", "fg:#cc5454"),
                ("instruction", ""),
                ("text", ""),
            ]
        )

        # Check if only one test - auto-select it
        if len(tests) == 1:
            print(f"\nFound 1 test: {_format_test_choice(tests[0], 1)}")
            confirm = questionary.confirm("Upload this test?", default=True, style=custom_style).ask()
            if confirm:
                return tests
            else:
                return []

        # Enter-only selection UX with optional multi-select via repeat
        remaining_indices = list(range(len(tests)))
        selected_indices: list[int] = []

        print("\n")
        print("Tip: Use ↑/↓ arrows to navigate and press ENTER to select.")
        print("     After selecting one, you can choose to add more.\n")

        while remaining_indices:
            # Build choices from remaining
            choices = []
            for idx, test_idx in enumerate(remaining_indices, 1):
                t = tests[test_idx]
                choice_text = _format_test_choice(t, idx)
                choices.append({"name": choice_text, "value": test_idx})

            selected = questionary.select(
                "Select an evaluation test to upload:", choices=choices, style=custom_style
            ).ask()

            if selected is None:  # Ctrl+C
                print("\nUpload cancelled.")
                return []

            if isinstance(selected, int):
                selected_indices.append(selected)
                # Remove from remaining
                if selected in remaining_indices:
                    remaining_indices.remove(selected)

                # Ask whether to add another (ENTER to finish)
                add_more = questionary.confirm("Add another?", default=False, style=custom_style).ask()
                if not add_more:
                    break
            else:
                break

        if not selected_indices:
            print("\n⚠️  No tests were selected.")
            return []

        print(f"\n✓ Selected {len(selected_indices)} test(s)")
        return [tests[i] for i in selected_indices]

    except ImportError:
        # Fallback to simpler implementation
        return _prompt_select_fallback(tests)
    except KeyboardInterrupt:
        print("\n\nUpload cancelled.")
        return []


def _prompt_select_fallback(tests: list[DiscoveredTest]) -> list[DiscoveredTest]:
    """Fallback prompt selection for when questionary is not available."""
    print("\n" + "=" * 80)
    print("Discovered evaluation tests:")
    print("=" * 80)
    print("\nTip: Install questionary for better UX: pip install questionary\n")

    for idx, t in enumerate(tests, 1):
        loc = f"{t.file_path}:{t.lineno}" if t.lineno else t.file_path
        print(f"  [{idx}] {t.qualname}")
        print(f"      Location: {loc}")

        if t.has_parametrize and t.nodeids:
            print(f"      Parameterized: {t.param_count} variant(s)")
            # Show first few variants as examples
            example_nodeids = t.nodeids[:3]
            for nodeid in example_nodeids:
                # Extract just the parameter part for display
                if "[" in nodeid:
                    param_part = nodeid.split("[", 1)[1].rstrip("]")
                    print(f"        - {param_part}")
            if len(t.nodeids) > 3:
                print(f"        ... and {len(t.nodeids) - 3} more")
        else:
            print("      Type: Single test (no parametrization)")
        print()

    print("=" * 80)
    try:
        choice = input("Enter numbers to upload (comma or space-separated), or 'all': ").strip()
    except KeyboardInterrupt:
        print("\n\nUpload cancelled.")
        return []

    if choice.lower() in ("all", "a", "*"):
        return tests

    indices: list[int] = []
    for token in re.split(r"[\s,]+", choice):
        if token.isdigit():
            n = int(token)
            if 1 <= n <= len(tests):
                indices.append(n - 1)
    indices = sorted(set(indices))
    return [tests[i] for i in indices]


def _prompt_select(tests: list[DiscoveredTest], non_interactive: bool) -> list[DiscoveredTest]:
    """Prompt user to select tests to upload."""
    if non_interactive:
        return tests

    return _prompt_select_interactive(tests)


def upload_command(args: argparse.Namespace) -> int:
    root = os.path.abspath(getattr(args, "path", "."))
    entries_arg = getattr(args, "entry", None)
    if entries_arg:
        entries = [e.strip() for e in re.split(r"[,\s]+", entries_arg) if e.strip()]
        selected_specs: list[tuple[str, str]] = []
        for e in entries:
            qualname, resolved_path = _resolve_entry_to_qual_and_source(e, root)
            selected_specs.append((qualname, resolved_path))
    else:
        print("Scanning for evaluation tests...")
        tests = _discover_tests(root)
        if not tests:
            print("No evaluation tests found.")
            print("\nHint: Make sure your tests use the @evaluation_test decorator.")
            return 1
        selected_tests = _prompt_select(tests, non_interactive=bool(getattr(args, "yes", False)))
        if not selected_tests:
            print("No tests selected.")
            return 1

        # Warn about parameterized tests
        parameterized_tests = [t for t in selected_tests if t.has_parametrize]
        if parameterized_tests:
            print("\nNote: Parameterized tests will be uploaded as a single evaluator that")
            print("      handles all parameter combinations. The evaluator will work with")
            print("      the same logic regardless of which model/parameters are used.")

        selected_specs = [(t.qualname, t.file_path) for t in selected_tests]

    base_id = getattr(args, "id", None)
    display_name = getattr(args, "display_name", None)
    description = getattr(args, "description", None)
    force = bool(getattr(args, "force", False))

    # Ensure FIREWORKS_API_KEY is available to the remote by storing it as a Fireworks secret
    try:
        fw_account_id = get_fireworks_account_id()
        fw_api_key_value = get_fireworks_api_key()
        if not fw_account_id and fw_api_key_value:
            # Attempt to verify and resolve account id from server headers
            resolved = verify_api_key_and_get_account_id(api_key=fw_api_key_value, api_base=get_fireworks_api_base())
            if resolved:
                fw_account_id = resolved
                # Propagate to environment so downstream calls use it if needed
                os.environ["FIREWORKS_ACCOUNT_ID"] = fw_account_id
                print(f"Resolved FIREWORKS_ACCOUNT_ID via API verification: {fw_account_id}")
        if fw_account_id and fw_api_key_value:
            print("Ensuring FIREWORKS_API_KEY is registered as a secret on Fireworks for rollout...")
            if create_or_update_fireworks_secret(
                account_id=fw_account_id,
                key_name="FIREWORKS_API_KEY",
                secret_value=fw_api_key_value,
            ):
                print("✓ FIREWORKS_API_KEY secret created/updated on Fireworks.")
            else:
                print("Warning: Failed to create/update FIREWORKS_API_KEY secret on Fireworks.")
        else:
            if not fw_account_id:
                print("Warning: FIREWORKS_ACCOUNT_ID not found; cannot register FIREWORKS_API_KEY secret.")
            if not fw_api_key_value:
                print("Warning: FIREWORKS_API_KEY not found locally; cannot register secret.")
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
        entry_point = None
        if source_file_path:
            # Use path relative to current working directory if possible
            abs_path = os.path.abspath(source_file_path)
            try:
                rel = os.path.relpath(abs_path, root)
            except Exception:
                rel = abs_path
            entry_point = f"{rel}::{func_name}"
        else:
            # Fallback: use filename from qualname only (rare)
            entry_point = f"{func_name}.py::{func_name}"

        print(f"\nUploading evaluator '{evaluator_id}' for {qualname.split('.')[-1]}...")
        try:
            # Always treat as a single evaluator (single-metric) even if folder has helper modules
            test_dir = os.path.dirname(source_file_path) if source_file_path else root
            metric_name = os.path.basename(test_dir) or "metric"
            result = create_evaluation(
                evaluator_id=evaluator_id,
                metric_folders=[f"{metric_name}={test_dir}"],
                display_name=display_name or evaluator_id,
                description=description or f"Evaluator for {qualname}",
                force=force,
                entry_point=entry_point,
            )
            name = result.get("name", evaluator_id) if isinstance(result, dict) else evaluator_id

            # Print success message with Fireworks dashboard link
            print(f"\n✅ Successfully uploaded evaluator: {evaluator_id}")
            print("📊 View in Fireworks Dashboard:")
            # Map API base to app host (e.g., dev.api.fireworks.ai -> dev.app.fireworks.ai)
            from urllib.parse import urlparse

            api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
            try:
                parsed = urlparse(api_base)
                host = parsed.netloc or parsed.path  # handle cases where scheme may be missing
                # Mapping rules:
                # - dev.api.fireworks.ai → dev.fireworks.ai
                # - *.api.fireworks.ai → *.app.fireworks.ai (default)
                if host.startswith("dev.api.fireworks.ai"):
                    app_host = "dev.fireworks.ai"
                elif host.startswith("api."):
                    app_host = host.replace("api.", "app.", 1)
                else:
                    app_host = host
                scheme = parsed.scheme or "https"
                dashboard_url = f"{scheme}://{app_host}/dashboard/evaluators/{evaluator_id}"
            except Exception:
                dashboard_url = f"https://app.fireworks.ai/dashboard/evaluators/{evaluator_id}"
            print(f"   {dashboard_url}")
            print()
        except Exception as e:
            print(f"Failed to upload {qualname}: {e}")
            exit_code = 2

    return exit_code
