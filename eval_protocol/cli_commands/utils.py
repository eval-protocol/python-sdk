import os
import sys
import time
import inspect
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

from ..auth import (
    get_fireworks_account_id,
    get_fireworks_api_base,
    get_fireworks_api_key,
    verify_api_key_and_get_account_id,
)
from ..fireworks_rft import _map_api_host_to_app_host


@dataclass
class DiscoveredTest:
    module_path: str
    module_name: str
    qualname: str
    file_path: str
    lineno: int | None
    has_parametrize: bool
    param_count: int
    nodeids: List[str]


def _is_eval_protocol_test(obj: Any) -> bool:
    """Return True if the given object looks like an eval_protocol evaluation test."""
    # evaluation_test decorator returns a dual_mode_wrapper with _origin_func and pytest marks
    if not callable(obj):
        return False
    origin = getattr(obj, "_origin_func", None)
    if origin is None:
        return False
    # Must have pytest marks from evaluation_test
    marks = getattr(obj, "pytestmark", [])
    # Handle pytest proxy objects (APIRemovedInV1Proxy)
    if not isinstance(marks, (list, tuple)):
        try:
            marks = list(marks) if marks else []
        except (TypeError, AttributeError):
            return False
    return len(marks) > 0


def _extract_param_info_from_marks(obj: Any) -> tuple[bool, int, list[str]]:
    """Extract parametrization info from pytest marks.

    Returns:
        (has_parametrize, param_count, param_ids)
    """
    marks = getattr(obj, "pytestmark", [])

    # Handle pytest proxy objects (APIRemovedInV1Proxy) - same as _is_eval_protocol_test
    if not isinstance(marks, (list, tuple)):
        try:
            marks = list(marks) if marks else []
        except (TypeError, AttributeError):
            marks = []

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
    """Discover eval_protocol tests under the given root directory."""
    abs_root = os.path.abspath(root)
    if abs_root not in sys.path:
        sys.path.insert(0, abs_root)

    discovered: list[DiscoveredTest] = []

    class CollectionPlugin:
        """Plugin to capture collected items without running code."""

        def __init__(self) -> None:
            self.items: list[Any] = []

        def pytest_ignore_collect(self, collection_path, config):  # type: ignore[override]
            """Ignore problematic files before pytest tries to import them."""
            # Ignore specific files
            ignored_files = ["setup.py", "versioneer.py", "conf.py", "__main__.py"]
            if collection_path.name in ignored_files:
                return True

            # Ignore hidden files (starting with .)
            if collection_path.name.startswith("."):
                return True

            # Ignore test_discovery files
            if collection_path.name.startswith("test_discovery"):
                return True

            return None

        def pytest_collection_modifyitems(self, items):  # type: ignore[override]
            """Hook called after collection is done."""
            self.items = items

    plugin = CollectionPlugin()

    # Run pytest collection only (--collect-only prevents code execution)
    # Override python_files to collect from ANY .py file
    args = [
        abs_root,
        "--collect-only",
        "-q",
        "--pythonwarnings=ignore",
        "-o",
        "python_files=*.py",  # Override to collect all .py files
    ]

    try:
        # Suppress pytest output
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pytest.main(args, plugins=[plugin])
    except Exception:
        # If pytest collection fails, fall back to empty list
        return []

    # Process collected items
    for item in plugin.items:
        if not hasattr(item, "obj"):
            continue

        obj = item.obj
        if not _is_eval_protocol_test(obj):
            continue

        origin = getattr(obj, "_origin_func", obj)
        try:
            src_file = inspect.getsourcefile(origin) or str(item.path)
            _, lineno = inspect.getsourcelines(origin)
        except Exception:
            src_file, lineno = str(item.path), None

        # Extract parametrization info from marks
        has_parametrize, param_count, param_ids = _extract_param_info_from_marks(obj)

        # Get module name and function name
        module_name = (
            item.module.__name__  # type: ignore[attr-defined]
            if hasattr(item, "module")
            else item.nodeid.split("::")[0].replace("/", ".").replace(".py", "")
        )
        func_name = item.name.split("[")[0] if "[" in item.name else item.name

        # Generate nodeids
        base_nodeid = f"{os.path.basename(src_file)}::{func_name}"
        if param_ids:
            nodeids = [f"{base_nodeid}[{pid}]" for pid in param_ids]
        else:
            nodeids = [base_nodeid]

        discovered.append(
            DiscoveredTest(
                module_path=module_name,
                module_name=module_name,
                qualname=f"{module_name}.{func_name}",
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
                ("answer", "fg:#2196f3 bold"),
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
            confirm = questionary.confirm("Select this test?", default=True, style=custom_style).ask()
            if confirm:
                return tests
            else:
                return []

        # Single-select UX
        print("\n")
        print("Tip: Use ↑/↓ arrows to navigate and press ENTER to select.\n")

        choices = []
        for idx, t in enumerate(tests, 1):
            choice_text = _format_test_choice(t, idx)
            choices.append({"name": choice_text, "value": idx - 1})

        selected = questionary.select(
            "Select an evaluation test to upload:", choices=choices, style=custom_style
        ).ask()

        if selected is None:  # Ctrl+C
            print("\nUpload cancelled.")
            return []

        print("\n✓ Selected 1 test")
        return [tests[selected]]

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
        choice = input("Enter the number to select: ").strip()
    except KeyboardInterrupt:
        print("\n\nUpload cancelled.")
        return []

    if not choice.isdigit():
        print("\n⚠️  Invalid selection.")
        return []
    n = int(choice)
    if not (1 <= n <= len(tests)):
        print("\n⚠️  Selection out of range.")
        return []
    return [tests[n - 1]]


def _prompt_select(tests: list[DiscoveredTest], non_interactive: bool) -> list[DiscoveredTest]:
    """Prompt user to select tests to upload."""
    if non_interactive:
        return tests

    return _prompt_select_interactive(tests)


def _discover_and_select_tests(project_root: str, non_interactive: bool) -> Optional[list[DiscoveredTest]]:
    """Discover evaluation tests under the given root and prompt the user to select some.

    Returns a list of selected tests, or None if discovery/selection failed or the user
    cancelled. Callers are responsible for enforcing additional constraints (e.g. exactly
    one selection).
    """
    print("Scanning for evaluation tests...")
    tests = _discover_tests(project_root)
    if not tests:
        print("No evaluation tests found.")
        print("\nHint: Make sure your tests use the @evaluation_test decorator.")
        return None

    try:
        selected_tests = _prompt_select(tests, non_interactive=non_interactive)
    except Exception:
        print("Error: Failed to open selector UI. Please pass --evaluator or --entry explicitly.")
        return None

    if not selected_tests:
        print("No tests selected.")
        return None

    return selected_tests


def _normalize_evaluator_id(evaluator_id: str) -> str:
    """
    Normalize evaluator ID to meet Fireworks requirements:
    - Only lowercase a-z, 0-9, and hyphen (-)
    - Maximum 63 characters
    """
    import re

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


def _ensure_account_id() -> Optional[str]:
    """Resolve Fireworks account id from FIREWORKS_API_KEY via verifyApiKey."""
    api_key = get_fireworks_api_key()
    if not api_key:
        return None
    return verify_api_key_and_get_account_id(api_key=api_key, api_base=get_fireworks_api_base())


def _extract_terminal_segment(resource_name: str) -> str:
    """Return the last path segment if a fully-qualified resource name is provided."""
    try:
        return resource_name.strip("/").split("/")[-1]
    except Exception:
        return resource_name


def _build_evaluator_dashboard_url(evaluator_id: str) -> str:
    """Build the evaluator dashboard URL for the given evaluator id or resource name."""
    api_base = get_fireworks_api_base()
    app_base = _map_api_host_to_app_host(api_base)
    evaluator_slug = _extract_terminal_segment(evaluator_id)
    return f"{app_base}/dashboard/evaluators/{evaluator_slug}"


def _print_links(evaluator_id: str, dataset_id: str, job_name: Optional[str]) -> None:
    """Print dashboard links for evaluator, dataset, and optional RFT job."""
    evaluator_url = _build_evaluator_dashboard_url(evaluator_id)
    print("\n📊 Dashboard Links:")
    print(f"   Evaluator: {evaluator_url}")
    if dataset_id:
        api_base = get_fireworks_api_base()
        app_base = _map_api_host_to_app_host(api_base)
        print(f"   Dataset:   {app_base}/dashboard/datasets/{dataset_id}")
    if job_name:
        # job_name likely like accounts/{account}/reinforcementFineTuningJobs/{id}
        try:
            job_id = job_name.strip().split("/")[-1]
            print(f"   RFT Job:   {app_base}/dashboard/fine-tuning/reinforcement/{job_id}")
        except Exception:
            pass


def _build_trimmed_dataset_id(evaluator_id: str) -> str:
    """Build a dataset id derived from evaluator_id, trimmed to 63 chars.

    Format: <normalized-base>-dataset-YYYYMMDDHHMMSS, where base is trimmed to fit.
    """
    base = _normalize_evaluator_id(evaluator_id)
    suffix = f"-dataset-{time.strftime('%Y%m%d%H%M%S')}"
    max_total = 63
    max_base_len = max_total - len(suffix)
    if max_base_len < 1:
        max_base_len = 1
    if len(base) > max_base_len:
        base = base[:max_base_len].rstrip("-")
        if not base:
            base = "dataset"
    # Ensure first char is a letter
    if not base:
        base = "dataset"
    if not base[0].isalpha():
        base = f"eval-{base}"
        if len(base) > max_base_len:
            base = base[:max_base_len]
            base = base.rstrip("-") or "dataset"
    return f"{base}{suffix}"


def _resolve_selected_test(
    project_root: str,
    evaluator_id: Optional[str],
    selected_tests: Optional[list[DiscoveredTest]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve a single test's source file path and function name to use downstream.
    Priority:
      1) If selected_tests provided and length == 1, use it.
      2) Else discover tests; if exactly one test, use it.
      3) Else, if evaluator_id provided, match by normalized '<file-stem>-<func-name>'.
    Returns: (file_path, func_name) or (None, None) if unresolved.
    """
    try:
        tests = selected_tests if selected_tests is not None else _discover_tests(project_root)
        if not tests:
            return None, None
        if len(tests) == 1:
            return tests[0].file_path, tests[0].qualname.split(".")[-1]
        if evaluator_id:
            for t in tests:
                func_name = t.qualname.split(".")[-1]
                source_file_name = os.path.splitext(os.path.basename(t.file_path))[0]
                candidate = _normalize_evaluator_id(f"{source_file_name}-{func_name}")
                if candidate == evaluator_id:
                    return t.file_path, func_name
        return None, None
    except Exception:
        return None, None


def _build_entry_point(project_root: str, source_file_path: Optional[str], func_name: str) -> str:
    """Build a pytest-style entry point (path::func) relative to the given root."""
    if source_file_path:
        abs_path = os.path.abspath(source_file_path)
        try:
            rel = os.path.relpath(abs_path, project_root)
        except Exception:
            rel = abs_path
        return f"{rel}::{func_name}"
    # Fallback: use filename only
    return f"{func_name}.py::{func_name}"
