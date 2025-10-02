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


def _parse_entry(entry: str, cwd: str) -> tuple[str, str]:
    # Accept module:function or path::function
    entry = entry.strip()
    if "::" in entry:
        path_part, func = entry.split("::", 1)
        abs_path = os.path.abspath(os.path.join(cwd, path_part))
        module_name = Path(abs_path).stem
        return abs_path, func
    elif ":" in entry:
        module, func = entry.split(":", 1)
        return module, func
    else:
        raise ValueError("--entry must be in 'module:function' or 'path::function' format")


def _generate_ts_mode_code_from_entry(entry: str, cwd: str) -> tuple[str, str, str]:
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
    else:
        # Treat as module path (e.g., "my_package.my_module")
        module_name = target
        module = importlib.import_module(module_name)

    if not hasattr(module, func):
        raise ValueError(f"Function '{func}' not found in module '{module_name}'")

    qualname = f"{module_name}.{func}"
    code, file_name = _generate_ts_mode_code(
        DiscoveredTest(
            module_path=module_name,
            module_name=module_name,
            qualname=qualname,
            file_path=getattr(module, "__file__", module_name),
            lineno=None,
            has_parametrize=False,
            param_count=0,
            nodeids=[],
        )
    )
    return code, file_name, qualname


def _generate_ts_mode_code(test: DiscoveredTest) -> tuple[str, str]:
    # Generate a minimal main.py that imports the test module and calls the function
    module = test.module_name
    func = test.qualname.split(".")[-1]
    code = f"""
from typing import Any, Dict, List, Optional, Union

from eval_protocol.models import EvaluationRow, Message
from {module} import {func} as _ep_test

def evaluate(messages: List[Dict[str, Any]], ground_truth: Optional[Union[str, List[Dict[str, Any]]]] = None, tools=None, **kwargs):
    row = EvaluationRow(messages=[Message(**m) for m in messages], ground_truth=ground_truth)
    result = _ep_test(row)  # Supports sync/async via decorator's dual-mode
    if hasattr(result, "__await__"):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(result)
    if result.evaluation_result is None:
        return {{"score": 0.0, "reason": "No evaluation_result set"}}
    out = {{
        "score": float(result.evaluation_result.score or 0.0),
        "reason": result.evaluation_result.reason,
        "metrics": {{k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in (result.evaluation_result.metrics or {{}}).items()}},
    }}
    return out
"""
    return (code, "main.py")


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

        # Create choices with nice formatting
        choices = []
        for idx, test in enumerate(tests, 1):
            choice_text = _format_test_choice(test, idx)
            choices.append({"name": choice_text, "value": idx - 1, "checked": False})

        print("\n")
        print("💡 Tip: Use ↑/↓ arrows to navigate, SPACE to select/deselect, ENTER when done")
        print("        You can select multiple tests!\n")
        selected_indices = questionary.checkbox(
            "Select evaluation tests to upload:",
            choices=choices,
            style=custom_style,
        ).ask()

        if selected_indices is None:  # User pressed Ctrl+C
            print("\nUpload cancelled.")
            return []

        if not selected_indices:
            print("\n⚠️  No tests were selected.")
            print("   Remember: Use SPACE bar to select tests, then press ENTER to confirm.")
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
        selected_specs: list[tuple[str, str, str, str]] = []
        for e in entries:
            code, file_name, qualname = _generate_ts_mode_code_from_entry(e, root)
            # For --entry mode, extract file path from the entry
            file_path = e.split("::")[0] if "::" in e else ""
            selected_specs.append((code, file_name, qualname, file_path))
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

        selected_specs = []
        for t in selected_tests:
            code, file_name = _generate_ts_mode_code(t)
            # Store test info for better ID generation
            selected_specs.append((code, file_name, t.qualname, t.file_path))

    base_id = getattr(args, "id", None)
    display_name = getattr(args, "display_name", None)
    description = getattr(args, "description", None)
    force = bool(getattr(args, "force", False))

    exit_code = 0
    for i, (code, file_name, qualname, source_file_path) in enumerate(selected_specs):
        # Use ts_mode to upload evaluator
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

        print(f"\nUploading evaluator '{evaluator_id}' for {qualname.split('.')[-1]}...")
        try:
            result = create_evaluation(
                evaluator_id=evaluator_id,
                python_code_to_evaluate=code,
                python_file_name_for_code=file_name,
                criterion_name_for_code=qualname,
                criterion_description_for_code=description or f"Evaluator for {qualname}",
                display_name=display_name or evaluator_id,
                description=description or f"Evaluator for {qualname}",
                force=force,
            )
            name = result.get("name", evaluator_id) if isinstance(result, dict) else evaluator_id

            # Print success message with Fireworks dashboard link
            print(f"\n✅ Successfully uploaded evaluator: {evaluator_id}")
            print("📊 View in Fireworks Dashboard:")
            print(f"   https://app.fireworks.ai/dashboard/evaluators/{evaluator_id}")
            print()
        except Exception as e:
            print(f"Failed to upload {qualname}: {e}")
            exit_code = 2

    return exit_code
