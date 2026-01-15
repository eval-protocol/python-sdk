import argparse
import os
import re
import sys
from pathlib import Path

from eval_protocol.cli_commands.utils import DiscoveredTest
from eval_protocol.evaluation import create_evaluation
from .secrets import handle_secrets_upload
from .utils import (
    _build_entry_point,
    _build_evaluator_dashboard_url,
    _discover_and_select_tests,
    load_module_from_file_path,
    _normalize_evaluator_id,
)


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
    env_file = getattr(args, "env_file", None)

    # Handle secrets upload (with double verification for existing secrets)
    handle_secrets_upload(
        project_root=root,
        env_file=env_file,
        non_interactive=non_interactive,
    )

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
            result, version_id = create_evaluation(
                evaluator_id=evaluator_id,
                display_name=display_name or evaluator_id,
                description=description or f"Evaluator for {qualname}",
                entry_point=entry_point,
            )
            name = result.get("name", evaluator_id) if isinstance(result, dict) else evaluator_id

            # Print success message with Fireworks dashboard link
            print(f"\n✅ Successfully uploaded evaluator: {evaluator_id}")
            if version_id:
                print(f"   Version: {version_id}")
            print("📊 View in Fireworks Dashboard:")
            dashboard_url = _build_evaluator_dashboard_url(evaluator_id)
            print(f"   {dashboard_url}\n")
        except Exception as e:
            print(f"Failed to upload {qualname}: {e}")
            exit_code = 2

    return exit_code
