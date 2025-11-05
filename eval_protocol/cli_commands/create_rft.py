import json
import os
import sys
import time
import argparse
from typing import Any, Dict, Optional

import requests

from ..auth import (
    get_fireworks_account_id,
    get_fireworks_api_base,
    get_fireworks_api_key,
    verify_api_key_and_get_account_id,
)
from ..common_utils import get_user_agent
from ..fireworks_rft import (
    _map_api_host_to_app_host,
    build_default_output_model,
    create_dataset_from_jsonl,
    create_reinforcement_fine_tuning_job,
)
from .upload import _discover_tests, _normalize_evaluator_id, _resolve_entry_to_qual_and_source


def _ensure_account_id() -> Optional[str]:
    account_id = get_fireworks_account_id()
    api_key = get_fireworks_api_key()
    if not account_id and api_key:
        resolved = verify_api_key_and_get_account_id(api_key=api_key, api_base=get_fireworks_api_base())
        if resolved:
            os.environ["FIREWORKS_ACCOUNT_ID"] = resolved
            return resolved
    return account_id


def _extract_terminal_segment(resource_name: str) -> str:
    """Return the last path segment if a fully-qualified resource name is provided."""
    try:
        return resource_name.strip("/").split("/")[-1]
    except Exception:
        return resource_name


def _print_links(evaluator_id: str, dataset_id: str, job_name: Optional[str]) -> None:
    api_base = get_fireworks_api_base()
    app_base = _map_api_host_to_app_host(api_base)
    print("\n📊 Dashboard Links:")
    evaluator_slug = _extract_terminal_segment(evaluator_id)
    print(f"   Evaluator: {app_base}/dashboard/evaluators/{evaluator_slug}")
    if dataset_id:
        print(f"   Dataset:   {app_base}/dashboard/datasets/{dataset_id}")
    if job_name:
        # job_name likely like accounts/{account}/reinforcementFineTuningJobs/{id}
        try:
            job_id = job_name.strip().split("/")[-1]
            print(f"   RFT Job:   {app_base}/dashboard/fine-tuning/reinforcement/{job_id}")
        except Exception:
            pass


def _auto_find_jsonl(cwd: str) -> Optional[str]:
    """Find a reasonable JSONL dataset file in the current project.

    Priority order:
    - dataset.jsonl in cwd
    - data/dataset.jsonl
    - first *.jsonl under cwd (depth-first, skipping common vendor/venv/build dirs)
    Returns a RELATIVE path from cwd if possible.
    """
    # Direct candidates
    direct_candidates = [
        os.path.join(cwd, "dataset.jsonl"),
        os.path.join(cwd, "data", "dataset.jsonl"),
    ]
    for p in direct_candidates:
        if os.path.isfile(p):
            try:
                return os.path.relpath(p, cwd)
            except Exception:
                return p

    # Walk and find any .jsonl
    skip_dirs = {".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".git", "vendor"}
    for dirpath, dirnames, filenames in os.walk(cwd):
        # prune
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for name in sorted(filenames):
            if name.endswith(".jsonl"):
                candidate = os.path.join(dirpath, name)
                try:
                    return os.path.relpath(candidate, cwd)
                except Exception:
                    return candidate
    return None


def _extract_jsonl_from_dataloader(test_file_path: str, test_func_name: str) -> Optional[str]:
    """Import the test module and extract a JSONL path from data_loaders param if present.

    Looks for a pytest.mark.parametrize with argnames containing 'data_loaders' and attempts to
    find an object with attribute 'jsonl_path'. If a relative path is found, it is resolved
    relative to the directory of the test file.
    """
    try:
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(Path(test_file_path).stem, test_file_path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        if not hasattr(module, test_func_name):
            return None
        wrapper = getattr(module, test_func_name)
        marks = getattr(wrapper, "pytestmark", [])
        for m in marks:
            if getattr(m, "name", "") == "parametrize":
                kwargs = getattr(m, "kwargs", {})
                argnames = kwargs.get("argnames", (m.args[0] if m.args else []))
                argvalues = kwargs.get("argvalues", (m.args[1] if len(m.args) > 1 else []))
                # Normalize argnames to list
                if isinstance(argnames, str):
                    names_list = [n.strip() for n in argnames.split(",") if n.strip()]
                else:
                    names_list = list(argnames)
                if "data_loaders" not in names_list:
                    continue
                idx = names_list.index("data_loaders")
                # argvalues is a list of tuples/values aligned with argnames
                for val in argvalues:
                    # Normalize to tuple
                    if not isinstance(val, (tuple, list)):
                        params = (val,)
                    else:
                        params = tuple(val)
                    if idx >= len(params):
                        continue
                    dataloaders_obj = params[idx]
                    # May be a list or single loader
                    candidates = (
                        list(dataloaders_obj) if isinstance(dataloaders_obj, (list, tuple)) else [dataloaders_obj]
                    )
                    for dl in candidates:
                        jsonl_path = getattr(dl, "jsonl_path", None)
                        if isinstance(jsonl_path, str) and jsonl_path:
                            if os.path.isabs(jsonl_path):
                                return jsonl_path
                            base_dir = os.path.dirname(os.path.abspath(test_file_path))
                            return os.path.abspath(os.path.join(base_dir, jsonl_path))
        return None
    except Exception:
        return None


def _extract_jsonl_from_input_dataset(test_file_path: str, test_func_name: str) -> Optional[str]:
    """Import the test module and extract a JSONL path from input_dataset (dataset_path) param if present.

    Looks for a pytest.mark.parametrize with argnames containing 'dataset_path' and extracts the
    first dataset path value. If a relative path is found, it is resolved relative to the directory
    of the test file.
    """
    try:
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(Path(test_file_path).stem, test_file_path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        if not hasattr(module, test_func_name):
            return None
        wrapper = getattr(module, test_func_name)
        marks = getattr(wrapper, "pytestmark", [])
        for m in marks:
            if getattr(m, "name", "") == "parametrize":
                kwargs = getattr(m, "kwargs", {})
                argnames = kwargs.get("argnames", (m.args[0] if m.args else []))
                argvalues = kwargs.get("argvalues", (m.args[1] if len(m.args) > 1 else []))
                # Normalize argnames to list
                if isinstance(argnames, str):
                    names_list = [n.strip() for n in argnames.split(",") if n.strip()]
                else:
                    names_list = list(argnames)
                if "dataset_path" not in names_list:
                    continue
                idx = names_list.index("dataset_path")
                # argvalues is a list of tuples/values aligned with argnames
                # Get the first value (first test case)
                if argvalues:
                    val = argvalues[0]
                    # Normalize to tuple
                    if not isinstance(val, (tuple, list)):
                        params = (val,)
                    else:
                        params = tuple(val)
                    if idx < len(params):
                        dataset_path = params[idx]
                        # dataset_path is typically a string, but could be a list if combine_datasets=True
                        if isinstance(dataset_path, (list, tuple)) and len(dataset_path) > 0:
                            dataset_path = dataset_path[0]
                        if isinstance(dataset_path, str) and dataset_path:
                            if os.path.isabs(dataset_path):
                                return dataset_path
                            base_dir = os.path.dirname(os.path.abspath(test_file_path))
                            resolved = os.path.abspath(os.path.join(base_dir, dataset_path))
                            if os.path.isfile(resolved):
                                return resolved
                            # Try resolving from project root if relative to test file doesn't work
                            if not os.path.isabs(dataset_path):
                                # Try resolving from current working directory
                                cwd_path = os.path.abspath(os.path.join(os.getcwd(), dataset_path))
                                if os.path.isfile(cwd_path):
                                    return cwd_path
        return None
    except Exception:
        return None


def _build_trimmed_dataset_id(evaluator_id: str) -> str:
    """Build a dataset id derived from evaluator_id, trimmed to 63 chars.

    Format: <normalized-base>-dataset-YYYYMMDDHHMMSS, where base is trimmed to fit.
    """
    # Normalize base similarly to evaluator id rules
    from .upload import _normalize_evaluator_id  # local import to avoid cycle at module import time

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
    if not base[0].isalpha():
        base = f"eval-{base}"
        if len(base) > max_base_len:
            base = base[:max_base_len]
            base = base.rstrip("-") or "dataset"
    return f"{base}{suffix}"


def _auto_select_evaluator_id(cwd: str) -> Optional[str]:
    # Try local traces
    traces_dir = os.path.join(cwd, ".eval_protocol", "evaluators")
    if os.path.isdir(traces_dir):
        candidates = [f[:-5] for f in os.listdir(traces_dir) if f.endswith(".json")]
        if len(candidates) == 1:
            return candidates[0]
    # Fall back to discovering a single evaluation_test
    tests = _discover_tests(cwd)
    if len(tests) == 1:
        qualname, source_file_path = tests[0].qualname, tests[0].file_path
        test_func_name = qualname.split(".")[-1]
        source_file_name = os.path.splitext(os.path.basename(source_file_path))[0]
        evaluator_id = _normalize_evaluator_id(f"{source_file_name}-{test_func_name}")
        return evaluator_id
    return None


def _poll_evaluator_status(
    evaluator_resource_name: str, api_key: str, api_base: str, timeout_minutes: int = 10
) -> bool:
    """
    Poll evaluator status until it becomes ACTIVE or times out.

    Args:
        evaluator_resource_name: Full evaluator resource name (e.g., accounts/xxx/evaluators/yyy)
        api_key: Fireworks API key
        api_base: Fireworks API base URL
        timeout_minutes: Maximum time to wait in minutes

    Returns:
        True if evaluator becomes ACTIVE, False if timeout or BUILD_FAILED
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    check_url = f"{api_base}/v1/{evaluator_resource_name}"
    timeout_seconds = timeout_minutes * 60
    poll_interval = 10  # seconds
    start_time = time.time()

    print(f"Polling evaluator status (timeout: {timeout_minutes}m, interval: {poll_interval}s)...")

    while time.time() - start_time < timeout_seconds:
        try:
            response = requests.get(check_url, headers=headers, timeout=30)
            response.raise_for_status()

            evaluator_data = response.json()
            state = evaluator_data.get("state", "STATE_UNSPECIFIED")
            status = evaluator_data.get("status", "")

            if state == "ACTIVE":
                print("✅ Evaluator is ACTIVE and ready!")
                return True
            elif state == "BUILD_FAILED":
                print(f"❌ Evaluator build failed. Status: {status}")
                return False
            elif state == "BUILDING":
                elapsed_minutes = (time.time() - start_time) / 60
                print(f"⏳ Evaluator is still building... ({elapsed_minutes:.1f}m elapsed)")
            else:
                print(f"⏳ Evaluator state: {state}, status: {status}")

        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to check evaluator status: {e}")

        # Wait before next poll
        time.sleep(poll_interval)

    # Timeout reached
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"⏰ Timeout after {elapsed_minutes:.1f}m - evaluator is not yet ACTIVE")
    return False


def create_rft_command(args) -> int:
    evaluator_id: Optional[str] = getattr(args, "evaluator_id", None)
    non_interactive: bool = bool(getattr(args, "yes", False))
    dry_run: bool = bool(getattr(args, "dry_run", False))
    force: bool = bool(getattr(args, "force", False))

    api_key = get_fireworks_api_key()
    if not api_key:
        print("Error: FIREWORKS_API_KEY not set.")
        return 1

    account_id = _ensure_account_id()
    if not account_id:
        print("Error: FIREWORKS_ACCOUNT_ID not set and could not be resolved.")
        return 1

    api_base = get_fireworks_api_base()

    # Resolve evaluator id if omitted
    project_root = os.getcwd()
    if not evaluator_id:
        evaluator_id = _auto_select_evaluator_id(project_root)
        if not evaluator_id:
            print("Error: Could not infer evaluator id. Provide --evaluator-id or run 'eval-protocol upload' first.")
            return 1

    # Resolve evaluator resource name to fully-qualified format required by API
    evaluator_resource_name = f"accounts/{account_id}/evaluators/{evaluator_id}"

    # Ensure evaluator exists by invoking the upload flow programmatically
    try:
        from .upload import upload_command

        tests = _discover_tests(project_root)
        selected_entry: Optional[str] = None
        if len(tests) == 1:
            func_name = tests[0].qualname.split(".")[-1]
            abs_path = os.path.abspath(tests[0].file_path)
            try:
                rel = os.path.relpath(abs_path, project_root)
            except Exception:
                rel = abs_path
            selected_entry = f"{rel}::{func_name}"
        else:
            # Try to match evaluator_id to a discovered test's normalized ID
            for t in tests:
                func_name = t.qualname.split(".")[-1]
                source_file_name = os.path.splitext(os.path.basename(t.file_path))[0]
                candidate = _normalize_evaluator_id(f"{source_file_name}-{func_name}")
                if candidate == evaluator_id:
                    abs_path = os.path.abspath(t.file_path)
                    try:
                        rel = os.path.relpath(abs_path, project_root)
                    except Exception:
                        rel = abs_path
                    selected_entry = f"{rel}::{func_name}"
                    break

        upload_args = argparse.Namespace(
            path=project_root,
            entry=selected_entry,
            id=evaluator_id,
            display_name=None,
            description=None,
            force=force,  # Pass through the --force flag
            yes=True,
            env_file=None,  # Add the new env_file parameter
        )

        if force:
            print(f"🔄 Force flag enabled - will overwrite existing evaluator '{evaluator_id}'")

        rc = upload_command(upload_args)
        if rc == 0:
            print(f"✓ Uploaded/ensured evaluator: {evaluator_id}")

            # Poll for evaluator status
            print(f"Waiting for evaluator '{evaluator_id}' to become ACTIVE...")
            is_active = _poll_evaluator_status(
                evaluator_resource_name=evaluator_resource_name, api_key=api_key, api_base=api_base, timeout_minutes=10
            )

            if not is_active:
                # Print helpful message with dashboard link
                app_base = _map_api_host_to_app_host(api_base)
                evaluator_slug = _extract_terminal_segment(evaluator_id)
                dashboard_url = f"{app_base}/dashboard/evaluators/{evaluator_slug}"

                print("\n❌ Evaluator is not ready within the timeout period.")
                print(f"📊 Please check the evaluator status at: {dashboard_url}")
                print("   Wait for it to become ACTIVE, then run 'eval-protocol create rft' again.")
                return 1
        else:
            print("Warning: Evaluator upload did not complete successfully; proceeding to RFT creation.")
    except Exception as e:
        print(f"Warning: Failed to upload evaluator automatically: {e}")

    # Determine dataset id and materialization path
    dataset_id = getattr(args, "dataset_id", None)
    dataset_jsonl = getattr(args, "dataset_jsonl", None)
    dataset_display_name = getattr(args, "dataset_display_name", None)
    dataset_builder = getattr(args, "dataset_builder", None)  # accepted but unused in simplified flow

    if not dataset_id:
        # Prefer explicit --dataset-jsonl, else attempt to extract from data loader or input_dataset of the single discovered test
        if not dataset_jsonl:
            tests = _discover_tests(project_root)
            if len(tests) == 1:
                func_name = tests[0].qualname.split(".")[-1]
                # Try data_loaders first (existing behavior)
                dataset_jsonl = _extract_jsonl_from_dataloader(tests[0].file_path, func_name)
                if dataset_jsonl:
                    # Display relative path for readability
                    try:
                        rel = os.path.relpath(dataset_jsonl, project_root)
                    except Exception:
                        rel = dataset_jsonl
                    print(f"✓ Using JSONL from data loader: {rel}")
                else:
                    # Fall back to input_dataset (dataset_path)
                    dataset_jsonl = _extract_jsonl_from_input_dataset(tests[0].file_path, func_name)
                    if dataset_jsonl:
                        # Display relative path for readability
                        try:
                            rel = os.path.relpath(dataset_jsonl, project_root)
                        except Exception:
                            rel = dataset_jsonl
                        print(f"✓ Using JSONL from input_dataset: {rel}")
        if not dataset_jsonl:
            print(
                "Error: Could not determine dataset. Provide --dataset-id or --dataset-jsonl, or ensure a JSONL-based data loader or input_dataset is used in your single discovered test."
            )
            return 1

        inferred_dataset_id = _build_trimmed_dataset_id(evaluator_id)
        if dry_run:
            print("--dry-run: would create dataset and upload JSONL")
            dataset_id = inferred_dataset_id
        else:
            try:
                # Resolve dataset_jsonl path relative to CWD if needed
                jsonl_path_for_upload = (
                    dataset_jsonl
                    if os.path.isabs(dataset_jsonl)
                    else os.path.abspath(os.path.join(project_root, dataset_jsonl))
                )
                dataset_id, _ = create_dataset_from_jsonl(
                    account_id=account_id,
                    api_key=api_key,
                    api_base=api_base,
                    dataset_id=inferred_dataset_id,
                    display_name=dataset_display_name or inferred_dataset_id,
                    jsonl_path=jsonl_path_for_upload,
                )
                print(f"✓ Created and uploaded dataset: {dataset_id}")
            except Exception as e:
                print(f"Error creating/uploading dataset: {e}")
                return 1

    # Build training config/body
    # Ensure base model is explicitly provided for clarity
    if not getattr(args, "base_model", None):
        print(
            "Error: --base-model is required. Please specify the base model resource id (e.g., accounts/{account}/models/<model_id>)."
        )
        return 1

    training_config: Dict[str, Any] = {"baseModel": args.base_model}
    if getattr(args, "warm_start_from", None):
        training_config["warmStartFrom"] = args.warm_start_from

    # Optional hyperparameters
    for key, arg_name in [
        ("epochs", "epochs"),
        ("batchSize", "batch_size"),
        ("learningRate", "learning_rate"),
        ("maxContextLength", "max_context_length"),
        ("loraRank", "lora_rank"),
        ("acceleratorCount", "accelerator_count"),
        ("region", "region"),
    ]:
        val = getattr(args, arg_name, None)
        if val is not None:
            training_config[key] = val

    inference_params: Dict[str, Any] = {}
    for key, arg_name in [
        ("temperature", "temperature"),
        ("topP", "top_p"),
        ("topK", "top_k"),
        ("maxTokens", "max_tokens"),
        ("n", "n"),
    ]:
        val = getattr(args, arg_name, None)
        if val is not None:
            inference_params[key] = val
    if getattr(args, "inference_extra_body", None):
        inference_params["extraBody"] = args.inference_extra_body

    wandb_config: Optional[Dict[str, Any]] = None
    if getattr(args, "wandb_enabled", False):
        wandb_config = {
            "enabled": True,
            "apiKey": getattr(args, "wandb_api_key", None),
            "project": getattr(args, "wandb_project", None),
            "entity": getattr(args, "wandb_entity", None),
            "runId": getattr(args, "wandb_run_id", None),
        }

    body: Dict[str, Any] = {
        # "displayName": getattr(args, "display_name", None) or f"{evaluator_id}-rft",
        "dataset": f"accounts/{account_id}/datasets/{dataset_id}",
        "evaluator": evaluator_resource_name,
        "evalAutoCarveout": bool(getattr(args, "eval_auto_carveout", True)),
        "trainingConfig": training_config,
        "inferenceParameters": inference_params or None,
        "wandbConfig": wandb_config,
        "chunkSize": getattr(args, "chunk_size", None),
        "outputStats": None,
        "outputMetrics": None,
        "mcpServer": None,
    }
    # Debug: print minimal summary
    print(f"Prepared RFT job for evaluator '{evaluator_id}' using dataset '{dataset_id}'")
    if getattr(args, "evaluation_dataset", None):
        body["evaluationDataset"] = args.evaluation_dataset
    if getattr(args, "output_model", None):
        body.setdefault("trainingConfig", {})["outputModel"] = f"accounts/{account_id}/models/{args.output_model}"
    else:
        # Auto-generate output model name if not provided
        auto_output_model = build_default_output_model(evaluator_id)
        body.setdefault("trainingConfig", {})["outputModel"] = f"accounts/{account_id}/models/{auto_output_model}"

    # Clean None fields to avoid noisy payloads
    body = {k: v for k, v in body.items() if v is not None}

    if dry_run:
        print("--dry-run: would create RFT job with body:")
        print(json.dumps(body, indent=2))
        _print_links(evaluator_id, dataset_id, None)
        return 0

    try:
        result = create_reinforcement_fine_tuning_job(
            account_id=account_id, api_key=api_key, api_base=api_base, body=body
        )
        job_name = result.get("name") if isinstance(result, dict) else None
        print("\n✅ Created Reinforcement Fine-tuning Job")
        if job_name:
            print(f"   name: {job_name}")
        _print_links(evaluator_id, dataset_id, job_name)
        return 0
    except Exception as e:
        print(f"Error creating RFT job: {e}")
        return 1
