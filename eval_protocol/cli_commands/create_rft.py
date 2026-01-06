import argparse
from fireworks._client import Fireworks
from fireworks.types.reinforcement_fine_tuning_job import ReinforcementFineTuningJob
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Optional
import inspect
import requests
import tempfile
from pydantic import ValidationError

from ..auth import get_fireworks_api_base, get_fireworks_api_key
from ..common_utils import get_user_agent, load_jsonl
from ..fireworks_rft import (
    create_dataset_from_jsonl,
    detect_dataset_builder,
    materialize_dataset_via_builder,
)
from ..models import EvaluationRow
from .upload import upload_command
from .utils import (
    _build_entry_point,
    _build_trimmed_dataset_id,
    _build_evaluator_dashboard_url,
    _discover_and_select_tests,
    _discover_tests,
    _ensure_account_id,
    _extract_terminal_segment,
    _normalize_evaluator_id,
    _print_links,
    _resolve_selected_test,
    load_module_from_file_path,
)
from .local_test import run_evaluator_test

from fireworks import Fireworks


def _extract_dataset_adapter(
    test_file_path: str, test_func_name: str
) -> Optional[Callable[[list[dict[str, Any]]], Any]]:
    """Extract dataset_adapter from an @evaluation_test wrapper via __ep_params__."""
    try:
        module = load_module_from_file_path(test_file_path)
        wrapper = getattr(module, test_func_name, None)
        if wrapper is None:
            return None
        ep_params = getattr(wrapper, "__ep_params__", None)
        if ep_params is None:
            return None
        adapter = getattr(ep_params, "dataset_adapter", None)
        if callable(adapter):
            return adapter
        return None
    except Exception:
        return None


def _maybe_transform_dataset_jsonl_via_adapter(
    project_root: str,
    dataset_jsonl: str,
    test_file_path: Optional[str],
    test_func_name: Optional[str],
) -> str:
    """Transform dataset_jsonl via the test's dataset_adapter (when available).

    For RFT dataset uploads, we want the uploaded dataset to match what evaluation-time
    would run on. If the selected evaluation test provides a dataset_adapter, that
    adapter is treated as the source of truth for constructing EvaluationRows.
    """
    if not dataset_jsonl:
        return dataset_jsonl

    if not test_file_path or not test_func_name:
        return dataset_jsonl

    adapter = _extract_dataset_adapter(test_file_path, test_func_name)
    if not adapter:
        return dataset_jsonl

    raw_rows: list[dict[str, Any]] = load_jsonl(dataset_jsonl)  # type: ignore[assignment]
    adapted = adapter(raw_rows)
    if not isinstance(adapted, list):
        raise ValueError("dataset_adapter must return a list of EvaluationRow (or dicts parseable as EvaluationRow).")

    eval_rows: list[EvaluationRow] = []
    for item in adapted:
        if isinstance(item, EvaluationRow):
            eval_rows.append(item)
        else:
            eval_rows.append(EvaluationRow.model_validate(item))

    output_dir = os.path.join(project_root, ".ep_tmp")
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".jsonl",
        prefix="ep_rft_dataset_",
        dir=output_dir,
    ) as f:
        for row in eval_rows:
            f.write(json.dumps(row.model_dump(mode="json", exclude_none=True), ensure_ascii=False) + "\n")
        out_path = os.path.abspath(f.name)
    try:
        rel = os.path.relpath(out_path, project_root)
    except Exception:
        rel = out_path
    print(f"✓ Transformed dataset via dataset_adapter into EvaluationRow JSONL: {rel} ({len(eval_rows)} rows)")
    return out_path


def _extract_jsonl_from_dataloader(test_file_path: str, test_func_name: str) -> Optional[str]:
    """Import the test module and extract a JSONL path from data_loaders param if present.

    Looks for a pytest.mark.parametrize with argnames containing 'data_loaders' and attempts to
    find an object with attribute 'jsonl_path'. If a relative path is found, it is resolved
    relative to the directory of the test file.
    """
    try:
        module = load_module_from_file_path(test_file_path)
        wrapper = getattr(module, test_func_name, None)
        if wrapper is None:
            return None
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
        module = load_module_from_file_path(test_file_path)
        wrapper = getattr(module, test_func_name, None)
        if wrapper is None:
            return None
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
                            candidate_paths = []
                            if os.path.isabs(dataset_path):
                                candidate_paths.append(dataset_path)
                            else:
                                base_dir = os.path.dirname(os.path.abspath(test_file_path))
                                candidate_paths.append(os.path.abspath(os.path.join(base_dir, dataset_path)))
                                # Also try resolving from current working directory
                                candidate_paths.append(os.path.abspath(os.path.join(os.getcwd(), dataset_path)))

                            for candidate in candidate_paths:
                                if os.path.isfile(candidate):
                                    return candidate
        return None
    except Exception:
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


def _validate_dataset_jsonl(jsonl_path: str, sample_limit: int = 50) -> bool:
    """Validate that a JSONL file contains rows compatible with EvaluationRow.

    We stream up to `sample_limit` rows, ensuring each is JSON-decodable and can be
    parsed by the EvaluationRow model. Returns True on success, False on any error.
    """
    try:
        if not os.path.isfile(jsonl_path):
            print(f"Error: dataset JSONL not found at path: {jsonl_path}")
            return False

        row_count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error: dataset JSONL contains invalid JSON (line {row_count + 1}): {e}")
                    return False

                try:
                    EvaluationRow.model_validate(data)
                except ValidationError as e:
                    print(f"Error: dataset JSONL row {row_count + 1} is not a valid EvaluationRow: {e}")
                    return False

                row_count += 1
                if row_count >= sample_limit:
                    break

        if row_count == 0:
            print(f"Error: dataset JSONL at {jsonl_path} appears to be empty.")
            return False

        return True
    except Exception as e:
        print(f"Error validating dataset JSONL at {jsonl_path}: {e}")
        return False


def _validate_dataset(dataset_jsonl: Optional[str]) -> bool:
    """Validate dataset JSONL path when available; no-op when using dataset IDs only."""
    if not dataset_jsonl:
        return True
    return _validate_dataset_jsonl(dataset_jsonl)


def _validate_evaluator_locally(
    project_root: str,
    selected_test_file: Optional[str],
    selected_test_func: Optional[str],
    ignore_docker: bool,
    docker_build_extra: str,
    docker_run_extra: str,
) -> bool:
    """Run pytest locally for the selected evaluation test to validate the evaluator.

    The pytest helpers always enforce a small success threshold (0.01) for
    evaluation_test-based suites so that an evaluation run where all scores are
    0.0 will naturally fail with a non-zero pytest exit code, which we then treat
    as a failed validator.
    """
    if not selected_test_file or not selected_test_func:
        # No local test associated; skip validation but warn the user.
        print("Warning: Could not resolve a local evaluation test for this evaluator; skipping local validation.")
        return True

    pytest_target = _build_entry_point(project_root, selected_test_file, selected_test_func)
    exit_code = run_evaluator_test(
        project_root=project_root,
        pytest_target=pytest_target,
        ignore_docker=ignore_docker,
        docker_build_extra=docker_build_extra,
        docker_run_extra=docker_run_extra,
    )
    return exit_code == 0


def _resolve_evaluator(
    project_root: str,
    evaluator_arg: Optional[str],
    non_interactive: bool,
    account_id: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolve evaluator id/resource and associated local test (file + func)."""
    evaluator_id = evaluator_arg
    selected_test_file_path: Optional[str] = None
    selected_test_func_name: Optional[str] = None

    if not evaluator_id:
        selected_tests = _discover_and_select_tests(project_root, non_interactive=non_interactive)
        if not selected_tests:
            return None, None, None, None

        if len(selected_tests) != 1:
            if non_interactive and len(selected_tests) > 1:
                print("Error: Multiple evaluation tests found in --yes (non-interactive) mode.")
                print("       Please pass --evaluator or --entry to disambiguate.")
            else:
                print("Error: Please select exactly one evaluation test for 'create rft'.")
            return None, None, None, None

        chosen = selected_tests[0]
        func_name = chosen.qualname.split(".")[-1]
        source_file_name = os.path.splitext(os.path.basename(chosen.file_path))[0]
        evaluator_id = _normalize_evaluator_id(f"{source_file_name}-{func_name}")
        # Resolve selected test once for downstream
        selected_test_file_path, selected_test_func_name = _resolve_selected_test(
            project_root, evaluator_id, selected_tests=selected_tests
        )
    else:
        # Caller provided an evaluator id or fully-qualified resource; try to resolve local test
        short_id = evaluator_id
        if evaluator_id.startswith("accounts/"):
            short_id = _extract_terminal_segment(evaluator_id)
        st_path, st_func = _resolve_selected_test(project_root, short_id)
        if st_path and st_func:
            selected_test_file_path = st_path
            selected_test_func_name = st_func
        evaluator_id = short_id

    if not evaluator_id:
        return None, None, None, None

    # Resolve evaluator resource name to fully-qualified format required by API.
    if evaluator_arg and evaluator_arg.startswith("accounts/"):
        evaluator_resource_name = evaluator_arg
    else:
        evaluator_resource_name = f"accounts/{account_id}/evaluators/{evaluator_id}"

    return evaluator_id, evaluator_resource_name, selected_test_file_path, selected_test_func_name


def _resolve_dataset(
    project_root: str,
    account_id: str,
    args: argparse.Namespace,
    selected_test_file_path: Optional[str],
    selected_test_func_name: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve dataset source without performing any uploads.

    Returns a tuple of:
      - dataset_id: existing dataset id when using --dataset or fully-qualified dataset resource
      - dataset_resource: fully-qualified dataset resource for existing datasets; None for JSONL sources
      - dataset_jsonl: local JSONL path when using --dataset-jsonl or inferred sources; None for id-only datasets
    """
    dataset_id = getattr(args, "dataset", None)
    dataset_jsonl = getattr(args, "dataset_jsonl", None)
    dataset_resource_override: Optional[str] = None

    if dataset_id and dataset_jsonl:
        print(
            "Error: --dataset and --dataset-jsonl cannot be used together.\n"
            "       Use --dataset to reference an existing dataset, or --dataset-jsonl to create a new one from JSONL."
        )
        return None, None, None

    if isinstance(dataset_id, str) and dataset_id.startswith("accounts/"):
        # Caller passed a fully-qualified dataset; capture it for body and keep only terminal id for printing
        dataset_resource_override = dataset_id
        dataset_id = _extract_terminal_segment(dataset_id)

    if not dataset_id:
        # Prefer explicit --dataset-jsonl, else attempt to extract from the selected test's data loader or input_dataset.
        if not dataset_jsonl:
            # Use specifically selected test if available; else only infer when exactly one test exists
            test_file_for_infer = None
            func_for_infer = None
            if selected_test_file_path and selected_test_func_name:
                test_file_for_infer = selected_test_file_path
                func_for_infer = selected_test_func_name
            else:
                tests = _discover_tests(project_root)
                if len(tests) == 1:
                    test_file_for_infer = tests[0].file_path
                    func_for_infer = tests[0].qualname.split(".")[-1]
            if test_file_for_infer and func_for_infer:
                # Block using data loaders as a dataset source
                dataset_jsonl = _extract_jsonl_from_dataloader(test_file_for_infer, func_for_infer)
                if dataset_jsonl:
                    print(
                        "Error: Evaluation tests that use 'data_loaders' to provide a dataset JSONL are not supported for 'create rft'.\n"
                        "       Please switch to a JSONL-based dataset via input_dataset arg in @evaluation_test decorator."
                    )
                    return None, None, None
                dataset_jsonl = _extract_jsonl_from_input_dataset(test_file_for_infer, func_for_infer)
                if dataset_jsonl:
                    try:
                        rel = os.path.relpath(dataset_jsonl, project_root)
                    except Exception:
                        rel = dataset_jsonl
                    print(f"✓ Using JSONL from input_dataset: {rel}")
                if not dataset_jsonl:
                    # Last resort: attempt to detect and run a dataset builder in the test's directory
                    metric_dir = os.path.dirname(test_file_for_infer)
                    builder_spec = detect_dataset_builder(metric_dir)
                    if builder_spec:
                        try:
                            tmp_jsonl, count = materialize_dataset_via_builder(builder_spec)
                            dataset_jsonl = tmp_jsonl
                            print(f"✓ Materialized {count} rows via dataset builder: {builder_spec}")
                        except Exception as e:
                            print(f"Warning: dataset builder failed: {e}")
        if not dataset_jsonl:
            print(
                "Error: Could not determine dataset. Provide --dataset or --dataset-jsonl, or ensure a JSONL-based data loader or input_dataset is used in your single discovered test."
            )
            return None, None, None

    # Build dataset resource for existing datasets; JSONL-based datasets will be uploaded later.
    dataset_resource = None
    if dataset_id:
        dataset_resource = dataset_resource_override or f"accounts/{account_id}/datasets/{dataset_id}"

    return dataset_id, dataset_resource, dataset_jsonl


def _upload_dataset(
    project_root: str,
    account_id: str,
    api_key: str,
    api_base: str,
    evaluator_id: str,
    dataset_id: Optional[str],
    dataset_resource: Optional[str],
    dataset_jsonl: Optional[str],
    args: argparse.Namespace,
    dry_run: bool,
) -> tuple[Optional[str], Optional[str]]:
    """Create/upload the dataset when using a local JSONL source.

    For existing datasets (--dataset or fully-qualified ids), this is a no-op that
    simply ensures dataset_id and dataset_resource are populated.
    """
    # Existing dataset case: nothing to upload
    if not dataset_jsonl:
        if not dataset_id:
            return None, None
        if not dataset_resource:
            dataset_resource = f"accounts/{account_id}/datasets/{dataset_id}"
        return dataset_id, dataset_resource

    # JSONL-based dataset: upload or simulate upload
    inferred_dataset_id = _build_trimmed_dataset_id(evaluator_id)
    dataset_display_name = getattr(args, "dataset_display_name", None) or inferred_dataset_id

    # Resolve dataset_jsonl path relative to CWD if needed
    jsonl_path_for_upload = (
        dataset_jsonl if os.path.isabs(dataset_jsonl) else os.path.abspath(os.path.join(project_root, dataset_jsonl))
    )

    if dry_run:
        print("--dry-run: would create dataset and upload JSONL")
        dataset_id = inferred_dataset_id
        dataset_resource = f"accounts/{account_id}/datasets/{dataset_id}"
        return dataset_id, dataset_resource

    try:
        dataset_id, _ = create_dataset_from_jsonl(
            account_id=account_id,
            api_key=api_key,
            api_base=api_base,
            dataset_id=inferred_dataset_id,
            display_name=dataset_display_name,
            jsonl_path=jsonl_path_for_upload,
        )
        print(f"✓ Created and uploaded dataset: {dataset_id}")
        dataset_resource = f"accounts/{account_id}/datasets/{dataset_id}"
        return dataset_id, dataset_resource
    except Exception as e:
        print(f"Error creating/uploading dataset: {e}")
        return None, None


def _upload_and_ensure_evaluator(
    project_root: str,
    evaluator_id: str,
    evaluator_resource_name: str,
    api_key: str,
    api_base: str,
    force: bool,
) -> bool:
    """Ensure the evaluator exists and is ACTIVE, uploading it if needed."""
    # Optional short-circuit: if evaluator already exists and not forcing, skip upload path
    if not force:
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": get_user_agent(),
            }
            resp = requests.get(f"{api_base}/v1/{evaluator_resource_name}", headers=headers, timeout=10)
            if resp.ok:
                state = resp.json().get("state", "STATE_UNSPECIFIED")
                print(f"✓ Evaluator exists (state: {state}). Skipping upload (use --force to overwrite).")
                # Poll for ACTIVE before proceeding
                print(f"Waiting for evaluator '{evaluator_id}' to become ACTIVE...")
                if not _poll_evaluator_status(
                    evaluator_resource_name=evaluator_resource_name,
                    api_key=api_key,
                    api_base=api_base,
                    timeout_minutes=10,
                ):
                    dashboard_url = _build_evaluator_dashboard_url(evaluator_id)
                    print("\n❌ Evaluator is not ready within the timeout period.")
                    print(f"📊 Please check the evaluator status at: {dashboard_url}")
                    print("   Wait for it to become ACTIVE, then run 'eval-protocol create rft' again.")
                    return False
                return True
        except requests.exceptions.RequestException:
            pass

    # Ensure evaluator exists by invoking the upload flow programmatically
    try:
        tests = _discover_tests(project_root)
        selected_entry: Optional[str] = None
        st_path, st_func = _resolve_selected_test(project_root, evaluator_id, selected_tests=tests)
        if st_path and st_func:
            selected_entry = _build_entry_point(project_root, st_path, st_func)
        # If still unresolved and multiple tests exist, fail fast to avoid uploading unintended evaluators
        if selected_entry is None and len(tests) > 1:
            print(
                f"Error: Multiple evaluation tests found, and the selected evaluator {evaluator_id} does not match any discovered test.\n"
                "       Please re-run specifying the evaluator.\n"
                "       Hints:\n"
                "         - eval-protocol create rft --evaluator <existing-evaluator-id>\n"
            )
            return False

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
                evaluator_resource_name=evaluator_resource_name,
                api_key=api_key,
                api_base=api_base,
                timeout_minutes=10,
            )

            if not is_active:
                dashboard_url = _build_evaluator_dashboard_url(evaluator_id)
                print("\n❌ Evaluator is not ready within the timeout period.")
                print(f"📊 Please check the evaluator status at: {dashboard_url}")
                print("   Wait for it to become ACTIVE, then run 'eval-protocol create rft' again.")
                return False
            return True
        else:
            print("Warning: Evaluator upload did not complete successfully; proceeding to RFT creation.")
            return False
    except Exception as e:
        print(f"Warning: Failed to upload evaluator automatically: {e}")
        return False


def _create_rft_job(
    account_id: str,
    api_key: str,
    api_base: str,
    evaluator_id: str,
    evaluator_resource_name: str,
    dataset_id: str,
    dataset_resource: str,
    args: argparse.Namespace,
    dry_run: bool,
) -> int:
    """Build and submit the RFT job request (via Fireworks SDK)."""

    signature = inspect.signature(Fireworks().reinforcement_fine_tuning_jobs.create)

    # Build top-level SDK kwargs
    sdk_kwargs: Dict[str, Any] = {
        "evaluator": evaluator_resource_name,
        "dataset": dataset_resource,
    }

    args_dict = vars(args)
    for name in signature.parameters:
        # Do NOT let raw CLI args overwrite the normalized resources passed into this function.
        if name in ("dataset", "evaluator"):
            continue
        prefix = name + "_"

        # Collect "flattened" argparse fields back into the nested dict expected by the SDK.
        # Example: training_config_epochs=3 becomes sdk_kwargs["training_config"]["epochs"] = 3.
        nested = {}
        for k, v in args_dict.items():
            if v is None:
                continue
            if not k.startswith(prefix):
                continue
            nested[k[len(prefix) :]] = v

        if nested:
            sdk_kwargs[name] = nested
        elif args_dict.get(name) is not None:
            sdk_kwargs[name] = args_dict[name]

    print(f"Prepared RFT job for evaluator '{evaluator_id}' using dataset '{dataset_id}'")

    if dry_run:
        print("--dry-run: would call Fireworks().reinforcement_fine_tuning_jobs.create with kwargs:")
        print(json.dumps(sdk_kwargs, indent=2))
        _print_links(evaluator_id, dataset_id, None)
        return 0

    try:
        fw: Fireworks = Fireworks(api_key=api_key, base_url=api_base)
        job: ReinforcementFineTuningJob = fw.reinforcement_fine_tuning_jobs.create(account_id=account_id, **sdk_kwargs)
        job_name = job.name
        print(f"\n✅ Created Reinforcement Fine-tuning Job: {job_name}")
        _print_links(evaluator_id, dataset_id, job_name)
        return 0
    except Exception as e:
        print(f"Error creating RFT job: {e}")
        return 1


def create_rft_command(args) -> int:
    # Pre-flight: resolve auth and environment
    api_key = get_fireworks_api_key()
    if not api_key:
        print("Error: FIREWORKS_API_KEY not set.")
        return 1

    account_id = _ensure_account_id()
    if not account_id:
        print("Error: Could not resolve Fireworks account id from FIREWORKS_API_KEY.")
        return 1

    api_base = get_fireworks_api_base()
    project_root = os.getcwd()
    evaluator_arg: Optional[str] = getattr(args, "evaluator", None)
    non_interactive: bool = bool(getattr(args, "yes", False))
    dry_run: bool = bool(getattr(args, "dry_run", False))
    force: bool = bool(getattr(args, "force", False))
    skip_validation: bool = bool(getattr(args, "skip_validation", False))
    ignore_docker: bool = bool(getattr(args, "ignore_docker", False))
    docker_build_extra: str = getattr(args, "docker_build_extra", "") or ""
    docker_run_extra: str = getattr(args, "docker_run_extra", "") or ""

    # 1) Resolve evaluator and associated local test
    (
        evaluator_id,
        evaluator_resource_name,
        selected_test_file_path,
        selected_test_func_name,
    ) = _resolve_evaluator(project_root, evaluator_arg, non_interactive, account_id)
    if not evaluator_id or not evaluator_resource_name:
        return 1

    # 2) Resolve dataset source (id or JSONL path)
    dataset_id, dataset_resource, dataset_jsonl = _resolve_dataset(
        project_root=project_root,
        account_id=account_id,
        args=args,
        selected_test_file_path=selected_test_file_path,
        selected_test_func_name=selected_test_func_name,
    )
    # Require either an existing dataset id or a JSONL source to materialize from
    if dataset_jsonl is None and not dataset_id:
        return 1

    # 2.5) If the selected evaluation test provides a dataset_adapter, always use it to
    # construct the EvaluationRow dataset that we upload for RFT.
    if dataset_jsonl is not None:
        dataset_jsonl = _maybe_transform_dataset_jsonl_via_adapter(
            project_root=project_root,
            dataset_jsonl=dataset_jsonl,
            test_file_path=selected_test_file_path,
            test_func_name=selected_test_func_name,
        )

    # 3) Optional local validation
    if not skip_validation:
        # Dataset validation (JSONL must be EvaluationRow-compatible when present)
        if not _validate_dataset(dataset_jsonl):
            return 1

        # Evaluator validation (run pytest for the selected test, possibly via Docker)
        if not _validate_evaluator_locally(
            project_root=project_root,
            selected_test_file=selected_test_file_path,
            selected_test_func=selected_test_func_name,
            ignore_docker=ignore_docker,
            docker_build_extra=docker_build_extra,
            docker_run_extra=docker_run_extra,
        ):
            return 1

    # 4) Upload dataset when using JSONL sources (no-op for existing datasets)
    dataset_id, dataset_resource = _upload_dataset(
        project_root=project_root,
        account_id=account_id,
        api_key=api_key,
        api_base=api_base,
        evaluator_id=evaluator_id,
        dataset_id=dataset_id,
        dataset_resource=dataset_resource,
        dataset_jsonl=dataset_jsonl,
        args=args,
        dry_run=dry_run,
    )
    if not dataset_id or not dataset_resource:
        return 1

    # 5) Ensure evaluator exists and is ACTIVE (upload + poll if needed)
    if not _upload_and_ensure_evaluator(
        project_root=project_root,
        evaluator_id=evaluator_id,
        evaluator_resource_name=evaluator_resource_name,
        api_key=api_key,
        api_base=api_base,
        force=force,
    ):
        return 1

    # 6) Create the RFT job
    return _create_rft_job(
        account_id=account_id,
        api_key=api_key,
        api_base=api_base,
        evaluator_id=evaluator_id,
        evaluator_resource_name=evaluator_resource_name,
        dataset_id=dataset_id,
        dataset_resource=dataset_resource,
        args=args,
        dry_run=dry_run,
    )
