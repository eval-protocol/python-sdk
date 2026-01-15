import argparse
from fireworks._client import Fireworks
from fireworks.types.reinforcement_fine_tuning_job import ReinforcementFineTuningJob
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Optional
import inspect
import tempfile
from pydantic import ValidationError

from ..auth import get_fireworks_api_base, get_fireworks_api_key
from ..fireworks_client import create_fireworks_client
from ..common_utils import load_jsonl
from ..fireworks_rft import (
    create_dataset_from_jsonl,
    detect_dataset_builder,
    materialize_dataset_via_builder,
)
from ..models import EvaluationRow
from .utils import (
    _build_entry_point,
    _build_trimmed_dataset_id,
    _build_evaluator_dashboard_url,
    _discover_and_select_tests,
    _discover_tests,
    _ensure_account_id,
    _extract_terminal_segment,
    _normalize_evaluator_id,
    _poll_evaluator_version_status,
    _print_links,
    _resolve_selected_test,
    load_module_from_file_path,
    resolve_evaluator,
    upload_and_ensure_evaluator,
    validate_evaluator_locally,
)
from .local_test import run_evaluator_test


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


def resolve_dataset(
    project_root: str,
    account_id: str,
    dataset_id_arg: Optional[str],
    dataset_jsonl_arg: Optional[str],
    selected_test_file_path: Optional[str],
    selected_test_func_name: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve dataset source without performing any uploads.

    Args:
        project_root: Path to the project root directory.
        account_id: The Fireworks account ID.
        dataset_id_arg: Dataset ID or fully-qualified resource name (from --dataset).
        dataset_jsonl_arg: Path to a local JSONL file (from --dataset-jsonl).
        selected_test_file_path: Path to the selected test file (for inference).
        selected_test_func_name: Name of the selected test function (for inference).

    Returns a tuple of:
      - dataset_id: existing dataset id when using --dataset or fully-qualified dataset resource
      - dataset_resource: fully-qualified dataset resource for existing datasets; None for JSONL sources
      - dataset_jsonl: local JSONL path when using --dataset-jsonl or inferred sources; None for id-only datasets
    """
    dataset_id = dataset_id_arg
    dataset_jsonl = dataset_jsonl_arg
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


def upload_dataset(
    project_root: str,
    account_id: str,
    api_key: str,
    api_base: str,
    evaluator_id: str,
    dataset_id: Optional[str],
    dataset_resource: Optional[str],
    dataset_jsonl: Optional[str],
    dataset_display_name: Optional[str],
    dry_run: bool,
) -> tuple[Optional[str], Optional[str]]:
    """Create/upload the dataset when using a local JSONL source.

    For existing datasets (--dataset or fully-qualified ids), this is a no-op that
    simply ensures dataset_id and dataset_resource are populated.

    Args:
        project_root: Path to the project root directory.
        account_id: The Fireworks account ID.
        api_key: Fireworks API key.
        api_base: Fireworks API base URL.
        evaluator_id: The evaluator ID (used for generating dataset ID if needed).
        dataset_id: Existing dataset ID (if known).
        dataset_resource: Existing dataset resource name (if known).
        dataset_jsonl: Path to local JSONL file to upload (if any).
        dataset_display_name: Display name for the dataset (optional).
        dry_run: If True, simulate the upload without actually creating.

    Returns:
        A tuple of (dataset_id, dataset_resource).
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
    display_name = dataset_display_name or inferred_dataset_id

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
            display_name=display_name,
            jsonl_path=jsonl_path_for_upload,
        )
        print(f"✓ Created and uploaded dataset: {dataset_id}")
        dataset_resource = f"accounts/{account_id}/datasets/{dataset_id}"
        return dataset_id, dataset_resource
    except Exception as e:
        print(f"Error creating/uploading dataset: {e}")
        return None, None


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

    signature = inspect.signature(create_fireworks_client().reinforcement_fine_tuning_jobs.create)

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
        fw: Fireworks = create_fireworks_client(api_key=api_key, base_url=api_base)
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
    ) = resolve_evaluator(project_root, evaluator_arg, non_interactive, account_id, command_name="create rft")
    if not evaluator_id or not evaluator_resource_name:
        return 1

    # 2) Resolve dataset source (id or JSONL path)
    dataset_id, dataset_resource, dataset_jsonl = resolve_dataset(
        project_root=project_root,
        account_id=account_id,
        dataset_id_arg=getattr(args, "dataset", None),
        dataset_jsonl_arg=getattr(args, "dataset_jsonl", None),
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
        if not validate_evaluator_locally(
            project_root=project_root,
            selected_test_file=selected_test_file_path,
            selected_test_func=selected_test_func_name,
            ignore_docker=ignore_docker,
            docker_build_extra=docker_build_extra,
            docker_run_extra=docker_run_extra,
        ):
            return 1

    # 4) Upload dataset when using JSONL sources (no-op for existing datasets)
    dataset_id, dataset_resource = upload_dataset(
        project_root=project_root,
        account_id=account_id,
        api_key=api_key,
        api_base=api_base,
        evaluator_id=evaluator_id,
        dataset_id=dataset_id,
        dataset_resource=dataset_resource,
        dataset_jsonl=dataset_jsonl,
        dataset_display_name=getattr(args, "dataset_display_name", None),
        dry_run=dry_run,
    )
    if not dataset_id or not dataset_resource:
        return 1

    # 5) Ensure evaluator exists and its latest version is ACTIVE (upload + poll if needed)
    if not upload_and_ensure_evaluator(
        project_root=project_root,
        evaluator_id=evaluator_id,
        api_key=api_key,
        api_base=api_base,
        selected_test_file_path=selected_test_file_path,
        selected_test_func_name=selected_test_func_name,
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
