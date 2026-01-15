import argparse
from fireworks._client import Fireworks
from fireworks.types.evaluation_job_create_response import EvaluationJobCreateResponse
import json
import os
import inspect
from typing import Any, Dict, Optional

from ..auth import get_fireworks_api_base, get_fireworks_api_key
from ..fireworks_client import create_fireworks_client
from .secrets import handle_secrets_upload
from .utils import (
    _build_trimmed_dataset_id,
    _build_evaluator_dashboard_url,
    _ensure_account_id,
    _extract_terminal_segment,
    resolve_evaluator,
    upload_and_ensure_evaluator,
    validate_evaluator_locally,
)
from .create_rft import (
    resolve_dataset,
    upload_dataset,
)


def _resolve_output_dataset(
    account_id: str,
    output_dataset_arg: Optional[str],
    evaluator_id: str,
) -> tuple[str, str]:
    """Resolve output dataset id and resource name.

    If not provided, auto-generates an output dataset ID based on the evaluator ID.
    """
    if output_dataset_arg:
        if output_dataset_arg.startswith("accounts/"):
            output_dataset_resource = output_dataset_arg
            output_dataset_id = _extract_terminal_segment(output_dataset_arg)
        else:
            output_dataset_id = output_dataset_arg
            output_dataset_resource = f"accounts/{account_id}/datasets/{output_dataset_id}"
    else:
        # Auto-generate output dataset ID
        output_dataset_id = _build_trimmed_dataset_id(f"{evaluator_id}-results")
        output_dataset_resource = f"accounts/{account_id}/datasets/{output_dataset_id}"
        print(f"Auto-generated output dataset ID: {output_dataset_id}")

    return output_dataset_id, output_dataset_resource


def _print_evj_links(
    evaluator_id: str, input_dataset_id: str, output_dataset_id: str, job_name: Optional[str]
) -> None:
    """Print helpful links to the Fireworks dashboard."""
    print("\n📊 Links:")
    print(f"   Evaluator: {_build_evaluator_dashboard_url(evaluator_id)}")
    if job_name:
        # Extract job id from resource name if present
        job_id = _extract_terminal_segment(job_name) if "/" in job_name else job_name
        print(f"   Evaluation Job: https://fireworks.ai/dashboard/evaluation-jobs/{job_id}")
    print(f"   Input Dataset: https://fireworks.ai/dashboard/datasets/{input_dataset_id}")
    print(f"   Output Dataset: https://fireworks.ai/dashboard/datasets/{output_dataset_id}")


def _create_evj_job(
    account_id: str,
    api_key: str,
    api_base: str,
    evaluator_id: str,
    evaluator_resource_name: str,
    input_dataset_id: str,
    input_dataset_resource: str,
    output_dataset_id: str,
    output_dataset_resource: str,
    args: argparse.Namespace,
    dry_run: bool,
) -> int:
    """Build and submit the Evaluation Job request (via Fireworks SDK)."""

    signature = inspect.signature(create_fireworks_client().evaluation_jobs.create)

    # Build top-level SDK kwargs
    sdk_kwargs: Dict[str, Any] = {}

    # Build the evaluation_job nested object
    evaluation_job: Dict[str, Any] = {
        "evaluator": evaluator_resource_name,
        "input_dataset": input_dataset_resource,
        "output_dataset": output_dataset_resource,
    }

    args_dict = vars(args)

    # Handle evaluation_job nested fields
    for k, v in args_dict.items():
        if v is None:
            continue
        if k.startswith("evaluation_job_") and k != "evaluation_job_id":
            field_name = k[len("evaluation_job_") :]
            # Don't overwrite the normalized resources
            if field_name in ("evaluator", "input_dataset", "output_dataset"):
                continue
            evaluation_job[field_name] = v

    sdk_kwargs["evaluation_job"] = evaluation_job

    # Handle top-level fields
    for name in signature.parameters:
        if name in ("account_id", "evaluation_job", "extra_headers", "extra_query", "extra_body", "timeout"):
            continue

        value = args_dict.get(name)
        if value is not None:
            sdk_kwargs[name] = value

    print(f"Prepared Evaluation Job for evaluator '{evaluator_id}' using dataset '{input_dataset_id}'")

    if dry_run:
        print("--dry-run: would call Fireworks().evaluation_jobs.create with kwargs:")
        print(json.dumps(sdk_kwargs, indent=2))
        _print_evj_links(evaluator_id, input_dataset_id, output_dataset_id, None)
        return 0

    try:
        fw: Fireworks = create_fireworks_client(api_key=api_key, base_url=api_base)
        job: EvaluationJobCreateResponse = fw.evaluation_jobs.create(account_id=account_id, **sdk_kwargs)
        job_name = job.name
        print(f"\n✅ Created Evaluation Job: {job_name}")
        _print_evj_links(evaluator_id, input_dataset_id, output_dataset_id, job_name)
        return 0
    except Exception as e:
        print(f"Error creating Evaluation Job: {e}")
        return 1


def create_evj_command(args) -> int:
    """Main entry point for the 'create evj' CLI command."""
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
    evaluator_arg: Optional[str] = getattr(args, "evaluation_job_evaluator", None)
    input_dataset_arg: Optional[str] = getattr(args, "evaluation_job_input_dataset", None)
    output_dataset_arg: Optional[str] = getattr(args, "evaluation_job_output_dataset", None)
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
    ) = resolve_evaluator(project_root, evaluator_arg, non_interactive, account_id, command_name="create evj")
    if not evaluator_id or not evaluator_resource_name:
        return 1

    # 1.5) Handle secrets upload (with double verification for existing secrets)
    env_file = getattr(args, "env_file", None)
    handle_secrets_upload(
        project_root=project_root,
        env_file=env_file,
        non_interactive=non_interactive,
    )

    # 2) Resolve input dataset source (id or JSONL path)
    input_dataset_id, input_dataset_resource, dataset_jsonl = resolve_dataset(
        project_root=project_root,
        account_id=account_id,
        dataset_id_arg=input_dataset_arg,
        dataset_jsonl_arg=None,  # EVJ doesn't support --dataset-jsonl flag yet
        selected_test_file_path=selected_test_file_path,
        selected_test_func_name=selected_test_func_name,
    )
    # Require either an existing dataset id or a JSONL source to materialize from
    if dataset_jsonl is None and not input_dataset_id:
        return 1

    # 3) Resolve output dataset
    output_dataset_id, output_dataset_resource = _resolve_output_dataset(account_id, output_dataset_arg, evaluator_id)

    # 4) Optional local validation
    if not skip_validation:
        if not validate_evaluator_locally(
            project_root=project_root,
            selected_test_file=selected_test_file_path,
            selected_test_func=selected_test_func_name,
            ignore_docker=ignore_docker,
            docker_build_extra=docker_build_extra,
            docker_run_extra=docker_run_extra,
        ):
            return 1

    # 5) Upload dataset when using JSONL sources (no-op for existing datasets)
    input_dataset_id, input_dataset_resource = upload_dataset(
        project_root=project_root,
        account_id=account_id,
        api_key=api_key,
        api_base=api_base,
        evaluator_id=evaluator_id,
        dataset_id=input_dataset_id,
        dataset_resource=input_dataset_resource,
        dataset_jsonl=dataset_jsonl,
        dataset_display_name=None,  # EVJ auto-generates display name
        dry_run=dry_run,
    )
    if not input_dataset_id or not input_dataset_resource:
        return 1

    # 6) Ensure evaluator exists and its latest version is ACTIVE (upload + poll if needed)
    if not dry_run:
        if not upload_and_ensure_evaluator(
            project_root=project_root,
            evaluator_id=evaluator_id,
            api_key=api_key,
            api_base=api_base,
            selected_test_file_path=selected_test_file_path,
            selected_test_func_name=selected_test_func_name,
        ):
            return 1

    # 7) Create the Evaluation Job
    return _create_evj_job(
        account_id=account_id,
        api_key=api_key,
        api_base=api_base,
        evaluator_id=evaluator_id,
        evaluator_resource_name=evaluator_resource_name,
        input_dataset_id=input_dataset_id,
        input_dataset_resource=input_dataset_resource,
        output_dataset_id=output_dataset_id,
        output_dataset_resource=output_dataset_resource,
        args=args,
        dry_run=dry_run,
    )
