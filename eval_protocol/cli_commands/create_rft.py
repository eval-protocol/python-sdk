import json
import os
import sys
from typing import Any, Dict, Optional

from ..auth import (
    get_fireworks_account_id,
    get_fireworks_api_base,
    get_fireworks_api_key,
    verify_api_key_and_get_account_id,
)
from ..fireworks_rft import (
    _map_api_host_to_app_host,
    build_default_dataset_id,
    build_default_output_model,
    create_dataset_from_jsonl,
    create_reinforcement_fine_tuning_job,
    detect_dataset_builder,
    load_evaluator_trace,
    materialize_dataset_via_builder,
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


def create_rft_command(args) -> int:
    evaluator_id: Optional[str] = getattr(args, "evaluator_id", None)
    non_interactive: bool = bool(getattr(args, "yes", False))
    dry_run: bool = bool(getattr(args, "dry_run", False))

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

    # Resolve evaluator resource name via local trace
    # trace = load_evaluator_trace(project_root, evaluator_id)
    # if not trace or not isinstance(trace, dict):
    #     print(
    #         "Error: Evaluator trace not found. Run 'eval-protocol upload' first or provide --dataset-id/--dataset-jsonl and --evaluator-id."
    #     )
    #     return 1
    # evaluator_resource_name = trace.get("evaluator_resource_name") or trace.get("name") or evaluator_id
    evaluator_resource_name = evaluator_id

    # Determine dataset id and materialization path
    dataset_id = getattr(args, "dataset_id", None)
    dataset_jsonl = getattr(args, "dataset_jsonl", None)
    dataset_display_name = getattr(args, "dataset_display_name", None)
    dataset_builder = getattr(args, "dataset_builder", None)

    if not dataset_id:
        # Try builder from args, else from trace detection
        # TODO: build dataset from traces directly
        # builder_spec = dataset_builder or trace.get("dataset_builder")
        # if not builder_spec:
        #     # Attempt detect from metric_dir
        #     metric_dir = trace.get("metric_dir")
        #     if metric_dir:
        #         builder_spec = detect_dataset_builder(metric_dir)
        # if not builder_spec:
        #     print(
        #         "Error: Could not determine dataset. Provide --dataset-id, --dataset-jsonl, or --dataset-builder."
        #     )
        #     return 1
        # try:
        #     dataset_jsonl, count = materialize_dataset_via_builder(builder_spec)
        #     print(f"✓ Materialized dataset via builder ({builder_spec}): {count} rows → {dataset_jsonl}")
        # except Exception as e:
        #     print(f"Error: dataset builder failed: {e}")
        #     return 1

        if not dataset_jsonl:
            print("Error: Could not determine dataset. Provide --dataset-id or --dataset-jsonl.")
            return 1

        inferred_dataset_id = build_default_dataset_id(evaluator_id)
        if dry_run:
            print("--dry-run: would create dataset and upload JSONL")
            dataset_id = inferred_dataset_id
        else:
            try:
                dataset_id, _ = create_dataset_from_jsonl(
                    account_id=account_id,
                    api_key=api_key,
                    api_base=api_base,
                    dataset_id=inferred_dataset_id,
                    display_name=dataset_display_name or inferred_dataset_id,
                    jsonl_path=dataset_jsonl,
                )
                print(f"✓ Created and uploaded dataset: {dataset_id}")
            except Exception as e:
                print(f"Error creating/uploading dataset: {e}")
                return 1

    # Build training config/body
    training_config: Dict[str, Any] = {}
    if getattr(args, "base_model", None):
        training_config["baseModel"] = args.base_model
    if getattr(args, "warm_start_from", None):
        training_config["warmStartFrom"] = args.warm_start_from
    if "baseModel" not in training_config and "warmStartFrom" not in training_config:
        # Provide a conservative default if neither is set
        training_config["baseModel"] = "accounts/fireworks/models/llama-v3p1-8b-instruct"

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
        "outputStats": None,
        "outputMetrics": None,
        "mcpServer": None,
    }
    print("Show body:")
    print(json.dumps(body, indent=2))
    if getattr(args, "evaluation_dataset", None):
        body["evaluationDataset"] = args.evaluation_dataset
    if getattr(args, "output_model", None):
        body.setdefault("trainingConfig", {})["outputModel"] = f"accounts/{account_id}/models/{args.output_model}"
    else:
        body.setdefault("trainingConfig", {})["outputModel"] = build_default_output_model(evaluator_id)

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
