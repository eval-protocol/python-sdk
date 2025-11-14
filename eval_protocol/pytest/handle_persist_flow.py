from collections import defaultdict
import configparser
import json
import os
from pathlib import Path
import pathlib
import re
from typing import Any

from eval_protocol.common_utils import get_user_agent
from eval_protocol.directory_utils import find_eval_protocol_dir
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.store_experiment_link import store_experiment_link
from eval_protocol.auth import (
    get_fireworks_api_key,
    get_fireworks_account_id,
    verify_api_key_and_get_account_id,
    get_fireworks_api_base,
)

import requests


def handle_persist_flow(all_results: list[list[EvaluationRow]], test_func_name: str):
    try:
        # Default is to save and upload experiment JSONL files, unless explicitly disabled
        custom_output_dir = os.getenv("EP_OUTPUT_DIR")
        should_save = os.getenv("EP_NO_UPLOAD") != "1" or custom_output_dir is not None

        if should_save:
            current_run_rows = [item for sublist in all_results for item in sublist]
            if current_run_rows:
                experiments: dict[str, list[EvaluationRow]] = defaultdict(list)
                for row in current_run_rows:
                    if row.execution_metadata and row.execution_metadata.experiment_id:
                        experiments[row.execution_metadata.experiment_id].append(row)

                eval_protocol_dir = find_eval_protocol_dir()
                if custom_output_dir:
                    eval_protocol_dir = custom_output_dir
                exp_dir = pathlib.Path(eval_protocol_dir) / "experiment_results"
                exp_dir.mkdir(parents=True, exist_ok=True)

                # Create one JSONL file per experiment_id
                for experiment_id, exp_rows in experiments.items():
                    if not experiment_id or not exp_rows:
                        continue

                    # Generate dataset name (sanitize for Fireworks API compatibility)
                    # API requires: lowercase a-z, 0-9, and hyphen (-) only
                    safe_experiment_id = re.sub(r"[^a-zA-Z0-9-]", "-", experiment_id).lower()
                    safe_test_func_name = re.sub(r"[^a-zA-Z0-9-]", "-", test_func_name).lower()
                    dataset_name = f"{safe_test_func_name}-{safe_experiment_id}"

                    if len(dataset_name) > 63:
                        dataset_name = dataset_name[:63]

                    # Fireworks requires: last character of id must not be '-'
                    dataset_name = dataset_name.rstrip("-")

                    # Ensure non-empty after stripping; fallback to safe_test_func_name
                    if not dataset_name:
                        dataset_name = safe_test_func_name[:63].rstrip("-") or "dataset"

                    exp_file = exp_dir / f"{experiment_id}.jsonl"
                    with open(exp_file, "w", encoding="utf-8") as f:
                        for row in exp_rows:
                            row_data = row.model_dump(exclude_none=True, mode="json")

                            if row.evaluation_result:
                                row_data["evals"] = {"score": row.evaluation_result.score}

                                row_data["eval_details"] = {
                                    "score": row.evaluation_result.score,
                                    "is_score_valid": row.evaluation_result.is_score_valid,
                                    "reason": row.evaluation_result.reason or "",
                                    "metrics": {
                                        name: metric.model_dump() if metric else {}
                                        for name, metric in (row.evaluation_result.metrics or {}).items()
                                    },
                                }
                            else:
                                # Default values if no evaluation result
                                row_data["evals"] = {"score": 0}
                                row_data["eval_details"] = {
                                    "score": 0,
                                    "is_score_valid": False,
                                    "reason": "No evaluation result",
                                    "metrics": {},
                                }

                            json.dump(row_data, f, ensure_ascii=False)
                            f.write("\n")

                    should_upload = os.getenv("EP_NO_UPLOAD") != "1"
                    if not should_upload:
                        continue

                    # Resolve credentials using centralized auth helpers with verification fallback
                    fireworks_api_key = get_fireworks_api_key()
                    fireworks_account_id = get_fireworks_account_id()
                    if not fireworks_account_id and fireworks_api_key:
                        try:
                            fireworks_account_id = verify_api_key_and_get_account_id(
                                api_key=fireworks_api_key, api_base=get_fireworks_api_base()
                            )
                        except Exception:
                            fireworks_account_id = None

                    if not fireworks_api_key and not fireworks_account_id:
                        store_experiment_link(
                            experiment_id,
                            "No Fireworks API key AND account ID found",
                            "failure",
                        )
                        continue
                    elif not fireworks_api_key:
                        store_experiment_link(
                            experiment_id,
                            "No Fireworks API key found",
                            "failure",
                        )
                        continue
                    elif not fireworks_account_id:
                        store_experiment_link(
                            experiment_id,
                            "No Fireworks account ID found",
                            "failure",
                        )
                        continue

                    api_base = get_fireworks_api_base()
                    headers = {
                        "Authorization": f"Bearer {fireworks_api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": get_user_agent(),
                    }

                    # Make dataset first

                    dataset_payload = {  # pyright: ignore[reportUnknownVariableType]
                        "dataset": {
                            "displayName": dataset_name,
                            "evalProtocol": {},
                            "format": "FORMAT_UNSPECIFIED",
                            "exampleCount": f"{len(exp_rows)}",
                        },
                        "datasetId": dataset_name,
                    }

                    dataset_url = f"{api_base}/v1/accounts/{fireworks_account_id}/datasets"
                    dataset_response = requests.post(dataset_url, json=dataset_payload, headers=headers)  # pyright: ignore[reportUnknownArgumentType]

                    # Skip if dataset creation failed
                    if dataset_response.status_code not in [200, 201]:
                        store_experiment_link(
                            experiment_id,
                            f"Dataset creation failed: {dataset_response.status_code} {dataset_response.text}",
                            "failure",
                        )
                        continue

                    dataset_data: dict[str, Any] = dataset_response.json()  # pyright: ignore[reportAny, reportExplicitAny]
                    dataset_id = dataset_data.get("datasetId", dataset_name)  # pyright: ignore[reportAny]

                    # Upload the JSONL file content
                    upload_url = f"{api_base}/v1/accounts/{fireworks_account_id}/datasets/{dataset_id}:upload"
                    with open(exp_file, "rb") as f:
                        files = {"file": f}
                        upload_headers = {
                            "Authorization": f"Bearer {fireworks_api_key}",
                            "User-Agent": get_user_agent(),
                        }
                        upload_response = requests.post(upload_url, files=files, headers=upload_headers)

                    # Skip if upload failed
                    if upload_response.status_code not in [200, 201]:
                        store_experiment_link(
                            experiment_id,
                            f"File upload failed: {upload_response.status_code} {upload_response.text}",
                            "failure",
                        )
                        continue

                    # Create evaluation job (optional - don't skip experiment if this fails)
                    # Truncate job ID to fit 63 character limit
                    job_id_base = f"{dataset_name}-job"
                    if len(job_id_base) > 63:
                        # Keep the "-job" suffix and truncate the dataset_name part
                        max_dataset_name_len = 63 - 4  # 4 = len("-job")
                        truncated_dataset_name = dataset_name[:max_dataset_name_len]
                        job_id_base = f"{truncated_dataset_name}-job"

                    eval_job_payload = {
                        "evaluationJobId": job_id_base,
                        "evaluationJob": {
                            "evaluator": f"accounts/{fireworks_account_id}/evaluators/dummy",
                            "inputDataset": f"accounts/{fireworks_account_id}/datasets/dummy",
                            "outputDataset": f"accounts/{fireworks_account_id}/datasets/{dataset_id}",
                        },
                    }

                    eval_job_url = f"{api_base}/v1/accounts/{fireworks_account_id}/evaluationJobs"
                    eval_response = requests.post(eval_job_url, json=eval_job_payload, headers=headers)

                    if eval_response.status_code in [200, 201]:
                        eval_job_data = eval_response.json()  # pyright: ignore[reportAny]
                        job_id = eval_job_data.get("evaluationJobId", job_id_base)  # pyright: ignore[reportAny]

                        store_experiment_link(
                            experiment_id,
                            f"https://app.fireworks.ai/dashboard/evaluation-jobs/{job_id}",
                            "success",
                        )
                    else:
                        store_experiment_link(
                            experiment_id,
                            f"Job creation failed: {eval_response.status_code} {eval_response.text}",
                            "failure",
                        )

    except Exception as e:
        # Do not fail evaluation if experiment JSONL writing fails
        print(f"Warning: Failed to persist results: {e}")
        pass
