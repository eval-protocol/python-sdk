import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from urllib.parse import urlencode

import requests

from .auth import get_fireworks_account_id, get_fireworks_api_base, get_fireworks_api_key
from .common_utils import get_user_agent


def _map_api_host_to_app_host(api_base: str) -> str:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(api_base)
        host = (parsed.netloc or parsed.path).lower()
        scheme = parsed.scheme or "https"

        # Explicit mappings first
        if host.startswith("dev.api.fireworks.ai"):
            return f"{scheme}://dev.fireworks.ai"
        if host == "staging.api.fireworks.ai" or host == "api.fireworks.ai":
            return f"{scheme}://app.fireworks.ai"

        # Generic mapping: api.<...> → app.<...>
        if host.startswith("api."):
            return f"{scheme}://{host.replace('api.', 'app.', 1)}"

        return f"{scheme}://{host}"
    except Exception:
        return "https://app.fireworks.ai"


def detect_dataset_builder(metric_dir: str) -> Optional[str]:
    """
    Best-effort scan for a dataset builder callable inside the metric directory.
    Returns a builder spec string in the form "path/to/module.py::function" if found.
    """
    try:
        candidates: list[Tuple[str, str]] = []
        for root, _, files in os.walk(metric_dir):
            for name in files:
                if not name.endswith(".py"):
                    continue
                file_path = os.path.join(root, name)
                # Load module via file location
                module_name = Path(file_path).stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                try:
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
                except Exception:
                    continue
                # Common exported symbol names
                symbol_names = [
                    "build_training_dataset",
                    "get_training_dataset",
                    "get_dataset",
                    "dataset",
                    "DATASET_BUILDER",
                ]
                for symbol in symbol_names:
                    if hasattr(module, symbol):
                        candidates.append((file_path, symbol))
        if not candidates:
            return None
        # Prefer build_training_dataset then get_training_dataset, else first
        preference = {
            "build_training_dataset": 0,
            "get_training_dataset": 1,
            "get_dataset": 2,
            "dataset": 3,
            "DATASET_BUILDER": 4,
        }
        candidates.sort(key=lambda x: preference.get(x[1], 99))
        best_file, best_symbol = candidates[0]
        return f"{best_file}::{best_symbol}"
    except Exception:
        return None


def _import_builder(builder_spec: str) -> Callable[[], Iterable[Dict[str, Any]]]:
    target, func = builder_spec.split("::", 1)
    # If target looks like a path, load from file
    if "/" in target or target.endswith(".py") or os.path.exists(target):
        file_path = target if target.endswith(".py") else f"{target}.py"
        if not os.path.isfile(file_path):
            raise ValueError(f"Builder file not found: {file_path}")
        module_name = Path(file_path).stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise ValueError(f"Unable to load builder module: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    else:
        # Treat as module path
        module = importlib.import_module(target)
    if not hasattr(module, func):
        raise ValueError(f"Function '{func}' not found in module '{getattr(module, '__name__', target)}'")
    callable_obj = getattr(module, func)
    if callable(callable_obj):
        return callable_obj  # type: ignore[return-value]
    # If symbol is a constant like DATASET_BUILDER, expect it to be callable
    if hasattr(callable_obj, "__call__"):
        return callable_obj  # type: ignore[return-value]
    raise ValueError("Dataset builder is not callable")


def materialize_dataset_via_builder(builder_spec: str, output_path: Optional[str] = None) -> Tuple[str, int]:
    builder = _import_builder(builder_spec)
    rows_iter = builder()
    if output_path is None:
        fd, tmp_path = tempfile.mkstemp(prefix="ep_rft_dataset_", suffix=".jsonl")
        os.close(fd)
        output_path = tmp_path
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows_iter:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return output_path, count


def create_dataset_from_jsonl(
    account_id: str,
    api_key: str,
    api_base: str,
    dataset_id: str,
    display_name: Optional[str],
    jsonl_path: str,
) -> Tuple[str, Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    # Count examples quickly
    example_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            example_count += 1

    dataset_url = f"{api_base.rstrip('/')}/v1/accounts/{account_id}/datasets"
    payload = {
        "dataset": {
            "displayName": display_name or dataset_id,
            "evalProtocol": {},
            "format": "FORMAT_UNSPECIFIED",
            "exampleCount": str(example_count),
        },
        "datasetId": dataset_id,
    }
    resp = requests.post(dataset_url, json=payload, headers=headers, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Dataset creation failed: {resp.status_code} {resp.text}")
    ds = resp.json()

    upload_url = f"{api_base.rstrip('/')}/v1/accounts/{account_id}/datasets/{dataset_id}:upload"
    with open(jsonl_path, "rb") as f:
        files = {"file": f}
        up_headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": get_user_agent(),
        }
        up_resp = requests.post(upload_url, files=files, headers=up_headers, timeout=600)
    if up_resp.status_code not in (200, 201):
        raise RuntimeError(f"Dataset upload failed: {up_resp.status_code} {up_resp.text}")
    return dataset_id, ds


def create_reinforcement_fine_tuning_job(
    account_id: str,
    api_key: str,
    api_base: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/accounts/{account_id}/reinforcementFineTuningJobs"
    # Move optional jobId from body to query parameter if provided
    job_id = body.get("jobId")
    if isinstance(job_id, str):
        job_id = job_id.strip()
    if job_id:
        # Remove from body and append as query param
        body.pop("jobId", None)
        url = f"{url}?{urlencode({'reinforcementFineTuningJobId': job_id})}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": get_user_agent(),
    }
    resp = requests.post(url, json=body, headers=headers, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"RFT job creation failed: {resp.status_code} {resp.text}")
    return resp.json()


def build_default_dataset_id(evaluator_id: str) -> str:
    ts = time.strftime("%Y%m%d%H%M%S")
    base = evaluator_id.lower().replace("_", "-")
    return f"{base}-dataset-{ts}"


def build_default_output_model(evaluator_id: str) -> str:
    base = evaluator_id.lower().replace("_", "-")
    uuid_suffix = str(uuid.uuid4())[:4]
    return f"{base}-rft-{uuid_suffix}"


__all__ = [
    "detect_dataset_builder",
    "materialize_dataset_via_builder",
    "create_dataset_from_jsonl",
    "create_reinforcement_fine_tuning_job",
    "build_default_dataset_id",
    "build_default_output_model",
    "_map_api_host_to_app_host",
]
