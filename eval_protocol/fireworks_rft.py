import importlib.util
import json
import os
import sys
import tempfile
import time
import uuid
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from .fireworks_client import create_fireworks_client


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
    """Create a dataset and upload a JSONL file using the Fireworks SDK client.

    This function uses the Fireworks SDK client which properly handles authentication
    including extra headers set via FIREWORKS_EXTRA_HEADERS environment variable.

    Args:
        account_id: The Fireworks account ID.
        api_key: Fireworks API key.
        api_base: Fireworks API base URL.
        dataset_id: The ID for the new dataset.
        display_name: Display name for the dataset (optional).
        jsonl_path: Path to the JSONL file to upload.

    Returns:
        A tuple of (dataset_id, dataset_response_dict).

    Raises:
        RuntimeError: If dataset creation or upload fails.
    """
    # Count examples quickly
    example_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            example_count += 1

    # Create Fireworks client with consistent configuration
    client = create_fireworks_client(
        api_key=api_key,
        account_id=account_id,
        base_url=api_base,
    )

    try:
        # Create the dataset
        dataset = client.datasets.create(
            account_id=account_id,
            dataset_id=dataset_id,
            dataset={
                "display_name": display_name or dataset_id,
                "eval_protocol": {},
                "format": "FORMAT_UNSPECIFIED",
                "example_count": str(example_count),
            },
            timeout=60.0,
        )
    except Exception as e:
        raise RuntimeError(f"Dataset creation failed: {e}") from e

    try:
        # Upload the JSONL file
        with open(jsonl_path, "rb") as f:
            client.datasets.upload(
                dataset_id=dataset_id,
                account_id=account_id,
                file=f,
                timeout=600.0,
            )
    except Exception as e:
        raise RuntimeError(f"Dataset upload failed: {e}") from e

    # Convert SDK response to dict for backwards compatibility
    ds_dict: Dict[str, Any] = {}
    if hasattr(dataset, "model_dump"):
        ds_dict = dataset.model_dump()
    elif hasattr(dataset, "dict"):
        ds_dict = dataset.dict()
    else:
        # Fallback: extract known fields
        ds_dict = {
            "name": getattr(dataset, "name", None),
            "state": getattr(dataset, "state", None),
        }

    return dataset_id, ds_dict


def build_default_dataset_id(evaluator_id: str) -> str:
    ts = time.strftime("%Y%m%d%H%M%S")
    base = evaluator_id.lower().replace("_", "-")
    return f"{base}-dataset-{ts}"


def build_default_output_model(evaluator_id: str) -> str:
    base = evaluator_id.lower().replace("_", "-")
    uuid_suffix = str(uuid.uuid4())[:4]

    # suffix is "-rft-{4chars}" -> 9 chars
    suffix_len = 9
    max_len = 63

    # Check if we need to truncate
    if len(base) + suffix_len > max_len:
        # Calculate hash of the full base to preserve uniqueness
        hash_digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:6]
        # New structure: {truncated_base}-{hash}-{uuid_suffix}
        # Space needed for "-{hash}" is 1 + 6 = 7
        hash_part_len = 7

        allowed_base_len = max_len - suffix_len - hash_part_len
        truncated_base = base[:allowed_base_len].strip("-")

        return f"{truncated_base}-{hash_digest}-rft-{uuid_suffix}"

    return f"{base}-rft-{uuid_suffix}"


__all__ = [
    "detect_dataset_builder",
    "materialize_dataset_via_builder",
    "create_dataset_from_jsonl",
    "build_default_dataset_id",
    "build_default_output_model",
    "_map_api_host_to_app_host",
]
