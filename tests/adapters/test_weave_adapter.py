import os
import importlib.util
from pathlib import Path

import pytest


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"Failed to load module spec for {name} from {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.mark.skip(reason="Weave example only: converter IO smoke-test placeholder (no live fetch script).")
def test_weave_converter_basic_messages():
    root = Path(__file__).resolve().parents[2]
    converter_path = root / "examples" / "tracing" / "weave" / "converter.py"
    mod = _load_module_from_path("weave_converter", str(converter_path))
    convert = getattr(mod, "convert_trace_to_evaluation_row")

    trace = {
        "id": "tr_123",
        "project_id": "team/proj",
        "inputs": {"messages": [{"role": "user", "content": "Hi"}]},
        "output": {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]},
    }

    row = convert(trace)
    assert len(row.messages) >= 1
    assert row.input_metadata.session_data.get("weave_trace_id") == "tr_123"


@pytest.mark.skip(reason="Credential-gated live fetch; enable locally with WANDB creds.")
def test_weave_fetch_and_convert_live():
    # Require explicit env to avoid CI failures
    if not os.getenv("WANDB_API_KEY"):
        pytest.skip("WANDB_API_KEY not set")

    team = os.getenv("WANDB_ENTITY") or os.getenv("WEAVE_TEAM_ID")
    project = os.getenv("WANDB_PROJECT") or os.getenv("WEAVE_PROJECT_ID")
    if not team or not project:
        pytest.skip("Weave project not configured")

    base_url = os.getenv("WEAVE_TRACE_BASE_URL", "https://trace.wandb.ai")
    root = Path(__file__).resolve().parents[2]
    pull_path = root / "examples" / "tracing" / "weave" / "pull_output_traces.py"
    conv_path = root / "examples" / "tracing" / "weave" / "converter.py"

    pull_mod = _load_module_from_path("weave_pull", str(pull_path))
    conv_mod = _load_module_from_path("weave_converter", str(conv_path))

    fetch_weave_traces = getattr(pull_mod, "fetch_weave_traces")
    convert = getattr(conv_mod, "convert_trace_to_evaluation_row")

    traces = fetch_weave_traces(
        base_url=base_url, project_id=f"{team}/{project}", api_token=os.environ["WANDB_API_KEY"], limit=1
    )
    rows = [convert(tr) for tr in traces]
    assert any(r is not None for r in rows)
