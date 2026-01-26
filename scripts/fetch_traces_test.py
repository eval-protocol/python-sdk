#!/usr/bin/env python3
"""Simple script to fetch traces directly from Langfuse and parse them.

This bypasses the Fireworks tracing proxy (and its Redis insertion_id check)
by querying Langfuse directly.

Required env vars:
  LANGFUSE_PUBLIC_KEY - Your Langfuse public key
  LANGFUSE_SECRET_KEY - Your Langfuse secret key
  LANGFUSE_HOST - Langfuse host (default: https://cloud.langfuse.com)
  ROLLOUT_ID - The rollout_id to search for (default: test-test-test)
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

from eval_protocol.adapters.fireworks_tracing import convert_trace_dict_to_evaluation_row
from eval_protocol.models import EvaluationRow


os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-9470ba98-7ace-4fe0-b1dc-3dda0f66d812")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-36b11237-a230-4524-a6e0-3af372b6f5b6")
os.environ.setdefault("LANGFUSE_HOST", "https://langfuse-prod.fireworks.ai")  # EU region


def fetch_traces_from_langfuse(
    tags: List[str],
    limit: int = 100,
    hours_back: int = 24,
) -> List[Dict[str, Any]]:
    """Fetch traces directly from Langfuse (bypassing Fireworks proxy).

    This avoids the Redis insertion_id check by going straight to Langfuse.
    """
    try:
        from langfuse import Langfuse
    except ImportError:
        print("ERROR: langfuse not installed. Run: pip install langfuse")
        return []

    # Get Langfuse credentials from environment
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set")
        return []

    print(f"Connecting to Langfuse at {host}...")
    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    # Calculate time range
    to_ts = datetime.now()
    from_ts = to_ts - timedelta(hours=hours_back)

    print(f"Fetching traces with tags: {tags}")
    print(f"Time range: {from_ts} to {to_ts}")

    # Fetch trace list
    traces_response = client.api.trace.list(
        page=1,
        limit=limit,
        tags=tags,
        from_timestamp=from_ts,
        to_timestamp=to_ts,
        order_by="timestamp.desc",
    )

    if not traces_response or not traces_response.data:
        print("No traces found in list response")
        return []

    print(f"Found {len(traces_response.data)} trace summaries")

    # Fetch full trace details and serialize to dict
    traces: List[Dict[str, Any]] = []
    for trace_info in traces_response.data:
        try:
            trace_full = client.api.trace.get(trace_info.id)

            # Serialize to dict (same format as proxy returns)
            trace_dict = _serialize_trace_to_dict(trace_full)
            traces.append(trace_dict)

        except Exception as e:
            print(f"  Failed to fetch trace {trace_info.id}: {e}")

    print(f"Successfully fetched {len(traces)} full traces")
    return traces


def _serialize_trace_to_dict(trace_full: Any) -> Dict[str, Any]:
    """Convert Langfuse trace object to dict format (same as proxy does)."""
    timestamp = getattr(trace_full, "timestamp", None)

    return {
        "id": trace_full.id,
        "name": getattr(trace_full, "name", None),
        "user_id": getattr(trace_full, "user_id", None),
        "session_id": getattr(trace_full, "session_id", None),
        "tags": getattr(trace_full, "tags", []),
        "timestamp": str(timestamp) if timestamp else None,
        "input": getattr(trace_full, "input", None),
        "output": getattr(trace_full, "output", None),
        "metadata": getattr(trace_full, "metadata", None),
        "observations": [
            {
                "id": obs.id,
                "type": getattr(obs, "type", None),
                "name": getattr(obs, "name", None),
                "start_time": str(getattr(obs, "start_time", None)) if getattr(obs, "start_time", None) else None,
                "end_time": str(getattr(obs, "end_time", None)) if getattr(obs, "end_time", None) else None,
                "input": getattr(obs, "input", None),
                "output": getattr(obs, "output", None),
                "parent_observation_id": getattr(obs, "parent_observation_id", None),
                "metadata": getattr(obs, "metadata", None),
            }
            for obs in getattr(trace_full, "observations", [])
        ]
        if hasattr(trace_full, "observations")
        else [],
    }


def parse_traces_to_rows(traces: List[Dict[str, Any]], include_tool_calls: bool = True) -> List[EvaluationRow]:
    """Parse raw trace dicts to EvaluationRows using the same logic as get_evaluation_rows."""
    rows = []
    for trace in traces:
        try:
            row = convert_trace_dict_to_evaluation_row(trace, include_tool_calls)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"  Failed to convert trace {trace.get('id')}: {e}")
    return rows


def print_row_details(row: EvaluationRow, index: int):
    """Print details of a single EvaluationRow."""
    print(f"\n--- Row {index + 1} ---")
    print(f"Row ID: {row.input_metadata.row_id}")
    print(
        f"Trace ID: {row.input_metadata.session_data.get('langfuse_trace_id') if row.input_metadata.session_data else None}"
    )
    print(f"Rollout ID: {row.execution_metadata.rollout_id}")
    print(f"Invocation ID: {row.execution_metadata.invocation_id}")
    print(f"Experiment ID: {row.execution_metadata.experiment_id}")
    print(f"Run ID: {row.execution_metadata.run_id}")
    print(f"Finish Reason: {row.execution_metadata.finish_reason}")  # NEW
    print(f"Num messages: {len(row.messages)}")
    print(f"Tools: {row.tools is not None}")

    print("\nMessages:")
    for j, msg in enumerate(row.messages):
        content_preview = str(msg.content)[:100] if msg.content else "(empty)"
        tool_calls_info = f" [tool_calls: {len(msg.tool_calls)}]" if msg.tool_calls else ""
        print(f"  [{j}] {msg.role}: {content_preview}{tool_calls_info}")


def main():
    rollout_id = os.environ.get("ROLLOUT_ID", "test-test-test")
    hours_back = int(os.environ.get("HOURS_BACK", "24"))

    print(f"Rollout ID: {rollout_id}")
    print(f"Hours back: {hours_back}")
    print("=" * 60)

    # Step 1: Fetch raw traces directly from Langfuse
    print("\n[1] Fetching raw traces from Langfuse...")
    traces = fetch_traces_from_langfuse(
        tags=[f"rollout_id:{rollout_id}"],
        limit=10,
        hours_back=hours_back,
    )

    if not traces:
        print("\nNo traces found!")
        return

    # Step 2: Print raw trace structure (first trace only)
    print("\n[2] Raw trace structure (first trace):")
    print("-" * 60)
    first_trace = traces[0]
    print(f"ID: {first_trace.get('id')}")
    print(f"Name: {first_trace.get('name')}")
    print(f"Tags: {first_trace.get('tags')}")
    print(f"Input type: {type(first_trace.get('input'))}")
    print(f"Input: {json.dumps(first_trace.get('input'), indent=2)[:500]}...")
    print(f"Output type: {type(first_trace.get('output'))}")
    print(f"Output: {json.dumps(first_trace.get('output'), indent=2)[:500] if first_trace.get('output') else None}...")
    print(f"Num observations: {len(first_trace.get('observations', []))}")

    # Print observations
    for obs in first_trace.get("observations", []):
        print(f"\n  Observation: {obs.get('name')} ({obs.get('type')})")
        print(f"    Input type: {type(obs.get('input'))}")
        print(f"    Input: {json.dumps(obs.get('input'), indent=2)[:300] if obs.get('input') else None}...")
        print(f"    Output type: {type(obs.get('output'))}")
        print(f"    Output: {json.dumps(obs.get('output'), indent=2)[:300] if obs.get('output') else None}...")

    # Step 3: Parse to EvaluationRows
    print("\n[3] Parsing traces to EvaluationRows...")
    print("-" * 60)
    rows = parse_traces_to_rows(traces)

    print(f"\nSuccessfully parsed {len(rows)} / {len(traces)} traces")

    # Step 4: Print row details
    print("\n[4] EvaluationRow details:")
    print("=" * 60)
    for i, row in enumerate(rows):
        print_row_details(row, i)


if __name__ == "__main__":
    main()
