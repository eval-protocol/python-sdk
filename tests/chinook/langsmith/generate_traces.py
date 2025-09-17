"""Generate synthetic Chinook traces and send them to LangSmith.

This module mirrors the Braintrust and Langfuse generators by:
- Reusing the Chinook dataset for inputs/ground truth
- Emitting assistant/tool conversations so the LangSmith adapter has
  realistic transcripts to parse
- Storing identifiers so follow-up evaluations can map runs back to
  their dataset rows

Run with pytest to create traces locally (skipped in CI):
    LANGSMITH_API_KEY=... pytest tests/chinook/langsmith/generate_traces.py
"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest import NoOpRolloutProcessor, evaluation_test

from tests.chinook.dataset import collect_dataset

try:  # Optional dependency: LangSmith client for logging traces
    from langsmith import Client, RunTree  # type: ignore

    LANGSMITH_INSTALLED = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    LANGSMITH_INSTALLED = False
    Client = None  # type: ignore


PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LS_PROJECT") or "ep-chinook-langsmith"
TRACE_TAGS = ["chinook_sql"]

langsmith_client = None
if LANGSMITH_INSTALLED:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", PROJECT_NAME)
    try:
        langsmith_client = Client()
    except Exception as exc:  # pragma: no cover - network/auth issues surfaced to caller
        print(f"⚠️ LangSmith client unavailable: {exc}")
        langsmith_client = None

dataset_rows = collect_dataset()

pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this generator locally")


def _synthetic_tool_schema() -> List[dict]:
    """Return a lightweight tool schema describing the Chinook SQL executor."""

    return [
        {
            "type": "function",
            "function": {
                "name": "chinook.sql_query",
                "description": "Execute a SQL statement against the Chinook dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "The SQL query to run"},
                        "notes": {"type": "string", "description": "Optional execution context"},
                    },
                    "required": ["sql"],
                },
            },
        }
    ]


def _populate_conversation(row: EvaluationRow, index: int) -> None:
    """Populate the row with a synthetic tool-call conversation."""

    question = row.messages[0].content if row.messages else f"Chinook task {index + 1}"
    task_id = f"chinook_task_{index + 1}"
    tool_call_id = f"chinook_sql_call_{index + 1}"

    row.tools = _synthetic_tool_schema()
    row.input_metadata = InputMetadata(dataset_info={"task_id": task_id})

    row.messages = [
        Message(role="user", content=question, name=task_id),
        Message(
            role="assistant",
            content="Let me query the Chinook database to gather those results.",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "chinook.sql_query",
                        "arguments": json.dumps(
                            {
                                "sql": f"-- Placeholder SQL for {task_id}",
                                "notes": "Synthetic LangSmith trace to exercise the adapter",
                            }
                        ),
                    },
                }
            ],
        ),
        Message(
            role="tool",
            name="chinook.sql_query",
            tool_call_id=tool_call_id,
            content=row.ground_truth or "No matching rows were returned.",
        ),
        Message(
            role="assistant",
            content=(
                row.ground_truth
                if row.ground_truth
                else "I could not find matching information in the Chinook catalog."
            ),
        ),
    ]


def _log_to_langsmith(row: EvaluationRow, index: int) -> None:
    """Send the populated conversation to LangSmith, if possible."""

    if langsmith_client is None:
        if LANGSMITH_INSTALLED:
            print("⚠️ Skipping LangSmith logging because the client could not be initialised.")
        else:
            print("⚠️ LangSmith package not installed; install `eval-protocol[langsmith]` to log traces.")
        return

    try:
        conversation = [message.model_dump(exclude_none=True) for message in row.messages]
        question = next((m.content for m in row.messages if m.role == "user"), "")
        conversation_inputs = conversation[:-1] if len(conversation) > 1 else conversation
        final_assistant = next(
            (m for m in reversed(row.messages) if m.role == "assistant" and m.content),
            None,
        )
        final_answer = (
            final_assistant.content
            if final_assistant and final_assistant.content
            else row.ground_truth
            if row.ground_truth
            else ""
        )

        task_id = None
        if row.input_metadata and row.input_metadata.dataset_info:
            task_id = row.input_metadata.dataset_info.get("task_id")
        if task_id is None:
            task_id = next(
                (m.name for m in row.messages if m.role == "user" and m.name),
                None,
            )

        structured_inputs = {
            "question": question,
            "ground_truth": row.ground_truth,
            "task_id": task_id,
            "messages": conversation_inputs,
        }
        if row.tools:
            structured_inputs["tools"] = row.tools

        now = datetime.now(timezone.utc)
        run_id = uuid.uuid4()
        run_tree = RunTree(
            id=run_id,
            trace_id=run_id,
            name=f"chinook-sql-trace-{index + 1}",
            run_type="chain",
            project_name=PROJECT_NAME,
            inputs=structured_inputs,
            outputs={"output": final_answer},
            tags=list(TRACE_TAGS),
            start_time=now,
            end_time=now,
            extra={
                "metadata": {
                    "task_id": task_id,
                    "logged_at": now.isoformat(),
                    "source": "eval-protocol chinook synthetic trace",
                }
            },
            ls_client=langsmith_client,
        )

        for offset, message in enumerate(row.messages):
            event_time = now + timedelta(milliseconds=offset)
            run_tree.add_event(
                {
                    "name": f"{message.role}_message",
                    "time": event_time.isoformat(),
                    "kwargs": {"message": message.model_dump(exclude_none=True)},
                }
            )

        assistant_with_tools = next(
            (m for m in row.messages if m.role == "assistant" and m.tool_calls),
            None,
        )
        if assistant_with_tools:
            for call_index, tool_call in enumerate(assistant_with_tools.tool_calls or []):
                tool_call_dict = (
                    tool_call.model_dump(exclude_none=True) if hasattr(tool_call, "model_dump") else tool_call
                )
                if not isinstance(tool_call_dict, dict):
                    tool_call_dict = {"raw_tool_call": tool_call_dict}

                function_call = tool_call_dict.get("function", {})
                if not isinstance(function_call, dict):
                    function_call = {
                        "name": function_call,
                    }
                arguments = function_call.get("arguments")
                parsed_arguments = {}
                if isinstance(arguments, str):
                    try:
                        parsed_arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        parsed_arguments = {"raw_arguments": arguments}
                elif isinstance(arguments, dict):
                    parsed_arguments = arguments

                call_time = now + timedelta(milliseconds=100 + call_index)
                tool_run = run_tree.create_child(
                    name=function_call.get("name", tool_call_dict.get("id", "chinook.sql_query")),
                    run_type="tool",
                    inputs=parsed_arguments,
                    outputs={"output": row.ground_truth if row.ground_truth else "No matching rows were returned."},
                    tags=list(TRACE_TAGS),
                    start_time=call_time,
                    end_time=call_time + timedelta(milliseconds=1),
                )
                metadata = tool_run.extra.setdefault("metadata", {})
                metadata.update(
                    {
                        "tool_call_id": tool_call_dict.get("id"),
                        "task_id": task_id,
                        "call_index": call_index,
                    }
                )

        run_tree.end_time = datetime.now(timezone.utc)
        run_tree.post(exclude_child_runs=False)

        if row.input_metadata is None:
            row.input_metadata = InputMetadata(dataset_info={})
        if row.input_metadata.dataset_info is None:
            row.input_metadata.dataset_info = {}
        if task_id:
            row.input_metadata.dataset_info.setdefault("task_id", task_id)
        session_data = row.input_metadata.session_data or {}
        session_data.setdefault("langsmith_run_id", str(run_id))
        session_data.setdefault("langsmith_trace_id", str(run_tree.trace_id))
        session_data.setdefault("langsmith_project", PROJECT_NAME)
        row.input_metadata.session_data = session_data
    except Exception as exc:  # pragma: no cover - surfaces API/network failures to the user
        print(f"❌ Failed to create LangSmith run: {exc}")


def _process_row(row: EvaluationRow, index: int) -> EvaluationRow:
    """Populate, log, and return the evaluation row."""

    _populate_conversation(row, index)
    _log_to_langsmith(row, index)
    return row


@pytest.mark.asyncio
@evaluation_test(
    input_rows=[dataset_rows[0:1]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        }
    ],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
async def test_generate_chinook_trace_0(row: EvaluationRow) -> EvaluationRow:
    return _process_row(row, 0)


@pytest.mark.asyncio
@evaluation_test(
    input_rows=[dataset_rows[1:2]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        }
    ],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
async def test_generate_chinook_trace_1(row: EvaluationRow) -> EvaluationRow:
    return _process_row(row, 1)


@pytest.mark.asyncio
@evaluation_test(
    input_rows=[dataset_rows[2:3]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        }
    ],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
async def test_generate_chinook_trace_2(row: EvaluationRow) -> EvaluationRow:
    return _process_row(row, 2)


@pytest.mark.asyncio
@evaluation_test(
    input_rows=[dataset_rows[3:4]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        }
    ],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
async def test_generate_chinook_trace_3(row: EvaluationRow) -> EvaluationRow:
    return _process_row(row, 3)


@pytest.mark.asyncio
@evaluation_test(
    input_rows=[dataset_rows[4:5]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        }
    ],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
async def test_generate_chinook_trace_4(row: EvaluationRow) -> EvaluationRow:
    return _process_row(row, 4)


@pytest.mark.asyncio
@evaluation_test(
    input_rows=[dataset_rows[5:6]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        }
    ],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
async def test_generate_chinook_trace_5(row: EvaluationRow) -> EvaluationRow:
    return _process_row(row, 5)
