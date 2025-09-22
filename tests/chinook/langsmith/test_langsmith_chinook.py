"""Evaluate LangSmith traces generated from the Chinook dataset."""

import os
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata
from eval_protocol.pytest import NoOpRolloutProcessor, evaluation_test

from tests.chinook.dataset import collect_dataset

try:
    from eval_protocol.adapters.langsmith import create_langsmith_adapter

    ADAPTER_AVAILABLE = True
except ImportError:  # pragma: no cover - adapter extras not installed
    ADAPTER_AVAILABLE = False
    create_langsmith_adapter = None  # type: ignore

try:
    from langsmith import Client  # type: ignore

    LANGSMITH_CLIENT: Optional[Client]
    try:
        LANGSMITH_CLIENT = Client()
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        print(f"⚠️ LangSmith client unavailable: {exc}")
        LANGSMITH_CLIENT = None
except ImportError:  # pragma: no cover - optional dependency
    LANGSMITH_CLIENT = None

PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LS_PROJECT") or "ep-chinook-langsmith"
TRACE_TAGS = ["chinook_sql"]

dataset_rows = collect_dataset()
TASK_ID_TO_GROUND_TRUTH: Dict[str, str] = {
    f"chinook_task_{index + 1}": row.ground_truth for index, row in enumerate(dataset_rows)
}
QUESTION_TO_GROUND_TRUTH: Dict[str, str] = {
    row.messages[0].content: row.ground_truth for row in dataset_rows if row.messages
}

LLM_JUDGE_PROMPT = (
    "Your job is to compare the response to the expected answer.\n"
    "The response will be a narrative report of the query results.\n"
    "If the response contains the same or well summarized information as the expected answer, return 1.0.\n"
    "If the response does not contain the same information or is missing information, return 0.0."
)


def _attach_ground_truth(row: EvaluationRow) -> None:
    """Populate the row's ground truth using the stored task identifiers."""

    dataset_id = None
    for message in row.messages:
        if message.role == "user" and message.name:
            dataset_id = message.name
            break

    ground_truth = None
    if dataset_id:
        ground_truth = TASK_ID_TO_GROUND_TRUTH.get(dataset_id)

    if ground_truth is None:
        user_message = next((msg.content for msg in row.messages if msg.role == "user"), None)
        if user_message:
            ground_truth = QUESTION_TO_GROUND_TRUTH.get(user_message)

    if ground_truth is None:
        ground_truth = ""

    row.ground_truth = ground_truth

    if row.input_metadata is None:
        row.input_metadata = InputMetadata(session_data={})

    if row.input_metadata.session_data is None:
        row.input_metadata.session_data = {}

    if dataset_id:
        row.input_metadata.session_data.setdefault("chinook_task_id", dataset_id)

    dataset_info = row.input_metadata.dataset_info or {}
    if dataset_id:
        dataset_info["task_id"] = dataset_id
    row.input_metadata.dataset_info = dataset_info or None


def fetch_langsmith_traces(limit: int = 20) -> List[EvaluationRow]:
    """Use the LangSmith adapter to convert traces into evaluation rows."""

    if not ADAPTER_AVAILABLE or create_langsmith_adapter is None:
        print("⚠️ LangSmith adapter unavailable - install `eval-protocol[langsmith]`.")
        return []

    try:
        adapter = create_langsmith_adapter()
    except Exception as exc:
        print(f"❌ Failed to create LangSmithAdapter: {exc}")
        return []

    try:
        rows = adapter.get_evaluation_rows(
            project_name=PROJECT_NAME,
            limit=limit,
            include_tool_calls=True,
            tags=TRACE_TAGS,
            order_by="-created_at",
        )
    except Exception as exc:
        print(f"❌ LangSmithAdapter failed to pull rows: {exc}")
        return []

    for row in rows:
        _attach_ground_truth(row)

    return rows


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip LangSmith adapter test in CI")
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[fetch_langsmith_traces()],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
async def test_langsmith_chinook(row: EvaluationRow) -> EvaluationRow:
    """Evaluate LangSmith-sourced traces using the Chinook judge."""

    assert row.tools, "Expected LangSmith traces to include available tool metadata"
    assert any(message.tool_calls for message in row.messages if message.role == "assistant"), (
        "Expected at least one assistant message to include tool calls"
    )

    last_assistant_message = row.last_assistant_message()
    if last_assistant_message is None or not last_assistant_message.content:
        row.evaluation_result = EvaluateResult(score=0.0, reason="No assistant message found")
        return row

    model = OpenAIChatModel("accounts/fireworks/models/kimi-k2-instruct", provider="fireworks")

    class JudgeResponse(BaseModel):
        score: float
        reason: str

    comparison_agent = Agent(
        model=model,
        system_prompt=LLM_JUDGE_PROMPT,
        output_type=JudgeResponse,
        output_retries=5,
    )

    result = await comparison_agent.run(
        f"Expected answer: {row.ground_truth}\nResponse: {last_assistant_message.content}"
    )
    row.evaluation_result = EvaluateResult(
        score=result.output.score,
        reason=result.output.reason,
    )

    if LANGSMITH_CLIENT and row.input_metadata and row.input_metadata.session_data:
        run_id = row.input_metadata.session_data.get("langsmith_run_id")
        if run_id:
            try:
                LANGSMITH_CLIENT.create_feedback(
                    run_id=run_id,
                    key="ep_chinook_accuracy",
                    score=row.evaluation_result.score,
                    comment=row.evaluation_result.reason,
                )
            except Exception as exc:
                print(f"⚠️ Failed to push LangSmith feedback: {exc}")

    return row
