"""
External Evaluation Pipeline: Pull Langfuse traces and evaluate them using EP framework.

This script:
1. Pulls traces from Langfuse (created by generate_traces.py)
2. Uses the fixed LangfuseAdapter for proper conversation extraction
3. Evaluates them using the same LLM judge as test_pydantic_chinook.py
4. Uses NoOpRolloutProcessor since traces already exist
5. Pushes evaluation scores back to Langfuse
"""

import os
from datetime import datetime, timedelta
from typing import List

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluateResult, EvaluationRow, Message, InputMetadata
from eval_protocol.pytest import evaluation_test, NoOpRolloutProcessor

# Langfuse client setup
try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
    langfuse = Langfuse()
except ImportError:
    LANGFUSE_AVAILABLE = False
    langfuse = None

# Same LLM judge logic from test_pydantic_chinook.py
LLM_JUDGE_PROMPT = (
    "Your job is to compare the response to the expected answer.\n"
    "The response will be a narrative report of the query results.\n"
    "If the response contains the same or well summarized information as the expected answer, return 1.0.\n"
    "If the response does not contain the same information or is missing information, return 0.0."
)


class Response(BaseModel):
    score: float
    reason: str


def langfuse_data_generator(hours_back: int = 168, tags: List[str] = ["chinook_sql"]) -> List[EvaluationRow]:
    try:
        from eval_protocol.adapters.langfuse import create_langfuse_adapter

        adapter = create_langfuse_adapter()

        now = datetime.now()
        from_timestamp = now - timedelta(hours=hours_back)

        return adapter.get_evaluation_rows(
            limit=20, from_timestamp=from_timestamp, to_timestamp=now, include_tool_calls=True, tags=tags
        )

    except Exception as e:
        print(f"❌ LangfuseAdapter failed: {e}")
        return []


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.asyncio
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[langfuse_data_generator],
    ),
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
async def test_langfuse_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Pull the complex query traces from Langfuse and evaluate using logic from test_pydantic_chinook.py::test_complex_queries

    This test:
    1. Gets traces from Langfuse (via fixed LangfuseAdapter)
    2. Uses NoOpRolloutProcessor (traces already exist)
    3. Evaluates each trace using same LLM judge as PydanticAI test
    4. Pushes scores back to Langfuse
    """
    # Same eval logic as PydanticAI example
    last_assistant_message = row.last_assistant_message()
    if last_assistant_message is None:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason="No assistant message found",
        )
    elif not last_assistant_message.content:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason="No assistant message found",
        )
    else:
        model = OpenAIChatModel(
            "accounts/fireworks/models/kimi-k2-instruct",
            provider="fireworks",
        )

        class Response(BaseModel):
            """
            A score between 0.0 and 1.0 indicating whether the response is correct.
            """

            score: float

            """
            A short explanation of why the response is correct or incorrect.
            """
            reason: str

        comparison_agent = Agent(
            model=model,
            system_prompt=LLM_JUDGE_PROMPT,
            output_type=Response,
            output_retries=5,
        )
        result = await comparison_agent.run(
            f"Expected answer: {row.ground_truth}\nResponse: {last_assistant_message.content}"
        )
        row.evaluation_result = EvaluateResult(
            score=result.output.score,
            reason=result.output.reason,
        )

    # Push score back to Langfuse
    if langfuse and row.evaluation_result and row.input_metadata:
        trace_id = row.input_metadata.dataset_info.get("trace_id") if row.input_metadata.dataset_info else None
        if trace_id:
            langfuse.create_score(
                trace_id=trace_id,
                name="ep_chinook_accuracy",
                value=row.evaluation_result.score,
                comment=row.evaluation_result.reason,
            )

    return row
