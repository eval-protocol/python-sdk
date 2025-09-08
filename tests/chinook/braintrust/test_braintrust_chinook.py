import os
from datetime import datetime, timedelta
from typing import List, Any, Dict

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, InputMetadata
from eval_protocol.pytest import evaluation_test, NoOpRolloutProcessor

try:
    from eval_protocol.adapters.braintrust import create_braintrust_adapter

    BRAINTRUST_AVAILABLE = True
except ImportError:
    BRAINTRUST_AVAILABLE = False
    create_braintrust_adapter = None


class Response(BaseModel):
    score: float
    reason: str


LLM_JUDGE_PROMPT = (
    "Your job is to compare the response to the expected answer.\n"
    "The response will be a narrative report of the query results.\n"
    "If the response contains the same or well summarized information as the expected answer, return 1.0.\n"
    "If the response does not contain the same information or is missing information, return 0.0."
)


def fetch_braintrust_traces_as_evaluation_rows(hours_back: int = 24) -> List[EvaluationRow]:
    """
    Dataset adapter: Use BraintrustAdapter to fetch traces from project logs.
    """
    if not BRAINTRUST_AVAILABLE or not create_braintrust_adapter:
        print("⚠️ Braintrust unavailable - no traces to evaluate")
        return []

    try:
        print("🧠 Using BraintrustAdapter to fetch Chinook traces")

        adapter = create_braintrust_adapter(
            project_id="df6863de-6ce2-4fcc-9995-1fa6605f8623"  # Your Braintrust project
        )

        # Use the adapter to fetch logs
        now = datetime.now()
        from_timestamp = now - timedelta(hours=hours_back)

        evaluation_rows = list(
            adapter.get_evaluation_rows(
                from_timestamp=from_timestamp,
                to_timestamp=now,
            )
        )

        print(f"✅ BraintrustAdapter extracted {len(evaluation_rows)} evaluation rows")
        return evaluation_rows

    except Exception as e:
        print(f"❌ BraintrustAdapter failed: {e}")
        return []


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[fetch_braintrust_traces_as_evaluation_rows(hours_back=168)],  # 1 week back
    rollout_processor=NoOpRolloutProcessor(),  # No-op since traces already exist
    mode="pointwise",
)
async def test_braintrust_trace_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    This test acts as an external evaluation pipeline for Braintrust traces.
    It:
    1. Gets traces from Braintrust (via dataset adapter)
    2. Uses NoOpRolloutProcessor (traces already exist)
    3. Evaluates each trace using same LLM judge as PydanticAI test
    4. Pushes scores back to Braintrust (if API supports it)
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
        model = OpenAIModel(
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

    return row
