import os
import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor

from tests.chinook.dataset import collect_dataset
from tests.chinook.pydantic.agent import setup_agent

dataset = collect_dataset()
current_idx = 0


class SpanIDCapturingProcessor:
    """Custom processor to capture span IDs when they open/close."""

    def on_start(self, span, parent_context=None):
        """Called when span starts - capture the ID."""
        global current_idx
        if span.name == "agent run":
            span.set_attribute("ground_truth", dataset[current_idx].ground_truth)
            current_idx += 1

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        pass


try:
    from braintrust.otel import BraintrustSpanProcessor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from pydantic_ai.agent import Agent
    from braintrust import init_logger

    BRAINTRUST_AVAILABLE = True

    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    provider.add_span_processor(BraintrustSpanProcessor())  # pyright: ignore[reportArgumentType]
    provider.add_span_processor(SpanIDCapturingProcessor())  # pyright: ignore[reportArgumentType]

    logger = init_logger(project="default-otel-project")

    Agent.instrument_all()

except ImportError:
    BRAINTRUST_AVAILABLE = False

    def setup_braintrust():
        pass


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[collect_dataset()[0:1]],
    completion_params=[
        {
            "model": {
                "orchestrator_agent_model": {
                    "model": "accounts/fireworks/models/kimi-k2-instruct",
                    "provider": "fireworks",
                }
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": setup_agent},
    mode="pointwise",
)
async def test_complex_query_0(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - Ground truth set by span processor during span creation.
    """
    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[collect_dataset()[1:2]],
    completion_params=[
        {
            "model": {
                "orchestrator_agent_model": {
                    "model": "accounts/fireworks/models/kimi-k2-instruct",
                    "provider": "fireworks",
                }
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": setup_agent},
    mode="pointwise",
)
async def test_complex_query_1(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Braintrust traces.
    """
    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[collect_dataset()[2:3]],
    completion_params=[
        {
            "model": {
                "orchestrator_agent_model": {
                    "model": "accounts/fireworks/models/kimi-k2-instruct",
                    "provider": "fireworks",
                }
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": setup_agent},
    mode="pointwise",
)
async def test_complex_query_2(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Braintrust traces.
    """
    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[collect_dataset()[3:4]],
    completion_params=[
        {
            "model": {
                "orchestrator_agent_model": {
                    "model": "accounts/fireworks/models/kimi-k2-instruct",
                    "provider": "fireworks",
                }
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": setup_agent},
    mode="pointwise",
)
async def test_complex_query_3(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Braintrust traces.
    """
    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[collect_dataset()[4:5]],
    completion_params=[
        {
            "model": {
                "orchestrator_agent_model": {
                    "model": "accounts/fireworks/models/kimi-k2-instruct",
                    "provider": "fireworks",
                }
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": setup_agent},
    mode="pointwise",
)
async def test_complex_query_4(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Braintrust traces.
    """
    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[collect_dataset()[5:6]],
    completion_params=[
        {
            "model": {
                "orchestrator_agent_model": {
                    "model": "accounts/fireworks/models/kimi-k2-instruct",
                    "provider": "fireworks",
                }
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": setup_agent},
    mode="pointwise",
)
async def test_complex_query_5(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Braintrust traces.
    """
    return row
