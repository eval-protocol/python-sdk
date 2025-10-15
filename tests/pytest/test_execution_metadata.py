import pytest
from openai.types import CompletionUsage

from eval_protocol.models import EvaluationRow, ExecutionMetadata, InputMetadata, CostMetrics, Message
from eval_protocol.pytest.evaluation_test_utils import add_cost_metrics


class TestExecutionMetadata:
    """Test execution metadata tracking including cost metrics, usage statistics, and timing."""

    def test_single_model_with_provider(self):
        """Test normal case: single model string with provider."""
        row = EvaluationRow(
            messages=[],
            input_metadata=InputMetadata(
                completion_params={"model": "accounts/fireworks/models/gpt-oss-120b", "provider": "fireworks"}
            ),
            execution_metadata=ExecutionMetadata(
                usage=CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            ),
        )

        add_cost_metrics(row)

        assert row.execution_metadata.cost_metrics is not None
        assert row.execution_metadata.cost_metrics.input_cost is not None
        assert row.execution_metadata.cost_metrics.output_cost is not None
        assert row.execution_metadata.cost_metrics.total_cost_dollar is not None

    @pytest.mark.skip(reason="Revisit when we figure out how to get cost metrics for multi-agent Pydantic.")
    def test_pydantic_ai_multi_agent_model_dict(self):
        """Test Pydantic AI multi-agent case: nested dictionary with multiple models."""
        row = EvaluationRow(
            messages=[],
            input_metadata=InputMetadata(
                completion_params={
                    "model": {
                        "joke_generation_model": {
                            "model": "accounts/fireworks/models/kimi-k2-instruct",
                            "provider": "fireworks",
                        },
                        "joke_selection_model": {
                            "model": "accounts/fireworks/models/deepseek-v3p1",
                            "provider": "fireworks",
                        },
                    }
                }
            ),
            execution_metadata=ExecutionMetadata(
                usage=CompletionUsage(prompt_tokens=200, completion_tokens=75, total_tokens=275)
            ),
        )

        add_cost_metrics(row)

        assert row.execution_metadata.cost_metrics is not None
        assert row.execution_metadata.cost_metrics.input_cost is not None
        assert row.execution_metadata.cost_metrics.output_cost is not None
        assert row.execution_metadata.cost_metrics.total_cost_dollar is not None

    def test_no_usage_stats(self):
        """Test case with no usage statistics."""
        row = EvaluationRow(
            messages=[],
            input_metadata=InputMetadata(completion_params={"model": "gpt-3.5-turbo", "provider": "openai"}),
            execution_metadata=ExecutionMetadata(usage=None),
        )

        add_cost_metrics(row)

        assert row.execution_metadata.cost_metrics is not None
        assert row.execution_metadata.cost_metrics.input_cost == 0.0
        assert row.execution_metadata.cost_metrics.output_cost == 0.0
        assert row.execution_metadata.cost_metrics.total_cost_dollar == 0.0

    def test_no_completion_params(self):
        """Test case with empty completion parameters."""
        row = EvaluationRow(
            messages=[],
            input_metadata=InputMetadata(completion_params={}),
            execution_metadata=ExecutionMetadata(
                usage=CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            ),
        )

        add_cost_metrics(row)

        assert row.execution_metadata.cost_metrics is not None
        assert row.execution_metadata.cost_metrics.input_cost == 0.0
        assert row.execution_metadata.cost_metrics.output_cost == 0.0
        assert row.execution_metadata.cost_metrics.total_cost_dollar == 0.0

    def test_zero_tokens(self):
        """Test case with zero token usage."""
        row = EvaluationRow(
            messages=[],
            input_metadata=InputMetadata(completion_params={"model": "gpt-3.5-turbo", "provider": "openai"}),
            execution_metadata=ExecutionMetadata(
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            ),
        )

        add_cost_metrics(row)

        assert row.execution_metadata.cost_metrics is not None
        assert row.execution_metadata.cost_metrics.input_cost == 0.0
        assert row.execution_metadata.cost_metrics.output_cost == 0.0
        assert row.execution_metadata.cost_metrics.total_cost_dollar == 0.0

    def test_provider_mapping_variations(self):
        """Test different provider mappings."""
        providers_and_expected = [
            ("openai", "gpt-3.5-turbo", "gpt-3.5-turbo"),  # No prefix - known model
            (
                "fireworks",
                "accounts/fireworks/models/llama-v2-7b-chat",
                "fireworks_ai/accounts/fireworks/models/llama-v2-7b-chat",
            ),
            ("unknown_provider", "gpt-3.5-turbo", "gpt-3.5-turbo"),  # Fallback to original - use known model
        ]

        for provider, model, expected_model_id in providers_and_expected:
            row = EvaluationRow(
                messages=[],
                input_metadata=InputMetadata(completion_params={"model": model, "provider": provider}),
                execution_metadata=ExecutionMetadata(
                    usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
                ),
            )

            add_cost_metrics(row)

            # Should not raise an error and should set cost metrics
            assert row.execution_metadata.cost_metrics is not None

    def test_model_without_provider(self):
        """Test model string without provider field."""
        row = EvaluationRow(
            messages=[],
            input_metadata=InputMetadata(
                completion_params={"model": "gpt-3.5-turbo"}  # No provider field
            ),
            execution_metadata=ExecutionMetadata(
                usage=CompletionUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75)
            ),
        )

        add_cost_metrics(row)

        assert row.execution_metadata.cost_metrics is not None
        # Should still work for OpenAI models even without explicit provider

    def test_execution_metadata_timing_field(self):
        """Test that the new duration_seconds field works correctly."""
        metadata = ExecutionMetadata()

        # Check field exists and defaults to None
        assert hasattr(metadata, "duration_seconds")
        assert metadata.duration_seconds is None

        # Check it can be set
        metadata.duration_seconds = 1.234
        assert metadata.duration_seconds == 1.234
