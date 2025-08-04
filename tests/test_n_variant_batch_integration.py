"""
End-to-end tests for N-variant generation to batch evaluation pipeline.
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from eval_protocol.execution.pipeline import EvaluationPipeline
from eval_protocol.generation.clients import GenerationResult
from eval_protocol.models import EvaluateResult, Message
from eval_protocol.typed_interface import reward_function
from eval_protocol.utils.batch_evaluation import (
    create_sample_batch_reward_function,
    run_batch_evaluation,
)
from eval_protocol.utils.batch_transformation import (
    create_batch_evaluation_dataset,
    transform_n_variant_jsonl_to_batch_format,
)


@pytest.fixture
def mock_reward_function():
    """Mock pointwise reward function for N-variant generation."""

    def mock_func(*args, **kwargs):
        return EvaluateResult(score=0.75, reason="Mock evaluation", is_score_valid=True, metrics={})

    return mock_func


@pytest.fixture
def n_variant_pipeline_config():
    """Pipeline configuration for N-variant generation."""
    return OmegaConf.create(
        {
            "generation": {
                "enabled": True,
                "model_name": "test-model",
                "n": 3,  # Generate 3 variants
                "cache": {"enabled": False},
                "api_params": {"max_concurrent_requests": 5},
            },
            "reward": {
                "function_path": "examples.row_wise.dummy_example.dummy_rewards.simple_echo_reward",
                "params": {},
            },
            "output": {"results_file": "test_results.jsonl"},
            "evaluation_params": {"limit_samples": None},
            "logging_params": {"batch_log_interval": 10},
        }
    )


@pytest.fixture
def sample_batch_reward_function():
    """Create a sample batch reward function for testing."""

    @reward_function(mode="batch")
    def test_batch_reward(
        rollouts_messages: list[list[Message]],
        ground_truth_for_eval: str = None,
        **kwargs,
    ) -> list[EvaluateResult]:
        """Test batch reward function that scores based on response length."""
        results = []

        for i, rollout in enumerate(rollouts_messages):
            # Find the assistant response
            assistant_response = ""
            for msg in rollout:
                if msg.role == "assistant":
                    assistant_response = msg.content
                    break

            # Score based on response length (normalized)
            length = len(assistant_response)
            score = min(1.0, length / 100.0)  # Normalize to max score of 1.0

            result = EvaluateResult(
                score=score,
                reason=f"Rollout {i}: Response length {length} characters",
                is_score_valid=True,
                metrics={
                    "length": {
                        "score": length,
                        "reason": f"Response has {length} characters",
                        "is_score_valid": True,
                    }
                },
            )
            results.append(result)

        return results

    return test_batch_reward


@pytest.mark.asyncio
async def test_end_to_end_n_variant_to_batch_evaluation(
    n_variant_pipeline_config, mock_reward_function, sample_batch_reward_function
):
    """
    Test the complete pipeline from N-variant generation to batch evaluation.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Paths for test files
        n_variant_output = temp_path / "n_variant_results.jsonl"
        batch_input = temp_path / "batch_input.jsonl"
        batch_output = temp_path / "batch_results.jsonl"

        # Update config to use temp directory
        n_variant_pipeline_config.output.results_file = str(n_variant_output)

        # --- Step 1: Generate N variants ---

        # Mock the model client to return different responses for each variant
        mock_model_client = Mock()
        variant_responses = [
            "Short answer",
            "A somewhat longer answer with more detail",
            "This is a very long and detailed answer that should score higher in the batch evaluation because it contains much more content and explanation.",
        ]

        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            response = variant_responses[call_count % len(variant_responses)]
            call_count += 1
            return GenerationResult(content=response, tool_calls=None, usage=None)

        mock_model_client.generate = AsyncMock(side_effect=mock_generate)
        mock_model_client.temperature = 0.0

        # Mock HTTP session
        mock_session = Mock()

        with patch(
            "eval_protocol.utils.module_loader.load_function",
            return_value=mock_reward_function,
        ):
            pipeline = EvaluationPipeline(n_variant_pipeline_config)
            pipeline.model_client = mock_model_client

            # Test samples (2 requests, each will generate 3 variants)
            test_samples = [
                {
                    "id": "sample_1",
                    "user_query": "What is 2+2?",
                    "ground_truth_for_eval": "4",
                },
                {
                    "id": "sample_2",
                    "user_query": "Explain photosynthesis",
                    "ground_truth_for_eval": "Process by which plants convert light to energy",
                },
            ]

            # Process samples
            all_results = []
            for i, sample in enumerate(test_samples):
                result = await pipeline._process_single_sample(
                    sample=sample, http_session=mock_session, original_index=i
                )

                # Should return list of 3 variants
                assert isinstance(result, list)
                assert len(result) == 3
                all_results.extend(result)

            # Write N-variant results to file
            with open(n_variant_output, "w") as f:
                for result in all_results:
                    f.write(json.dumps(result) + "\n")

        # Verify N-variant output format
        assert n_variant_output.exists()
        with open(n_variant_output, "r") as f:
            lines = f.readlines()
            assert len(lines) == 6  # 2 samples × 3 variants each

            # Check first few results have correct structure
            first_result = json.loads(lines[0])
            assert "request_id" in first_result
            assert "response_id" in first_result
            assert first_result["request_id"] == "sample_1"
            assert first_result["response_id"] in [0, 1, 2]

        # --- Step 2: Transform to batch format ---

        batch_data = transform_n_variant_jsonl_to_batch_format(
            input_file_path=str(n_variant_output), output_file_path=str(batch_input)
        )

        # Verify transformation
        assert len(batch_data) == 2  # 2 original requests
        assert batch_input.exists()

        for entry in batch_data:
            assert "request_id" in entry
            assert "rollouts_messages" in entry
            assert "num_variants" in entry
            assert entry["num_variants"] == 3
            assert len(entry["rollouts_messages"]) == 3

            # Verify rollouts_messages structure
            for rollout in entry["rollouts_messages"]:
                assert isinstance(rollout, list)
                assert len(rollout) >= 2  # At least user and assistant messages

                # Check message structure
                has_user = any(msg.get("role") == "user" for msg in rollout)
                has_assistant = any(msg.get("role") == "assistant" for msg in rollout)
                assert has_user and has_assistant

        # --- Step 3: Run batch evaluation ---

        # Save the sample batch reward function to a temporary module for testing
        batch_func_module = temp_path / "test_batch_func.py"
        with open(batch_func_module, "w") as f:
            f.write(
                '''
from eval_protocol.typed_interface import reward_function
from eval_protocol.models import EvaluateResult, Message
from typing import List

@reward_function(mode="batch")
def test_batch_reward(
    rollouts_messages: List[List[Message]],
    ground_truth_for_eval: str = None,
    **kwargs
) -> List[EvaluateResult]:
    """Test batch reward function that scores based on response length."""
    results = []

    for i, rollout in enumerate(rollouts_messages):
        # Find the assistant response
        assistant_response = ""
        for msg in rollout:
            if msg.role == "assistant":
                assistant_response = msg.content
                break

        # Score based on response length (normalized)
        length = len(assistant_response)
        score = min(1.0, length / 100.0)  # Normalize to max score of 1.0

        result = EvaluateResult(
            score=score,
            reason=f"Rollout {i}: Response length {length} characters",
            is_score_valid=True,
            metrics={}
        )
        results.append(result)

    return results
'''
            )

        # Add temp directory to Python path for import
        import sys

        sys.path.insert(0, str(temp_path))

        try:
            batch_results = run_batch_evaluation(
                batch_jsonl_path=str(batch_input),
                reward_function_path="test_batch_func.test_batch_reward",
                output_path=str(batch_output),
            )

            # Verify batch evaluation results
            assert len(batch_results) == 6  # 2 requests × 3 variants each
            assert batch_output.exists()

            # Group results by request_id
            results_by_request: dict[str, list[dict[str, Any]]] = {}
            for result in batch_results:
                request_id = result["request_id"]
                if request_id not in results_by_request:
                    results_by_request[request_id] = []
                results_by_request[request_id].append(result)

            assert len(results_by_request) == 2  # 2 original requests

            for request_id, request_results in results_by_request.items():
                assert len(request_results) == 3  # 3 variants per request

                # Verify result structure
                for result in request_results:
                    assert "request_id" in result
                    assert "response_id" in result
                    assert "rollout_index" in result
                    assert "evaluation_score" in result
                    assert "evaluation_reason" in result
                    assert isinstance(result["evaluation_score"], (int, float))
                    assert 0.0 <= result["evaluation_score"] <= 1.0

                # Verify scores are different (based on different response lengths)
                scores = [r["evaluation_score"] for r in request_results]
                assert len(set(scores)) > 1  # Should have different scores

        finally:
            # Clean up Python path
            sys.path.remove(str(temp_path))


def test_transformation_function_standalone():
    """Test the transformation function with mock data."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "test_input.jsonl"
        output_file = temp_path / "test_output.jsonl"

        # Create mock N-variant data
        mock_data = [
            {
                "id": "sample_1_v0",
                "request_id": "sample_1",
                "response_id": 0,
                "user_query": "Test question",
                "ground_truth_for_eval": "Test answer",
                "assistant_response": "Response 0",
                "full_conversation_history": [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Response 0"},
                ],
                "evaluation_score": 0.8,
            },
            {
                "id": "sample_1_v1",
                "request_id": "sample_1",
                "response_id": 1,
                "user_query": "Test question",
                "ground_truth_for_eval": "Test answer",
                "assistant_response": "Response 1",
                "full_conversation_history": [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Response 1"},
                ],
                "evaluation_score": 0.6,
            },
        ]

        # Write mock data
        with open(input_file, "w") as f:
            for item in mock_data:
                f.write(json.dumps(item) + "\n")

        # Transform
        result = transform_n_variant_jsonl_to_batch_format(
            input_file_path=str(input_file), output_file_path=str(output_file)
        )

        # Verify result
        assert len(result) == 1  # 1 request
        batch_entry = result[0]

        assert batch_entry["request_id"] == "sample_1"
        assert batch_entry["num_variants"] == 2
        assert len(batch_entry["rollouts_messages"]) == 2
        assert "user_query" in batch_entry
        assert "ground_truth_for_eval" in batch_entry

        # Verify file was written
        assert output_file.exists()
        with open(output_file, "r") as f:
            file_data = json.loads(f.read().strip())
            assert file_data == batch_entry


def test_batch_reward_function_example():
    """Test the sample batch reward function."""

    sample_func = create_sample_batch_reward_function()

    # Create test rollouts
    rollouts = [
        [
            Message(role="user", content="Test question"),
            Message(role="assistant", content="Short"),
        ],
        [
            Message(role="user", content="Test question"),
            Message(role="assistant", content="Much longer response with more detail"),
        ],
    ]

    results = sample_func(rollouts_messages=rollouts)

    assert len(results) == 2
    assert all(isinstance(r, EvaluateResult) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)

    # Longer response should have higher score
    assert results[1].score > results[0].score
