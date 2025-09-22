import json
import logging
from typing import Dict

import pytest

from eval_protocol.models import (  # Added Message to existing import
    EvaluateResult,
    EvaluationRow,
    InputMetadata,
    Message,
    MetricResult,
    StepOutput,
)


def dummy_row() -> EvaluationRow:
    from eval_protocol.models import (
        EvaluateResult as _EvaluateResult,
        EvaluationRow as _EvaluationRow,
        InputMetadata as _InputMetadata,
        Message as _Message,
        MetricResult as _MetricResult,
    )

    msgs = [
        _Message(role="system", content="You are a helpful assistant"),
        _Message(role="user", content="Compute 2+2"),
        _Message(role="assistant", content="4"),
    ]
    eval_res = _EvaluateResult(
        score=1.0,
        reason="Correct",
        metrics={
            "accuracy": _MetricResult(score=1.0, reason="matches ground truth"),
        },
    )
    child_row = _EvaluationRow(
        messages=msgs,
        ground_truth="4",
        evaluation_result=eval_res,
        input_metadata=_InputMetadata(
            row_id="arith_0001",
            completion_params={"model": "dummy/local-model", "temperature": 0.0},
            dataset_info={"source": "unit_test", "variant": "subprocess"},
            session_data={"attempt": 1},
        ),
    )
    return child_row


def _child_compute_hash_value(_unused=None) -> int:
    row = dummy_row()
    return hash(row)


def test_metric_result_creation():
    """Test creating a MetricResult."""
    metric = MetricResult(score=0.5, reason="Test reason", is_score_valid=True)
    assert metric.score == 0.5
    assert metric.reason == "Test reason"
    assert metric.is_score_valid is True


def test_metric_result_serialization():
    """Test serializing MetricResult to JSON."""
    metric = MetricResult(score=0.75, reason="Test serialization", is_score_valid=True)
    json_str = metric.model_dump_json()
    data = json.loads(json_str)
    assert data["score"] == 0.75
    assert data["reason"] == "Test serialization"
    assert data["is_score_valid"] is True


def test_metric_result_deserialization():
    """Test deserializing MetricResult from JSON."""
    json_str = '{"score": 0.9, "reason": "Test deserialization", "is_score_valid": true}'
    metric = MetricResult.model_validate_json(json_str)
    assert metric.score == 0.9
    assert metric.reason == "Test deserialization"
    assert metric.is_score_valid is True


def test_evaluate_result_creation():
    """Test creating an EvaluateResult."""
    metrics: Dict[str, MetricResult] = {
        "metric1": MetricResult(score=0.5, reason="Reason 1", is_score_valid=True),
        "metric2": MetricResult(score=0.7, reason="Reason 2", is_score_valid=True),
    }
    result = EvaluateResult(score=0.6, reason="Overall assessment", metrics=metrics, is_score_valid=True)
    assert result.score == 0.6
    assert result.reason == "Overall assessment"
    assert len(result.metrics) == 2
    assert result.metrics["metric1"].score == 0.5
    assert result.metrics["metric2"].reason == "Reason 2"
    assert result.metrics["metric2"].is_score_valid is True
    assert result.is_score_valid is True


def test_evaluate_result_serialization():
    """Test serializing EvaluateResult to JSON."""
    metrics = {
        "metric1": MetricResult(score=0.5, reason="Reason 1", is_score_valid=True),
        "metric2": MetricResult(score=0.7, reason="Reason 2", is_score_valid=True),
    }
    result = EvaluateResult(score=0.6, reason="Overall assessment", metrics=metrics, is_score_valid=True)
    json_str = result.model_dump_json()
    data = json.loads(json_str)
    assert data["score"] == 0.6
    assert data["reason"] == "Overall assessment"
    assert len(data["metrics"]) == 2
    assert data["metrics"]["metric1"]["score"] == 0.5
    assert data["metrics"]["metric1"]["is_score_valid"] is True
    assert data["metrics"]["metric2"]["reason"] == "Reason 2"
    assert data["is_score_valid"] is True


def test_evaluate_result_deserialization():
    """Test deserializing EvaluateResult from JSON."""
    json_str = (
        '{"score": 0.8, "reason": "Overall", "metrics": {'
        '"metric1": {"score": 0.4, "reason": "Reason A", "is_score_valid": true}, '
        '"metric2": {"score": 0.9, "reason": "Reason B", "is_score_valid": true}'
        '}, "error": null, "is_score_valid": true}'
    )
    result = EvaluateResult.model_validate_json(json_str)
    assert result.score == 0.8
    assert result.reason == "Overall"
    assert len(result.metrics) == 2
    assert result.metrics["metric1"].score == 0.4
    assert result.metrics["metric1"].is_score_valid is True
    assert result.metrics["metric2"].reason == "Reason B"
    assert result.error is None
    assert result.is_score_valid is True


def test_empty_metrics_evaluate_result():
    """Test EvaluateResult with empty metrics dictionary."""
    result = EvaluateResult(score=1.0, reason="Perfect score", metrics={}, is_score_valid=True)
    assert result.score == 1.0
    assert result.reason == "Perfect score"
    assert result.metrics == {}
    assert result.is_score_valid is True

    json_str = result.model_dump_json()
    data = json.loads(json_str)
    assert data["score"] == 1.0
    assert data["reason"] == "Perfect score"
    assert data["metrics"] == {}
    assert data["is_score_valid"] is True


def test_metric_result_dict_access():
    """Test dictionary-style access for MetricResult."""
    metric = MetricResult(score=0.7, reason="Dict access test", is_score_valid=True)

    # __getitem__
    assert metric["score"] == 0.7
    assert metric["reason"] == "Dict access test"
    assert metric["is_score_valid"] is True
    with pytest.raises(KeyError):
        _ = metric["invalid_key"]

    # __contains__
    assert "score" in metric
    assert "reason" in metric
    assert "is_score_valid" in metric
    assert "invalid_key" not in metric

    # get()
    assert metric.get("score") == 0.7
    assert metric.get("reason") == "Dict access test"
    assert metric.get("is_score_valid") is True
    assert metric.get("invalid_key") is None
    assert metric.get("invalid_key", "default_val") == "default_val"

    # keys()
    assert set(metric.keys()) == {"score", "reason", "is_score_valid", "data"}

    # values() - order might not be guaranteed by model_fields, so check content
    # Pydantic model_fields preserves declaration order.
    expected_values = [
        True,
        0.7,
        "Dict access test",
    ]  # Based on current field order in model
    actual_values = list(metric.values())
    # To make it order-independent for this test, let's check presence
    assert metric.score in actual_values
    assert metric.reason in actual_values
    assert metric.is_score_valid in actual_values

    # items()
    expected_items = {
        ("score", 0.7),
        ("reason", "Dict access test"),
        ("is_score_valid", True),
    }
    assert set(metric.items()) == expected_items

    # __iter__
    assert set(list(metric)) == {"score", "reason", "is_score_valid"}


def test_evaluate_result_dict_access():
    """Test dictionary-style access for EvaluateResult."""
    metric1_obj = MetricResult(score=0.5, reason="Reason 1", is_score_valid=True)
    metrics_dict: Dict[str, MetricResult] = {
        "metric1": metric1_obj,
    }
    result = EvaluateResult(
        score=0.6,
        reason="Overall assessment",
        metrics=metrics_dict,
        error="Test Error",
        is_score_valid=False,
    )

    # __getitem__
    assert result["score"] == 0.6
    assert result["reason"] == "Overall assessment"
    assert result["error"] == "Test Error"
    assert result["metrics"] == metrics_dict  # Returns the dict of MetricResult objects
    assert result["metrics"]["metric1"] == metric1_obj
    assert result["metrics"]["metric1"]["score"] == 0.5  # Accessing MetricResult via __getitem__

    with pytest.raises(KeyError):
        _ = result["invalid_key"]
    with pytest.raises(KeyError):  # Accessing non-existent key in nested metric
        _ = result["metrics"]["metric1"]["invalid_sub_key"]

    # __contains__
    assert "score" in result
    assert "reason" in result
    assert "metrics" in result
    assert "error" in result
    assert "invalid_key" not in result

    # get()
    assert result.get("score") == 0.6
    assert result.get("invalid_key") is None
    assert result.get("invalid_key", "default_val") == "default_val"

    # keys()
    assert set(result.keys()) == {
        "score",
        "reason",
        "metrics",
        "error",
        "is_score_valid",
        "step_outputs",
        "trajectory_info",
        "final_control_plane_info",
        "agg_score",
        "standard_error",
    }

    # values() - check presence due to potential order variation of model_fields
    actual_values = list(result.values())
    assert result.score in actual_values
    assert result.reason in actual_values
    assert result.metrics in actual_values
    assert result.error in actual_values

    # items()
    # Note: result.metrics is a dict of MetricResult objects.
    # For exact item matching, we compare sorted lists of (key, value) tuples.
    expected_items_list = sorted(
        [
            ("score", 0.6),
            ("reason", "Overall assessment"),
            ("metrics", metrics_dict),
            ("error", "Test Error"),
            ("is_score_valid", False),
            ("step_outputs", None),
            ("trajectory_info", None),
            ("final_control_plane_info", None),
            ("agg_score", None),
            ("standard_error", None),
        ]
    )
    # result.items() returns a list of tuples, so convert to list then sort.
    actual_items_list = sorted(list(result.items()))
    print(actual_items_list)
    print(expected_items_list)
    assert actual_items_list == expected_items_list

    # __iter__
    assert set(list(result)) == {
        "score",
        "reason",
        "metrics",
        "error",
        "is_score_valid",
        "step_outputs",
        "trajectory_info",
        "final_control_plane_info",
        "agg_score",
        "standard_error",
    }


# Removed the redundant import from here


def test_evaluation_row_creation():
    """Test creating an EvaluationRow."""
    messages = [Message(role="user", content="What is 2+2?"), Message(role="assistant", content="2+2 equals 4.")]

    evaluation_result = EvaluateResult(
        score=1.0, reason="Correct answer", metrics={"accuracy": MetricResult(score=1.0, reason="Perfect")}
    )

    row = EvaluationRow(
        messages=messages,
        ground_truth="4",
        evaluation_result=evaluation_result,
        input_metadata=InputMetadata(
            row_id="math_001",
            completion_params={"model": "gpt-4"},
            dataset_info={"source": "math_eval"},
            session_data={"timestamp": 1234567890},
        ),
    )

    assert len(row.messages) == 2
    assert row.ground_truth == "4"
    assert row.evaluation_result.score == 1.0
    assert row.get_input_metadata("row_id") == "math_001"
    assert not row.is_trajectory_evaluation()


def test_stable_json():
    """Test the stable hash method."""
    row = EvaluationRow(
        messages=[Message(role="user", content="What is 2+2?"), Message(role="assistant", content="2+2 equals 4.")],
        ground_truth="4",
    )
    row2 = EvaluationRow(
        messages=[Message(role="user", content="What is 2+2?"), Message(role="assistant", content="2+2 equals 4.")],
        ground_truth="4",
    )
    stable_json = row._stable_json()
    stable_json2 = row2._stable_json()
    assert stable_json == stable_json2
    assert "created_at" not in stable_json
    assert "execution_metadata" not in stable_json


def test_evaluation_row_trajectory_evaluation():
    """Test EvaluationRow with trajectory evaluation."""
    messages = [
        Message(role="user", content="Start task"),
        Message(role="assistant", content="Step 1"),
        Message(role="user", content="Continue"),
        Message(role="assistant", content="Step 2"),
    ]

    step_outputs = [
        StepOutput(step_index=0, base_reward=0.3, terminated=False),
        StepOutput(step_index=1, base_reward=0.7, terminated=True),
    ]

    evaluation_result = EvaluateResult(score=0.5, reason="Task completed", step_outputs=step_outputs)

    row = EvaluationRow(
        messages=messages, ground_truth="Task completed successfully", evaluation_result=evaluation_result
    )

    assert row.is_trajectory_evaluation()
    assert row.ground_truth == "Task completed successfully"
    assert len(row.get_assistant_messages()) == 2
    assert len(row.get_user_messages()) == 2


def test_evaluation_row_serialization():
    """Test serializing EvaluationRow to JSON."""
    messages = [Message(role="user", content="Test question"), Message(role="assistant", content="Test answer")]

    evaluation_result = EvaluateResult(score=0.8, reason="Good response")

    row = EvaluationRow(
        messages=messages,
        ground_truth="Expected answer",
        evaluation_result=evaluation_result,
        input_metadata=InputMetadata(
            row_id="test_123",
            completion_params={"model": "gpt-4"},
            dataset_info={"test": True},
            session_data={"timestamp": 1234567890},
        ),
    )

    json_str = row.model_dump_json()
    data = json.loads(json_str)

    assert len(data["messages"]) == 2
    assert data["ground_truth"] == "Expected answer"
    assert data["evaluation_result"]["score"] == 0.8
    assert data["input_metadata"]["dataset_info"]["test"] is True
    assert data["input_metadata"]["row_id"] == "test_123"
    assert data["input_metadata"]["completion_params"]["model"] == "gpt-4"


def test_message_creation_requires_role():
    """Test that creating a Message requires the 'role' field."""
    from pydantic import ValidationError  # Ensure ValidationError is imported

    # Test direct instantiation
    with pytest.raises(ValidationError, match="Field required"):  # Pydantic's typical error for missing field
        Message(content="test content")

    # Test model_validate if it's intended to be a primary validation path
    # (though Pydantic's __init__ should catch it first)
    with pytest.raises(ValueError, match="Role is required"):
        Message.model_validate({"content": "test content"})

    # Test valid creation
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"

    msg_none_content = Message(role="user")  # content defaults to ""
    assert msg_none_content.role == "user"
    assert msg_none_content.content == ""


def test_stable_hash_consistency():
    """Test that the same EvaluationRow produces the same hash value consistently."""
    row1 = EvaluationRow(
        messages=[Message(role="user", content="What is 2+2?"), Message(role="assistant", content="2+2 equals 4.")],
        ground_truth="4",
    )
    row2 = EvaluationRow(
        messages=[Message(role="user", content="What is 2+2?"), Message(role="assistant", content="2+2 equals 4.")],
        ground_truth="4",
    )

    # Same content should produce same hash
    assert hash(row1) == hash(row2)

    # Hash should be consistent across multiple calls
    hash1_first = hash(row1)
    hash1_second = hash(row1)
    hash1_third = hash(row1)

    assert hash1_first == hash1_second == hash1_third

    # Hash should be a positive integer
    assert isinstance(hash1_first, int)
    assert hash1_first > 0


def test_stable_hash_different_content():
    """Test that different content produces different hash values."""
    row1 = EvaluationRow(
        messages=[Message(role="user", content="What is 2+2?"), Message(role="assistant", content="2+2 equals 4.")],
        ground_truth="4",
    )
    row2 = EvaluationRow(
        messages=[Message(role="user", content="What is 3+3?"), Message(role="assistant", content="3+3 equals 6.")],
        ground_truth="6",
    )

    # Different content should produce different hashes
    assert hash(row1) != hash(row2)


def test_stable_hash_ignores_volatile_fields():
    """Test that volatile fields like timestamps don't affect the hash."""
    messages = [Message(role="user", content="Test"), Message(role="assistant", content="Response")]

    # Create rows with different timestamps
    row1 = EvaluationRow(messages=messages, ground_truth="test")
    row2 = EvaluationRow(messages=messages, ground_truth="test")

    # Wait a moment to ensure different timestamps
    import time

    time.sleep(0.001)

    # Create another row
    row3 = EvaluationRow(messages=messages, ground_truth="test")

    # All should have the same hash despite different timestamps
    assert hash(row1) == hash(row2) == hash(row3)


def test_stable_hash_with_complex_data():
    """Test stable hashing with complex nested data structures."""
    complex_messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="Solve this math problem: 15 * 23"),
        Message(
            role="assistant",
            content="Let me solve this step by step:\n1. 15 * 20 = 300\n2. 15 * 3 = 45\n3. 300 + 45 = 345",
        ),
        Message(role="user", content="Thank you!"),
        Message(role="assistant", content="You're welcome! Let me know if you need help with anything else."),
    ]

    complex_evaluation = EvaluateResult(
        score=0.95,
        reason="Excellent step-by-step solution with clear explanation",
        metrics={
            "accuracy": MetricResult(score=1.0, reason="Correct mathematical calculation"),
            "explanation_quality": MetricResult(score=0.9, reason="Clear step-by-step breakdown"),
            "completeness": MetricResult(score=0.95, reason="Covers all aspects of the problem"),
        },
    )

    row1 = EvaluationRow(
        messages=complex_messages,
        ground_truth="345",
        evaluation_result=complex_evaluation,
        input_metadata=InputMetadata(
            row_id="complex_math_001",
            completion_params={"model": "gpt-4", "temperature": 0.1},
            dataset_info={"source": "math_eval", "difficulty": "medium"},
            session_data={"user_id": "test_user", "session_id": "session_123"},
        ),
    )

    row2 = EvaluationRow(
        messages=complex_messages,
        ground_truth="345",
        evaluation_result=complex_evaluation,
        input_metadata=InputMetadata(
            row_id="complex_math_001",
            completion_params={"model": "gpt-4", "temperature": 0.1},
            dataset_info={"source": "math_eval", "difficulty": "medium"},
            session_data={"user_id": "test_user", "session_id": "session_123"},
        ),
    )

    # Complex data should still produce consistent hashes
    assert hash(row1) == hash(row2)

    # Hash should be different from simple rows
    simple_row = EvaluationRow(
        messages=[Message(role="user", content="Simple"), Message(role="assistant", content="Response")],
        ground_truth="test",
    )
    assert hash(row1) != hash(simple_row)


def test_stable_hash_json_representation():
    """Test that the stable JSON representation is consistent and excludes volatile fields."""
    row = EvaluationRow(
        messages=[Message(role="user", content="Test"), Message(role="assistant", content="Response")],
        ground_truth="test",
    )

    # Get the stable JSON representation
    stable_json = row._stable_json()

    # Should be a valid JSON string
    parsed = json.loads(stable_json)

    # Should contain the core data
    assert "messages" in parsed
    assert "ground_truth" in parsed
    assert parsed["ground_truth"] == "test"

    # Should NOT contain volatile fields
    assert "created_at" not in parsed
    assert "execution_metadata" not in parsed

    # Should be deterministic (same content produces same JSON)
    stable_json2 = row._stable_json()
    assert stable_json == stable_json2


def test_stable_hash_consistency_for_identical_rows():
    """Test that identical EvaluationRow objects produce the same stable hash.

    This simulates the behavior expected across Python process restarts by
    creating multiple identical objects and ensuring their hashes match.
    """
    # Create a complex evaluation row
    messages = [
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="The capital of France is Paris."),
        Message(role="user", content="What about Germany?"),
        Message(role="assistant", content="The capital of Germany is Berlin."),
    ]

    evaluation_result = EvaluateResult(
        score=0.9,
        reason="Correct answers for both questions",
        metrics={
            "geography_knowledge": MetricResult(score=1.0, reason="Both capitals correctly identified"),
            "response_quality": MetricResult(score=0.8, reason="Clear and concise responses"),
        },
    )

    # Create multiple identical rows
    rows = []
    for i in range(5):
        row = EvaluationRow(
            messages=messages,
            ground_truth="Paris, Berlin",
            evaluation_result=evaluation_result,
            input_metadata=InputMetadata(
                completion_params={"model": "gpt-4"},
                dataset_info={"source": "geography_eval"},
            ),
        )
        rows.append(row)

    # All rows should have identical hashes
    first_hash = hash(rows[0])
    for row in rows[1:]:
        assert hash(row) == first_hash

    # The hash should be a large positive integer (SHA-256 first 8 bytes)
    assert first_hash > 0
    assert first_hash < 2**64  # 8 bytes = 64 bits


def test_stable_hash_edge_cases():
    """Test stable hashing with edge cases like empty data and None values."""
    # Empty messages
    empty_row = EvaluationRow(messages=[], ground_truth="")
    empty_hash = hash(empty_row)
    assert isinstance(empty_hash, int)
    assert empty_hash > 0

    # None values in optional fields
    none_row = EvaluationRow(
        messages=[Message(role="user", content="Test")], ground_truth=None, evaluation_result=None
    )
    none_hash = hash(none_row)
    assert isinstance(none_hash, int)
    assert none_hash > 0

    # Different from empty row
    assert empty_hash != none_hash

    # Row with only required fields
    minimal_row = EvaluationRow(messages=[Message(role="user", content="Minimal")])
    minimal_hash = hash(minimal_row)
    assert isinstance(minimal_hash, int)
    assert minimal_hash > 0

    # Should be different from other edge cases
    assert minimal_hash != empty_hash
    assert minimal_hash != none_hash


def test_stable_hash_across_subprocess():
    """Verify the same EvaluationRow produces the same hash in a separate Python process."""
    import multiprocessing as mp

    row = dummy_row()
    parent_hash = hash(row)
    # Compute the same hash in a fresh interpreter via Pool.map (spawned process)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        [child_hash] = pool.map(_child_compute_hash_value, [None])

    assert isinstance(child_hash, int)
    assert parent_hash == child_hash


def test_evaluation_row_extra_fields():
    example = {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        "ground_truth": "Paris",
        "evaluation_result": {"score": 1.0, "reason": "Correct"},
        "input_metadata": {"model": "gpt-4"},
        "eval": {"score": 0.5},
        "eval_details": {
            "score": 0.5,
            "reason": "Correct",
            "is_score_valid": True,
            "metrics": {
                "accuracy": {
                    "score": 1.0,
                    "reason": "Correct",
                    "is_score_valid": True,
                },
            },
        },
        "extra_fields": {
            "test": "test",
        },
    }
    row = EvaluationRow(**example)
    dictionary = json.loads(row.model_dump_json())
    assert "eval" in dictionary
    assert "accuracy" in dictionary["eval_details"]["metrics"]
    assert "test" in dictionary["extra_fields"]


def test_message_with_weight_dump():
    example = {
        "role": "user",
        "content": "Hello, how are you?",
        "weight": 0,
    }

    message = Message(**example)
    dictionary = message.model_dump()
    assert "weight" in dictionary
    assert dictionary["weight"] == 0


def test_message_dump_for_chat_completion_request():
    example = {
        "role": "user",
        "content": "Hello, how are you?",
        "weight": 0,
        "reasoning_content": "I am thinking about the user's question",
    }
    message = Message(**example)
    dictionary = message.dump_mdoel_for_chat_completion_request()
    assert "weight" not in dictionary
    assert "reasoning_content" not in dictionary
    assert dictionary["content"] == "Hello, how are you?"
