import json
from typing import Dict

import pytest

from eval_protocol.models import (  # Added Message to existing import
    CompletionParams,
    EvaluateResult,
    EvaluationRow,
    InputMetadata,
    Message,
    MetricResult,
    StepOutput,
)


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
    assert set(metric.keys()) == {"score", "reason", "is_score_valid"}

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
            completion_params=CompletionParams(model="gpt-4"),
            dataset_info={"source": "math_eval"},
            session_data={"timestamp": 1234567890},
        ),
    )

    assert len(row.messages) == 2
    assert row.ground_truth == "4"
    assert row.evaluation_result.score == 1.0
    assert row.get_input_metadata("row_id") == "math_001"
    assert not row.is_trajectory_evaluation()


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
            completion_params=CompletionParams(model="gpt-4"),
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
