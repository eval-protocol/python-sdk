from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.parameterize import DefaultParameterIdGenerator, pytest_parametrize
from eval_protocol.pytest.generate_parameter_combinations import generate_parameter_combinations


def test_parameterized_ids():
    """Test that evaluation_test generates proper parameter IDs."""
    collected_ids = []

    @evaluation_test(
        input_messages=[[[Message(role="user", content="Hello, how are you?")]]],
        completion_params=[
            {"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"},
            {"model": "gpt-4"},
            {"temperature": 0.5},  # No model - should not generate ID
        ],
    )
    def test_parameterized_ids(row: EvaluationRow) -> EvaluationRow:
        # Collect the row to verify it was processed
        collected_ids.append(row.input_metadata.row_id)
        return row

    # The function should exist and be callable
    assert test_parameterized_ids is not None
    assert callable(test_parameterized_ids)

    # Test that the decorator was applied (function should have pytest marks)
    import pytest

    marks = getattr(test_parameterized_ids, "pytestmark", [])
    assert len(marks) > 0, "Function should have pytest marks from evaluation_test decorator"

    # Verify it's a parametrize mark
    parametrize_marks = [mark for mark in marks if hasattr(mark, "name") and mark.name == "parametrize"]
    assert len(parametrize_marks) > 0, "Should have parametrize mark"

    # Check that the parametrize mark has IDs
    parametrize_mark = parametrize_marks[0]
    assert hasattr(parametrize_mark, "kwargs"), "Parametrize mark should have kwargs"
    assert "ids" in parametrize_mark.kwargs, "Should have ids in kwargs"

    # Extract the IDs from the parametrize mark
    ids = parametrize_mark.kwargs.get("ids")
    if ids is not None:
        # Should have IDs for all parameters that have string/numeric values
        expected_ids = ["fireworks_ai/accounts/fireworks/models/gpt-oss-120b", "gpt-4", "0.5"]
        assert list(ids) == expected_ids, f"Expected {expected_ids}, got {list(ids)}"


def test_default_id_generator():
    """Test the DefaultParameterIdGenerator with various parameter combinations."""
    generator = DefaultParameterIdGenerator()

    # Test with full model path
    combo1 = (None, {"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}, None, None, None)
    id1 = generator.generate_id(combo1)
    assert id1 == "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"

    # Test with simple model name
    combo2 = (None, {"model": "gpt-4"}, None, None, None)
    id2 = generator.generate_id(combo2)
    assert id2 == "gpt-4"

    # Test with multiple string parameters
    combo3 = (None, {"model": "gpt-4", "stream": "true", "temperature": "0.7"}, None, None, None)
    id3 = generator.generate_id(combo3)
    assert id3 == "gpt-4:true:0.7"

    # Test with mixed string and numeric parameters
    combo4 = (None, {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}, None, None, None)
    id4 = generator.generate_id(combo4)
    assert id4 == "100:gpt-4:0.7"  # Keys are sorted alphabetically: max_tokens, model, temperature

    # Test with only numeric values
    combo5 = (None, {"temperature": 0.5, "max_tokens": 100}, None, None, None)
    id5 = generator.generate_id(combo5)
    assert id5 == "100:0.5"  # Keys are sorted alphabetically: max_tokens, temperature

    # Test with boolean values
    combo6 = (None, {"stream": True, "echo": False}, None, None, None)
    id6 = generator.generate_id(combo6)
    assert id6 == "False:True"  # Keys are sorted alphabetically: echo, stream

    # Test with mixed string, numeric, and boolean values
    combo7 = (None, {"model": "gpt-4", "temperature": 0.7, "stream": True}, None, None, None)
    id7 = generator.generate_id(combo7)
    assert id7 == "gpt-4:True:0.7"  # Keys are sorted alphabetically: model, stream, temperature

    # Test with no supported values (only non-supported types like lists, dicts)
    combo8 = (None, {"messages": [{"role": "user"}], "config": {"key": "value"}}, None, None, None)
    id8 = generator.generate_id(combo8)
    assert id8 is None

    # Test with None completion_params
    combo9 = (None, None, None, None, None)
    id9 = generator.generate_id(combo9)
    assert id9 is None


def test_pytest_parametrize_with_custom_id_generator():
    """Test pytest_parametrize with a custom ID generator."""

    # Create test combinations
    combinations = [
        (None, {"model": "gpt-4"}, None, None, None),
        (None, {"model": "claude-3"}, None, None, None),
        (None, {"temperature": 0.5}, None, None, None),  # Only numeric values
    ]

    # Test with default generator
    result = pytest_parametrize(
        combinations=combinations,
        input_dataset=None,
        completion_params=[{"model": "gpt-4"}, {"model": "claude-3"}, {"temperature": 0.5}],
        input_messages=None,
        input_rows=None,
        evaluation_test_kwargs=None,
    )

    assert result["argnames"] == ["completion_params"]
    assert len(list(result["argvalues"])) == 3
    assert result["ids"] == ["gpt-4", "claude-3", "0.5"]  # All have string/numeric values


def test_id_generator_max_length():
    """Test that ID generator respects max_length parameter."""
    generator = DefaultParameterIdGenerator(max_length=10)

    # Test with long model name
    combo = (None, {"model": "very-long-model-name-that-exceeds-max-length"}, None, None, None)
    id_str = generator.generate_id(combo)
    assert id_str == "very-lo..."
    assert len(id_str) <= 10
