"""Tests for OpenAIResponsesAdapter."""

from syrupy.assertion import SnapshotAssertion

from eval_protocol.adapters.openai_responses import OpenAIResponsesAdapter


def test_openai_responses_adapter_with_real_response_simple(snapshot: SnapshotAssertion):
    """Test OpenAIResponsesAdapter with a real response ID that is a simple 4
    message conversation with: system, user, tool, tool response, assistant.

    https://platform.openai.com/logs/resp_05639dcaca074fbc0068c9946593b481908cac70075926d85c
    """
    adapter = OpenAIResponsesAdapter()

    response_id = "resp_05639dcaca074fbc0068c9946593b481908cac70075926d85c"

    eval_rows = adapter.get_evaluation_rows(response_ids=[response_id])

    # Basic assertions about the returned data structure
    assert isinstance(eval_rows, list)
    assert len(eval_rows) == 1

    # Convert to dict for snapshot testing
    eval_rows_dict = [
        row.model_dump(exclude={"created_at": True, "execution_metadata": True, "messages": {"__all__": {"weight"}}})
        for row in eval_rows
    ]

    # Assert against snapshot
    assert eval_rows_dict == snapshot


def test_openai_responses_adapter_with_real_response_parallel_tool_calls(snapshot: SnapshotAssertion):
    """
    https://platform.openai.com/logs/resp_0e1b7db5d96e92470068c99506443c819e9305e92915d2405f
    """
    adapter = OpenAIResponsesAdapter()
    response_id = "resp_0e1b7db5d96e92470068c99506443c819e9305e92915d2405f"

    eval_rows = adapter.get_evaluation_rows(response_ids=[response_id])

    # Basic assertions about the returned data structure
    assert isinstance(eval_rows, list)
    assert len(eval_rows) == 1

    # Convert to dict for snapshot testing
    eval_rows_dict = [
        row.model_dump(exclude={"created_at": True, "execution_metadata": True, "messages": {"__all__": {"weight"}}})
        for row in eval_rows
    ]

    # Assert against snapshot
    assert eval_rows_dict == snapshot
