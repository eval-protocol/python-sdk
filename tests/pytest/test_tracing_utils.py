from eval_protocol.models import EvaluationRow, ExecutionMetadata, Message
from eval_protocol.pytest.tracing_utils import _merge_payloads_into_longest_row


def test_merge_payloads_into_longest_row_preserves_each_assistant_turn():
    first_turn_logprobs = {"content": [{"logprob": -0.1}, {"logprob": -0.2}]}
    second_turn_logprobs = {"content": [{"logprob": -0.3}]}
    first_turn = EvaluationRow(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4", logprobs=first_turn_logprobs),
        ],
        execution_metadata=ExecutionMetadata(
            extra={
                "completion_logprobs": [-0.1, -0.2],
                "routing_matrices": ["first-matrix"],
                "routing_metadata": {"total_token_count": 1},
            },
        ),
    )
    second_turn = EvaluationRow(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4"),
            Message(role="user", content="Use that in a sentence."),
            Message(role="assistant", content="4", logprobs=second_turn_logprobs),
        ],
        execution_metadata=ExecutionMetadata(
            extra={
                "completion_logprobs": [-0.3],
                "routing_matrices": ["second-matrix"],
                "routing_metadata": {"total_token_count": 1},
            },
        ),
    )

    _merge_payloads_into_longest_row(second_turn, [first_turn, second_turn])

    assistant_messages = second_turn.get_assistant_messages()
    assert assistant_messages[0].logprobs == first_turn_logprobs
    assert assistant_messages[1].logprobs == second_turn_logprobs
    assert second_turn.execution_metadata.extra is not None
    assert second_turn.execution_metadata.extra["routing_matrices"] == ["second-matrix"]
    assert second_turn.execution_metadata.extra["assistant_turn_payloads"] == [
        {
            "assistant_turn_index": 0,
            "completion_logprobs": [-0.1, -0.2],
            "routing_matrices": ["first-matrix"],
            "routing_metadata": {"total_token_count": 1},
        },
        {
            "assistant_turn_index": 1,
            "completion_logprobs": [-0.3],
            "routing_matrices": ["second-matrix"],
            "routing_metadata": {"total_token_count": 1},
        },
    ]
