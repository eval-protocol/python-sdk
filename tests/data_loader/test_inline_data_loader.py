from eval_protocol.data_loader.inline_data_loader import InlineDataLoader
from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor


@evaluation_test(
    data_loaders=InlineDataLoader(
        messages=[[Message(role="user", content="What is 2 + 2?")]],
    ),
)
def test_inline_data_loader(row: EvaluationRow) -> EvaluationRow:
    """Inline data loader should feed pre-constructed message bundles."""

    assert row.messages[0].content == "What is 2 + 2?"
    assert row.input_metadata.dataset_info is not None
    assert row.input_metadata.dataset_info.get("data_loader_variant_id") == "inline"
    assert row.input_metadata.dataset_info.get("data_loader_num_rows") == 1
    assert row.input_metadata.dataset_info.get("data_loader_num_rows_after_preprocessing") == 1
    assert row.input_metadata.dataset_info.get("data_loader_type") == "InlineDataLoader"
    assert row.input_metadata.dataset_info.get("data_loader_variant_description") is None
    assert row.input_metadata.dataset_info.get("data_loader_preprocessed") is False
    row.evaluation_result = EvaluateResult(score=0.0, reason="Dummy evaluation result")
    return row


@evaluation_test(
    data_loaders=InlineDataLoader(
        messages=[[Message(role="user", content=f"What is {i} + {i}?")] for i in range(5)],
    ),
    max_dataset_rows=2,
)
def test_inline_data_loader_max_dataset_rows(row: EvaluationRow) -> EvaluationRow:
    """Inline data loader should respect max_dataset_rows parameter."""

    # This test should only process 2 rows despite the loader having 5
    content = row.messages[0].content
    assert content in ["What is 0 + 0?", "What is 1 + 1?"]

    assert row.input_metadata.dataset_info is not None
    assert row.input_metadata.dataset_info.get("data_loader_variant_id") == "inline"
    assert row.input_metadata.dataset_info.get("data_loader_type") == "InlineDataLoader"
    assert row.input_metadata.dataset_info.get("data_loader_preprocessed") is False

    row.evaluation_result = EvaluateResult(score=0.0, reason="Dummy evaluation result")
    return row
