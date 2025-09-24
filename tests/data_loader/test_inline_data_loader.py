from eval_protocol.data_loader.inline_data_loader import InlineDataLoader
from eval_protocol.models import EvaluationRow, Message
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
    return row
