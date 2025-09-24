from eval_protocol.data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test


def my_factory() -> list[EvaluationRow]:
    """Factory function that generates evaluation rows dynamically."""
    return [EvaluationRow(messages=[Message(role="user", content="What is 2 + 2?")])]


@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[my_factory],
    ),
)
def test_dynamic_data_loader(row: EvaluationRow) -> EvaluationRow:
    """Dynamic data loader should feed dynamically generated message bundles."""

    assert row.messages[0].content == "What is 2 + 2?"
    assert row.input_metadata.dataset_info is not None
    assert row.input_metadata.dataset_info.get("data_loader_variant_id") == "my_factory"
    assert row.input_metadata.dataset_info.get("data_loader_num_rows") == 1
    assert row.input_metadata.dataset_info.get("data_loader_num_rows_after_preprocessing") == 1
    assert row.input_metadata.dataset_info.get("data_loader_type") == "DynamicDataLoader"
    assert (
        row.input_metadata.dataset_info.get("data_loader_variant_description")
        == "Factory function that generates evaluation rows dynamically."
    )
    assert row.input_metadata.dataset_info.get("data_loader_preprocessed") is False
    return row


@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[lambda: [EvaluationRow(messages=[Message(role="user", content="What is 3 * 3?")])]],
    ),
)
def test_dynamic_data_loader_lambda(row: EvaluationRow) -> EvaluationRow:
    """Dynamic data loader should work with lambda functions."""

    assert row.messages[0].content == "What is 3 * 3?"
    assert row.input_metadata.dataset_info is not None
    assert row.input_metadata.dataset_info.get("data_loader_variant_id") == "<lambda>"
    assert row.input_metadata.dataset_info.get("data_loader_num_rows") == 1
    assert row.input_metadata.dataset_info.get("data_loader_num_rows_after_preprocessing") == 1
    assert row.input_metadata.dataset_info.get("data_loader_type") == "DynamicDataLoader"
    assert row.input_metadata.dataset_info.get("data_loader_preprocessed") is False
    return row
