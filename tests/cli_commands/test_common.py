import importlib
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from eval_protocol.cli_commands import common

try:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

    class _DummyDSType:
        pass

    Dataset = IterableDataset = DatasetDict = IterableDatasetDict = _DummyDSType


@pytest.fixture(autouse=True)
def manage_pyarrow_registration(monkeypatch):
    if DATASETS_AVAILABLE:
        try:
            import pyarrow

            if hasattr(pyarrow, "register_extension_type"):
                _original_register_extension_type = pyarrow.register_extension_type

                def _idempotent_register_extension_type(ext_type):
                    try:
                        _original_register_extension_type(ext_type)
                    except pyarrow.lib.ArrowKeyError as e:
                        if "already defined" in str(e).lower():
                            pass
                        else:
                            raise

                monkeypatch.setattr(
                    pyarrow,
                    "register_extension_type",
                    _idempotent_register_extension_type,
                )
        except ImportError:
            pass
        except Exception:
            pass


@pytest.fixture
def sample_jsonl_content_valid():
    return [
        {
            "messages": [
                {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "A1"},
            ],
            "ground_truth": "GT1",
            "id": "s1",
        },
        {
            "messages": [
                {"role": "user", "content": "U2"},
                {"role": "assistant", "content": "A2"},
            ],
            "id": "s2",
        },
        {
            "messages": [
                {"role": "user", "content": "U3"},
                {"role": "assistant", "content": "A3"},
            ]
        },
    ]


def create_jsonl_file_helper(tmp_path: Path, content_list: list) -> Path:
    file_path = tmp_path / "samples.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in content_list:
            f.write(json.dumps(item) + "\n")
    return file_path


def test_load_samples_from_file_valid(tmp_path, sample_jsonl_content_valid):
    file_path = create_jsonl_file_helper(tmp_path, sample_jsonl_content_valid)
    samples = list(common.load_samples_from_file(str(file_path), max_samples=10))
    assert len(samples) == 3
    assert samples[0]["id"] == "s1"


def test_load_samples_from_file_max_samples(tmp_path, sample_jsonl_content_valid):
    file_path = create_jsonl_file_helper(tmp_path, sample_jsonl_content_valid)
    samples = list(common.load_samples_from_file(str(file_path), max_samples=2))
    assert len(samples) == 2


def test_load_samples_from_file_file_not_found(caplog):
    with caplog.at_level(logging.ERROR):
        samples = list(
            common.load_samples_from_file("non_existent_file.jsonl", max_samples=10)
        )
    assert len(samples) == 0
    assert "Sample file not found: non_existent_file.jsonl" in caplog.text


def test_load_samples_from_file_invalid_json(tmp_path, caplog):
    file_path = tmp_path / "invalid.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [{"role": "user", "content": "Good"}]}\n')
        f.write("this is not json\n")
        f.write('{"messages": [{"role": "user", "content": "Good again"}]}\n')
    with caplog.at_level(logging.WARNING):
        samples = list(common.load_samples_from_file(str(file_path), max_samples=10))
    assert len(samples) == 2
    assert "Invalid JSON. Skipping line: this is not json..." in caplog.text


def test_load_samples_from_file_empty_file(tmp_path):
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("")
    with patch("eval_protocol.cli_commands.common.logger.info") as mock_info:
        samples = list(common.load_samples_from_file(str(file_path), max_samples=10))
        assert len(samples) == 0
        mock_info.assert_called_once_with(
            f"No valid samples loaded from {str(file_path)} after processing 0 lines."
        )


def test_load_samples_from_file_skip_empty_lines(tmp_path):
    file_path = tmp_path / "samples_with_empty.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [{"role": "user", "content": "U1"}]}\n')
        f.write("\n")
        f.write('{"messages": [{"role": "user", "content": "U2"}]}\n')
    samples = list(common.load_samples_from_file(str(file_path), max_samples=10))
    assert len(samples) == 2


def test_validate_sample_messages_valid():
    messages = [{"role": "user", "content": "Hello"}]
    assert common._validate_sample_messages(messages, 1, 1) is True


def test_validate_sample_messages_not_a_list(caplog):
    with caplog.at_level(logging.WARNING):
        assert common._validate_sample_messages("not a list", 1, 1) is False
    assert "'messages' field is not a list" in caplog.text


def test_validate_sample_messages_empty_list(caplog):
    with caplog.at_level(logging.WARNING):
        assert common._validate_sample_messages([], 1, 1) is False
    assert "'messages' list is empty" in caplog.text


def test_validate_sample_messages_item_not_a_dict(caplog):
    messages = ["not a dict"]
    with caplog.at_level(logging.WARNING):
        assert common._validate_sample_messages(messages, 1, 1) is False
    assert "message item 0 is not a dictionary" in caplog.text


def test_validate_sample_messages_missing_role(caplog):
    messages = [{"content": "Hello"}]
    with caplog.at_level(logging.WARNING):
        assert common._validate_sample_messages(messages, 1, 1) is False
    assert "missing 'role' or 'content' string fields" in caplog.text


def test_validate_sample_messages_missing_content(caplog):
    messages = [{"role": "user"}]
    with caplog.at_level(logging.WARNING):
        assert common._validate_sample_messages(messages, 1, 1) is False
    assert "missing 'role' or 'content' string fields" in caplog.text


def test_validate_sample_messages_role_not_string(caplog):
    messages = [{"role": 123, "content": "Hello"}]
    with caplog.at_level(logging.WARNING):
        assert common._validate_sample_messages(messages, 1, 1) is False
    assert "missing 'role' or 'content' string fields" in caplog.text


@pytest.fixture
def mock_hf_dataset_data():
    return [
        {
            "prompt_col": "HF_U1",
            "response_col": "HF_A1",
            "id_col": "hf1",
            "gt_col": "HF_GT1",
        },
        {"prompt_col": "HF_U2", "response_col": "HF_A2"},
        {"prompt_col": "HF_U3", "response_col": 123},
        {"prompt_col": None, "response_col": "HF_A4"},
    ]


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets library not installed")
@patch("datasets.load_dataset")
def test_load_samples_from_hf_valid(mock_load_dataset, mock_hf_dataset_data):
    mock_iterable_ds = MagicMock(spec=IterableDataset)
    mock_iterable_ds.__iter__.return_value = iter(mock_hf_dataset_data)
    mock_load_dataset.return_value = mock_iterable_ds
    samples = list(
        common.load_samples_from_huggingface(
            "dummy/dataset",
            "train",
            "prompt_col",
            "response_col",
            {"id_col": "sample_id", "gt_col": "ground_truth"},
            10,
        )
    )
    assert len(samples) == 2
    assert samples[0]["messages"][0]["content"] == "HF_U1"
    assert samples[0]["sample_id"] == "hf1"
    assert samples[1]["messages"][0]["content"] == "HF_U2"


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets library not installed")
@patch("datasets.load_dataset")
def test_load_samples_from_hf_max_samples(mock_load_dataset, mock_hf_dataset_data):
    mock_iterable_ds = MagicMock(spec=IterableDataset)
    mock_iterable_ds.__iter__.return_value = iter(mock_hf_dataset_data)
    mock_load_dataset.return_value = mock_iterable_ds
    samples = list(
        common.load_samples_from_huggingface(
            "d", "s", "prompt_col", "response_col", None, 1
        )
    )
    assert len(samples) == 1


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets library not installed")
@patch("datasets.load_dataset")
def test_load_samples_from_hf_key_not_found_in_record(mock_load_dataset, caplog):
    mock_iterable_ds = MagicMock(spec=IterableDataset)
    mock_iterable_ds.__iter__.return_value = iter(
        [{"prompt_col": "P1", "response_col": "R1"}]
    )
    mock_load_dataset.return_value = mock_iterable_ds
    with caplog.at_level(logging.WARNING):
        samples = list(
            common.load_samples_from_huggingface(
                "d",
                "s",
                "prompt_col",
                "response_col",
                {"id_col_missing": "sample_id"},
                1,
            )
        )
    assert len(samples) == 1
    assert "Key 'id_col_missing' from key_map not found" in caplog.text


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets library not installed")
@patch(
    "datasets.load_dataset",
    side_effect=ImportError("Simulated ImportError for load_dataset"),
)
@patch(
    "datasets.DatasetDict",
    side_effect=ImportError("Simulated ImportError for DatasetDict"),
)
@patch("datasets.Dataset", side_effect=ImportError("Simulated ImportError for Dataset"))
@patch(
    "datasets.IterableDatasetDict",
    side_effect=ImportError("Simulated ImportError for IterableDatasetDict"),
)
@patch(
    "datasets.IterableDataset",
    side_effect=ImportError("Simulated ImportError for IterableDataset"),
)
def test_load_samples_from_hf_import_component_error(
    mock_ds_load,
    mock_ds_dict,
    mock_ds_dataset,
    mock_ds_iter_dict,
    mock_ds_iter_dataset,
    caplog,
):
    """
    Tests that if importing any component from 'datasets' fails, the ImportError is caught.
    This relies on the manage_pyarrow_registration fixture to prevent ArrowKeyError from patching.
    """
    with caplog.at_level(logging.ERROR):
        samples = list(
            common.load_samples_from_huggingface("d", "s", "p", "r", None, 1)
        )

    assert len(samples) == 0
    # This scenario (mocked load_dataset raising ImportError) is caught by the generic Exception handler
    # around the load_dataset call, not the initial 'from datasets import...' ImportError.
    assert (
        "Error loading HuggingFace dataset 'd' (split: s): Simulated ImportError for load_dataset"
        in caplog.text
    )
    # mock_ds_iter_dataset is the mock for 'datasets.load_dataset' due to decorator order.
    assert mock_ds_iter_dataset.called  # Check that the patched load_dataset was called


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets library not installed")
@patch("datasets.load_dataset")
def test_load_samples_from_hf_dataset_load_exception(mock_load_dataset, caplog):
    mock_load_dataset.side_effect = Exception("Network error or dataset not found")
    with caplog.at_level(logging.ERROR):
        samples = list(
            common.load_samples_from_huggingface("d", "s", "p", "r", None, 1)
        )
    assert len(samples) == 0
    assert "Error loading HuggingFace dataset" in caplog.text
