import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from eval_protocol.cli_commands import preview as preview_cmd_module
from eval_protocol.cli_commands.preview import preview_command
from eval_protocol.generic_server import EvaluationRequest
from eval_protocol.models import EvaluateResult, Message, MetricResult

try:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

    class _DummyDSType:
        pass

    IterableDataset = _DummyDSType


class MockArgs:
    def __init__(self, **kwargs):
        self.verbose = False
        self.metrics_folders = None
        self.samples = None
        self.max_samples = 5
        self.huggingface_dataset = None
        self.huggingface_split = "train"
        self.huggingface_prompt_key = "prompt"
        self.huggingface_response_key = "response"
        self.huggingface_key_map = None
        self.remote_url = None
        self.__dict__.update(kwargs)


@pytest.fixture
def mock_check_environment():
    with patch(
        "eval_protocol.cli_commands.preview.check_environment", return_value=True
    ) as mock_check:
        yield mock_check


def create_temp_jsonl(tmp_path: Path, samples_data: list) -> str:
    sample_file_path = tmp_path / "temp_samples.jsonl"
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for sample in samples_data:
            f.write(json.dumps(sample) + "\n")
    return str(sample_file_path)


class TestPreviewCommandRemoteUrl:

    @patch("requests.post")
    def test_preview_remote_url_success_with_file(
        self, mock_post, mock_check_environment, tmp_path, capsys
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        sample_data_for_file = [
            {
                "messages": [
                    {"role": "user", "content": "User prompt 1"},
                    {"role": "assistant", "content": "Assistant response 1"},
                ],
                "ground_truth": "GT 1",
                "custom_kwarg": "custom_val_1",
            }
        ]
        temp_sample_file = create_temp_jsonl(tmp_path, sample_data_for_file)

        eval_result_payload = EvaluateResult(
            score=0.8,
            reason="Remote success",
            is_score_valid=True,
            metrics={
                "accuracy": MetricResult(
                    score=0.9, reason="High acc", is_score_valid=True
                )
            },  # This already has metrics
        ).model_dump()
        mock_response.json.return_value = eval_result_payload
        mock_post.return_value = mock_response

        args = MockArgs(
            remote_url="http://fake-remote-eval.com",
            samples=temp_sample_file,
            max_samples=1,
        )
        return_code = preview_command(args)
        assert return_code == 0

        expected_endpoint = "http://fake-remote-eval.com/evaluate"
        expected_payload_sample1 = EvaluationRequest(
            messages=sample_data_for_file[0]["messages"],
            ground_truth=sample_data_for_file[0]["ground_truth"],
            kwargs={"custom_kwarg": sample_data_for_file[0]["custom_kwarg"]},
        ).model_dump()
        mock_post.assert_called_once_with(
            expected_endpoint, json=expected_payload_sample1, timeout=30
        )

        captured = capsys.readouterr()
        assert (
            "Previewing against remote URL: http://fake-remote-eval.com" in captured.out
        )
        assert "--- Sample 1 ---" in captured.out
        assert "Score: 0.8" in captured.out

    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets library not installed")
    @patch("datasets.load_dataset")
    @patch("requests.post")
    def test_preview_remote_url_success_with_hf(
        self, mock_post, mock_hf_load_dataset, mock_check_environment, capsys
    ):
        hf_sample_data = [
            {
                "prompt": "HF User prompt",
                "response": "HF Assistant response",
                "ground_truth_col": "HF GT",
            }
        ]
        mock_iterable_ds = MagicMock(spec=IterableDataset)
        mock_iterable_ds.__iter__.return_value = iter(hf_sample_data)
        mock_hf_load_dataset.return_value = mock_iterable_ds

        mock_response = MagicMock()
        mock_response.status_code = 200
        # Corrected: Explicitly provide metrics={}
        eval_result_payload = EvaluateResult(
            score=0.7, reason="HF Remote success", metrics={}
        ).model_dump()
        mock_response.json.return_value = eval_result_payload
        mock_post.return_value = mock_response

        args = MockArgs(
            remote_url="http://fake-hf-eval.com",
            huggingface_dataset="test/hf-dataset",
            huggingface_prompt_key="prompt",
            huggingface_response_key="response",
            huggingface_key_map=json.dumps({"ground_truth_col": "ground_truth"}),
            max_samples=1,
        )
        return_code = preview_command(args)
        assert return_code == 0

        expected_payload = EvaluationRequest(
            messages=[
                {"role": "user", "content": "HF User prompt"},
                {"role": "assistant", "content": "HF Assistant response"},
            ],
            ground_truth="HF GT",
            kwargs={},
        ).model_dump()
        mock_post.assert_called_once_with(
            "http://fake-hf-eval.com/evaluate", json=expected_payload, timeout=30
        )
        captured = capsys.readouterr()
        assert "Score: 0.7" in captured.out

    @patch("requests.post")
    def test_preview_remote_url_http_error(
        self, mock_post, mock_check_environment, tmp_path, capsys
    ):
        sample_data = [{"messages": [{"role": "user", "content": "Test"}]}]
        temp_sample_file = create_temp_jsonl(tmp_path, sample_data)
        mock_post.side_effect = requests.exceptions.HTTPError(
            "403 Client Error: Forbidden for url"
        )

        args = MockArgs(
            remote_url="http://fake-remote-eval.com",
            samples=temp_sample_file,
            max_samples=1,
        )
        return_code = preview_command(args)
        assert return_code == 0

        captured = capsys.readouterr()
        assert "Error calling remote URL" in captured.out
        assert "403 Client Error: Forbidden for url" in captured.out

    def test_preview_remote_url_invalid_url_format(
        self, mock_check_environment, tmp_path, capsys
    ):
        sample_data = [{"messages": [{"role": "user", "content": "Test"}]}]
        temp_sample_file = create_temp_jsonl(tmp_path, sample_data)
        args = MockArgs(remote_url="ftp://invalid-url.com", samples=temp_sample_file)
        return_code = preview_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error: Invalid --remote-url 'ftp://invalid-url.com'" in captured.out

    def test_preview_remote_url_no_samples_provided(
        self, mock_check_environment, capsys
    ):
        args = MockArgs(remote_url="http://fake-remote-eval.com")
        return_code = preview_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "Error: Either sample file (--samples) or HuggingFace dataset (--huggingface-dataset) is required."
            in captured.out
        )

    def test_preview_remote_url_sample_file_not_found(
        self, mock_check_environment, capsys
    ):
        args = MockArgs(
            remote_url="http://fake-remote-eval.com", samples="non_existent.jsonl"
        )
        return_code = preview_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error: Sample file 'non_existent.jsonl' not found" in captured.out


class TestPreviewCommandLocalMode:
    @patch("eval_protocol.cli_commands.preview.preview_evaluation")
    def test_preview_local_mode_success(
        self, mock_preview_eval, mock_check_environment, tmp_path, capsys
    ):
        sample_data = [{"messages": [{"role": "user", "content": "Test"}]}]
        temp_sample_file = create_temp_jsonl(tmp_path, sample_data)

        mock_preview_result = MagicMock()
        mock_preview_eval.return_value = mock_preview_result

        args = MockArgs(metrics_folders=["mf=path"], samples=temp_sample_file)
        return_code = preview_command(args)

        assert return_code == 0
        mock_preview_eval.assert_called_once()
        mock_preview_result.display.assert_called_once()

    def test_preview_local_mode_missing_metrics_folders(
        self, mock_check_environment, tmp_path, capsys
    ):
        sample_data = [{"messages": [{"role": "user", "content": "Test"}]}]
        temp_sample_file = create_temp_jsonl(tmp_path, sample_data)
        args = MockArgs(samples=temp_sample_file)

        return_code = preview_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "Error: Either --remote-url or --metrics-folders must be specified."
            in captured.out
        )
