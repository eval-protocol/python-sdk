import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval_protocol.evaluation import Evaluator, create_evaluation, preview_evaluation


def create_test_folder():
    """Create a temporary folder with a main.py file for testing"""
    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "main.py"), "w") as f:
        f.write(
            """
def evaluate(messages, ground_truth=None, tools=None, **kwargs): # Changed original_messages to ground_truth
    if not messages:
        return {'score': 0.0, 'reason': 'No messages found'}
    last_message = messages[-1]
    content = last_message.get('content', '')
    word_count = len(content.split())
    score = min(word_count / 100, 1.0)
    return {
        'score': score,
        'reason': f'Word count: {word_count}'
    }
"""
        )
    return tmp_dir


def create_sample_file():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    samples = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {
                    "role": "assistant",
                    "content": "AI stands for Artificial Intelligence.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for information",
                    },
                }
            ],
        },
    ]
    with os.fdopen(fd, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return path


@pytest.fixture
def mock_env_variables(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_account")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")


@pytest.fixture
def mock_requests_post():
    with patch("requests.post") as mock_post:
        default_response = {
            "name": "accounts/test_account/evaluators/test-eval",
            "displayName": "Test Evaluator",
            "description": "Test description",
            "multiMetrics": False,
        }
        preview_response = {
            "totalSamples": 2,
            "totalRuntimeMs": 1234,
            "results": [
                {
                    "success": True,
                    "score": 0.7,
                    "perMetricEvals": {"quality": 0.8, "relevance": 0.7, "safety": 0.9},
                },
                {
                    "success": True,
                    "score": 0.5,
                    "perMetricEvals": {"quality": 0.6, "relevance": 0.4, "safety": 0.8},
                },
            ],
        }

        def side_effect(*args, **kwargs):
            url = args[0]
            response = mock_post.return_value
            if "previewEvaluator" in url:
                response.json.return_value = preview_response
            else:
                response.json.return_value = default_response
            return response

        mock_post.side_effect = side_effect
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = default_response
        yield mock_post


def test_integration_single_metric(mock_env_variables, mock_requests_post):
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    try:
        preview_result = preview_evaluation(
            metric_folders=[f"test_metric={tmp_dir}"],
            sample_file=sample_file,
            max_samples=2,
        )
        assert preview_result.total_samples == 2
        assert len(preview_result.results) == 2
        evaluator = create_evaluation(
            evaluator_id="test-eval",
            metric_folders=[f"test_metric={tmp_dir}"],
            display_name="Test Evaluator",
            description="Test description",
        )
        assert evaluator["name"] == "accounts/test_account/evaluators/test-eval"
        assert evaluator["displayName"] == "Test Evaluator"
        assert mock_requests_post.call_count >= 1
        args_call, kwargs_call = mock_requests_post.call_args_list[-1]
        url = args_call[0]
        payload = kwargs_call.get("json")
        assert "api.fireworks.ai/v1/accounts/test_account/evaluators" in url
        if "evaluator" in payload:  # Dev API
            assert "evaluatorId" in payload and payload["evaluatorId"] == "test-eval"
            assert "criteria" in payload["evaluator"] and len(payload["evaluator"]["criteria"]) > 0
            assert payload["evaluator"]["criteria"][0]["type"] == "CODE_SNIPPETS"
        else:  # Prod API
            assert "evaluationId" in payload and payload["evaluationId"] == "test-eval"
            assert "assertions" in payload["evaluation"] and len(payload["evaluation"]["assertions"]) > 0
            assert payload["evaluation"]["assertions"][0]["assertionType"] == "CODE"
    finally:
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)


def test_integration_multi_metrics(mock_env_variables, mock_requests_post):
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    try:
        preview_result = preview_evaluation(multi_metrics=True, folder=tmp_dir, sample_file=sample_file, max_samples=2)
        assert preview_result.total_samples == 2
        assert len(preview_result.results) == 2
        assert hasattr(preview_result.results[0], "per_metric_evals")
        assert "quality" in preview_result.results[0].per_metric_evals
        mock_requests_post.reset_mock()
        mock_requests_post.return_value.json.return_value = {
            "name": "accounts/test_account/evaluators/test-eval",
            "displayName": "Multi Metrics Evaluator",
            "description": "Test multi-metrics evaluator",
            "multiMetrics": True,
        }
        evaluator = create_evaluation(
            evaluator_id="multi-metrics-eval",
            multi_metrics=True,
            folder=tmp_dir,
            display_name="Multi Metrics Evaluator",
            description="Test multi-metrics evaluator",
        )
        assert evaluator["name"] == "accounts/test_account/evaluators/test-eval"
        assert mock_requests_post.call_count >= 1
        args_call, kwargs_call = mock_requests_post.call_args_list[-1]
        payload = kwargs_call.get("json")
        if "evaluator" in payload:  # Dev API
            assert payload["evaluatorId"] == "multi-metrics-eval"
            assert payload["evaluator"]["multiMetrics"] is True
        else:  # Prod API
            assert payload["evaluationId"] == "multi-metrics-eval"
    finally:
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)


@patch("sys.exit")
def test_integration_cli_commands(mock_sys_exit, mock_env_variables, mock_requests_post):  # Corrected parameter name
    from eval_protocol.cli import deploy_command, preview_command

    mock_sys_exit.side_effect = lambda code=0: None

    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    try:
        # Test preview command
        with patch("eval_protocol.cli_commands.preview.preview_evaluation") as mock_preview_eval_func:
            mock_preview_result = MagicMock()
            mock_preview_result.display = MagicMock()
            mock_preview_eval_func.return_value = mock_preview_result
            args = MagicMock()
            args.metrics_folders = [f"test_metric={tmp_dir}"]
            args.samples = sample_file
            args.max_samples = 2
            args.huggingface_dataset = None
            args.huggingface_split = "train"
            args.huggingface_prompt_key = "prompt"
            args.huggingface_response_key = "response"
            args.huggingface_key_map = None
            args.remote_url = None  # Explicitly set for local path

            with patch("eval_protocol.cli_commands.preview.Path.exists", return_value=True):
                result = preview_command(args)
                assert result == 0
                mock_preview_eval_func.assert_called_once_with(
                    metric_folders=[f"test_metric={tmp_dir}"],
                    sample_file=sample_file,
                    max_samples=2,
                    huggingface_dataset=None,
                    huggingface_split="train",
                    huggingface_prompt_key="prompt",
                    huggingface_response_key="response",
                    huggingface_message_key_map=None,
                )
                mock_preview_result.display.assert_called_once()

        # Test deploy command
        with patch("eval_protocol.cli_commands.deploy.create_evaluation") as mock_create_eval_func:
            mock_create_eval_func.return_value = {
                "name": "accounts/test_account/evaluators/test-eval",
                "displayName": "Test Evaluator",
                "description": "Test description",
                "multiMetrics": False,
            }
            args = MagicMock()
            args.metrics_folders = [f"test_metric={tmp_dir}"]
            args.id = "test-eval"
            args.display_name = "Test Evaluator"
            args.description = "Test description"
            args.force = False
            args.huggingface_dataset = None
            args.huggingface_split = "train"
            args.huggingface_prompt_key = "prompt"
            args.huggingface_response_key = "response"
            args.huggingface_key_map = None
            args.remote_url = None  # Explicitly set for local path
            args.target = "fireworks"  # Explicitly set target for this test path

            result = deploy_command(args)
            assert result == 0
            mock_create_eval_func.assert_called_once_with(
                evaluator_id="test-eval",
                metric_folders=[f"test_metric={tmp_dir}"],
                display_name="Test Evaluator",
                description="Test description",
                force=False,
                huggingface_dataset=None,
                huggingface_split="train",
                huggingface_message_key_map=None,
                huggingface_prompt_key="prompt",
                huggingface_response_key="response",
            )
    finally:
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)
