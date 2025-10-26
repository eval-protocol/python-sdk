import json
import os
import shutil
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
    # Create requirements.txt (required for upload)
    with open(os.path.join(tmp_dir, "requirements.txt"), "w") as f:
        f.write("eval-protocol>=0.1.0\n")
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
def mock_requests_get():
    """Mock requests.get for force flow check"""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 404  # Evaluator doesn't exist
        mock_get.return_value.raise_for_status = MagicMock()
        yield mock_get


@pytest.fixture
def mock_requests_delete():
    """Mock requests.delete for force flow"""
    with patch("requests.delete") as mock_delete:
        mock_delete.return_value.status_code = 200
        mock_delete.return_value.raise_for_status = MagicMock()
        yield mock_delete


@pytest.fixture
def mock_gcs_upload():
    """Mock the GCS upload via requests.Session"""
    with patch("requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock successful GCS upload
        mock_gcs_response = MagicMock()
        mock_gcs_response.status_code = 200
        mock_gcs_response.raise_for_status = MagicMock()
        mock_session.send.return_value = mock_gcs_response

        yield mock_session


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
        validate_response = {"success": True, "valid": True}

        def side_effect(*args, **kwargs):
            url = args[0]
            payload = kwargs.get("json", {})
            response = mock_post.return_value
            if "previewEvaluator" in url:
                response.json.return_value = preview_response
            elif "getUploadEndpoint" in url:
                # Dynamically create signed URLs for whatever filenames are requested
                filename_to_size = payload.get("filename_to_size", {})
                signed_urls = {}
                for filename in filename_to_size.keys():
                    signed_urls[filename] = f"https://storage.googleapis.com/test-bucket/{filename}?signed=true"
                response.json.return_value = {"filenameToSignedUrls": signed_urls}
            elif "validateUpload" in url:
                response.json.return_value = validate_response
            else:
                response.json.return_value = default_response
            return response

        mock_post.side_effect = side_effect
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = default_response
        mock_post.return_value.raise_for_status = MagicMock()
        yield mock_post


def test_integration_single_metric(mock_env_variables, mock_requests_post, mock_gcs_upload):
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)
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

        # Verify all API calls in the new upload flow
        post_calls = [call[0][0] for call in mock_requests_post.call_args_list]

        # 1. Create evaluator call (V2 endpoint)
        assert any("evaluatorsV2" in url for url in post_calls), "Should call V2 create endpoint"

        # 2. Get upload endpoint call
        assert any("getUploadEndpoint" in url for url in post_calls), "Should call getUploadEndpoint"

        # 3. Validate upload call
        assert any("validateUpload" in url for url in post_calls), "Should call validateUpload"

        # 4. Verify GCS upload happened
        assert mock_gcs_upload.send.called, "Should upload tar.gz to GCS"
        gcs_request = mock_gcs_upload.send.call_args[0][0]
        assert gcs_request.method == "PUT", "GCS upload should use PUT"
        assert "storage.googleapis.com" in gcs_request.url, "Should upload to GCS"

        # Verify create payload structure
        create_call_payload = None
        for call in mock_requests_post.call_args_list:
            url = call[0][0]
            if "evaluatorsV2" in url:
                create_call_payload = call[1].get("json")
                break

        assert create_call_payload is not None, "Should have create payload"
        assert "evaluator" in create_call_payload
        assert "evaluatorId" in create_call_payload and create_call_payload["evaluatorId"] == "test-eval"
        assert "criteria" in create_call_payload["evaluator"]
        assert len(create_call_payload["evaluator"]["criteria"]) > 0
        assert create_call_payload["evaluator"]["criteria"][0]["type"] == "CODE_SNIPPETS"
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.unlink(sample_file)


def test_integration_multi_metrics(mock_env_variables, mock_requests_post, mock_gcs_upload):
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)
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

        # Verify all API calls in the new upload flow
        post_calls = [call[0][0] for call in mock_requests_post.call_args_list]
        assert any("evaluatorsV2" in url for url in post_calls), "Should call V2 create endpoint"
        assert any("getUploadEndpoint" in url for url in post_calls), "Should call getUploadEndpoint"
        assert any("validateUpload" in url for url in post_calls), "Should call validateUpload"

        # Verify GCS upload happened
        assert mock_gcs_upload.send.called, "Should upload tar.gz to GCS"

        # Verify create payload uses V2 format
        create_call_payload = None
        for call in mock_requests_post.call_args_list:
            url = call[0][0]
            if "evaluatorsV2" in url:
                create_call_payload = call[1].get("json")
                break

        assert create_call_payload is not None
        assert "evaluator" in create_call_payload
        assert create_call_payload["evaluatorId"] == "multi-metrics-eval"
        assert create_call_payload["evaluator"]["multiMetrics"] is True
    finally:
        import shutil

        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.unlink(sample_file)


@patch("sys.exit")
def test_integration_cli_commands(mock_sys_exit, mock_env_variables, mock_requests_post):  # Corrected parameter name
    from eval_protocol.cli import deploy_command, preview_command

    mock_sys_exit.side_effect = lambda code=0: None

    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)
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
        os.chdir(original_cwd)
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.unlink(sample_file)
