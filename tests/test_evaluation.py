import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import requests

from eval_protocol.evaluation import Evaluator, create_evaluation


def create_test_folder():
    """Create a temporary folder with a main.py file for testing"""
    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "main.py"), "w") as f:
        f.write(
            """
def evaluate(messages, original_messages=None, tools=None, **kwargs):
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
            "original_messages": [
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


def test_evaluator_load_metric_folder():
    tmp_dir = create_test_folder()
    try:
        evaluator = Evaluator()
        files = evaluator.load_metric_folder("test_metric", tmp_dir)
        assert "main.py" in files
        assert "test_metric" in evaluator.metric_folders
        assert "test_metric/main.py" in evaluator.code_files
        assert "evaluate" in evaluator.code_files["test_metric/main.py"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluator_load_multi_metrics_folder():
    tmp_dir = create_test_folder()
    try:
        evaluator = Evaluator(multi_metrics=True)
        files = evaluator.load_multi_metrics_folder(tmp_dir)
        assert "main.py" in files
        assert "main.py" in evaluator.code_files
        assert "evaluate" in evaluator.code_files["main.py"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_create_evaluation_helper(monkeypatch):
    tmp_dir = create_test_folder()
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setattr("eval_protocol.evaluation.get_fireworks_account_id", lambda: "test_account")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")

    original_cwd = os.getcwd()

    # Track SDK calls
    create_called = False
    upload_endpoint_called = False
    validate_called = False

    # Mock the Fireworks SDK client methods
    mock_evaluator_result = MagicMock()
    mock_evaluator_result.name = "accounts/test_account/evaluators/test-eval"
    mock_evaluator_result.display_name = "Test Evaluator"
    mock_evaluator_result.description = "Test description"

    def mock_create(evaluator_id, evaluator):
        nonlocal create_called
        create_called = True
        # Verify the evaluator params
        assert evaluator_id == "test-eval"
        assert "display_name" in evaluator
        assert evaluator["display_name"] == "Test Evaluator"
        assert "description" in evaluator
        assert evaluator["description"] == "Test description"
        return mock_evaluator_result

    def mock_get_upload_endpoint(evaluator_id, filename_to_size):
        nonlocal upload_endpoint_called
        upload_endpoint_called = True
        mock_response = MagicMock()
        signed_urls = {}
        for filename in filename_to_size.keys():
            signed_urls[filename] = f"https://storage.googleapis.com/test-bucket/{filename}?signed=true"
        mock_response.filename_to_signed_urls = signed_urls
        return mock_response

    def mock_validate_upload(evaluator_id, body):
        nonlocal validate_called
        validate_called = True
        return MagicMock()

    # Mock GCS upload (still uses requests.Session)
    mock_session = MagicMock()
    mock_gcs_response = MagicMock()
    mock_gcs_response.status_code = 200
    mock_gcs_response.raise_for_status = MagicMock()
    mock_session.send.return_value = mock_gcs_response

    # Patch the Fireworks client
    with patch("eval_protocol.evaluation.Fireworks") as mock_fireworks_class:
        mock_client = MagicMock()
        mock_fireworks_class.return_value = mock_client
        mock_client.evaluators.create = mock_create
        mock_client.evaluators.get_upload_endpoint = mock_get_upload_endpoint
        mock_client.evaluators.validate_upload = mock_validate_upload

        # Patch requests.Session for GCS upload
        monkeypatch.setattr("requests.Session", lambda: mock_session)

        try:
            os.chdir(tmp_dir)
            api_response = create_evaluation(
                evaluator_id="test-eval",
                metric_folders=[f"test_metric={tmp_dir}"],
                display_name="Test Evaluator",
                description="Test description",
            )

            # Verify response (SDK returns an object, not dict)
            assert api_response.name == "accounts/test_account/evaluators/test-eval"
            assert api_response.display_name == "Test Evaluator"
            assert api_response.description == "Test description"

            # Verify full upload flow was executed
            assert create_called, "Create endpoint should be called"
            assert upload_endpoint_called, "GetUploadEndpoint should be called"
            assert validate_called, "ValidateUpload should be called"
            assert mock_session.send.called, "GCS upload should happen"

        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp_dir, ignore_errors=True)
