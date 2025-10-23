import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from eval_protocol.evaluation import Evaluator, create_evaluation, preview_evaluation
from eval_protocol.models import MetricResult


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


def test_evaluator_update_evaluate_signature():
    evaluator = Evaluator()
    old_code = """
def evaluate(entry):
    messages = entry.get('messages', [])
    if not messages: return {'score': 0.0, 'reason': 'No messages found'}
    last_message = messages[-1]
    content = last_message.get('content', '')
    word_count = len(content.split())
    score = min(word_count / 100, 1.0)
    return {'score': score, 'reason': f'Word count: {word_count}'}
    """
    updated_code = evaluator._update_evaluate_signature(old_code)
    assert (
        "def evaluate(messages, ground_truth: Optional[Union[str, List[Dict[str, Any]]]] = None, tools=None, **kwargs)"
        in updated_code
    )
    # The "entry = {" line is no longer part of the compatibility layer for the old_pattern.
    # The compatibility layer now focuses on handling ground_truth.
    assert (
        "if ground_truth is None: # Default ground_truth from messages if not provided" in updated_code
    )  # Check for new compat layer logic
    new_code = """
def evaluate(messages, ground_truth: Optional[Union[str, List[Dict[str, Any]]]] = None, tools=None, **kwargs):
    if not messages: return {'score': 0.0, 'reason': 'No messages found'}
    last_message = messages[-1]
    content = last_message.get('content', '')
    word_count = len(content.split())
    score = min(word_count / 100, 1.0)
    return {'score': score, 'reason': f'Word count: {word_count}'}
    """
    unchanged_code = evaluator._update_evaluate_signature(new_code)
    assert new_code == unchanged_code


@patch("eval_protocol.evaluation.requests.post")
def test_evaluator_preview(mock_requests_post, monkeypatch):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "totalSamples": 2,
        "totalRuntimeMs": 123,
        "results": [
            {
                "index": 0,
                "success": True,
                "score": 0.5,
                "reason": "Reason 1",
                "perMetricEvals": {
                    "test_metric": MetricResult(score=0.5, reason="Metric reason 1", is_score_valid=True).model_dump()
                },
            },
            {
                "index": 1,
                "success": True,
                "score": 0.8,
                "reason": "Reason 2",
                "perMetricEvals": {
                    "test_metric": MetricResult(score=0.8, reason="Metric reason 2", is_score_valid=True).model_dump()
                },
            },
        ],
    }
    mock_requests_post.return_value = mock_response

    monkeypatch.setenv("FIREWORKS_API_KEY", "test_preview_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_preview_account")
    # Using a mock API base to prevent real calls
    monkeypatch.setenv("FIREWORKS_API_BASE", "http://mock-api-server")  # Changed to avoid actual localhost call

    # Mock requests.post for the preview call
    class MockResponsePreview:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.exceptions.HTTPError(f"Mock API Error: {self.status_code}")

    def mock_post_preview(*args, **kwargs):
        expected_url_preview = "http://mock-api-server/v1/accounts/test_preview_account/evaluators:previewEvaluator"
        if args[0] == expected_url_preview:
            # Simulate a successful preview API response
            return MockResponsePreview(
                {
                    "totalSamples": 2,
                    "totalRuntimeMs": 150,  # Example runtime
                    "results": [
                        {
                            "success": True,
                            "score": 0.75,
                            "perMetricEvals": {"test_metric": 0.75},
                        },
                        {
                            "success": True,
                            "score": 0.85,
                            "perMetricEvals": {"test_metric": 0.85},
                        },
                    ],
                }
            )
        # Fallback for other URLs if any, though not expected in this test
        return MockResponsePreview({"error": "Unexpected URL"}, 404)

    monkeypatch.setattr("requests.post", mock_post_preview)

    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    try:
        evaluator = Evaluator()
        evaluator.load_metric_folder("test_metric", tmp_dir)
        preview_result = evaluator.preview(sample_file, max_samples=2)
        assert preview_result.total_samples == 2
        assert preview_result.total_runtime_ms >= 0
        assert len(preview_result.results) == 2
        assert preview_result.results[0].index == 0
        assert preview_result.results[0].success is True
        assert hasattr(preview_result.results[0], "score")
        assert preview_result.results[0].score == 0.75
        assert hasattr(preview_result.results[0], "per_metric_evals")  # Attribute name in Python object
        assert "test_metric" in preview_result.results[0].per_metric_evals
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.unlink(sample_file)


@patch("eval_protocol.evaluation.requests.post")
def test_preview_evaluation_helper(mock_requests_post, monkeypatch):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "totalSamples": 2,
        "totalRuntimeMs": 100,
        "results": [
            {
                "index": 0,
                "success": True,
                "score": 0.6,
                "reason": "Helper Reason 1",
                "perMetricEvals": {
                    "test_metric": MetricResult(
                        score=0.6, reason="Helper Metric reason 1", is_score_valid=True
                    ).model_dump()
                },
            },
            {
                "index": 1,
                "success": True,
                "score": 0.7,
                "reason": "Helper Reason 2",
                "perMetricEvals": {
                    "test_metric": MetricResult(
                        score=0.7, reason="Helper Metric reason 2", is_score_valid=True
                    ).model_dump()
                },
            },
        ],
    }
    mock_requests_post.return_value = mock_response

    monkeypatch.setenv("FIREWORKS_API_KEY", "test_helper_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_helper_account")
    # Using a mock API base to prevent real calls
    monkeypatch.setenv("FIREWORKS_API_BASE", "http://mock-api-server-helper")  # Changed

    # Mock requests.post for the preview_evaluation helper call
    class MockResponseHelperPreview:  # Renamed to avoid conflict if in same scope, though not strictly necessary here
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.exceptions.HTTPError(f"Mock API Error: {self.status_code}")

    def mock_post_helper_preview(*args, **kwargs):
        expected_url_helper_preview = (
            "http://mock-api-server-helper/v1/accounts/test_helper_account/evaluators:previewEvaluator"
        )
        if args[0] == expected_url_helper_preview:
            return MockResponseHelperPreview(
                {
                    "totalSamples": 2,
                    "totalRuntimeMs": 160,
                    "results": [
                        {
                            "success": True,
                            "score": 0.65,
                            "perMetricEvals": {"test_metric": 0.65},
                        },
                        {
                            "success": True,
                            "score": 0.70,
                            "perMetricEvals": {"test_metric": 0.70},
                        },
                    ],
                }
            )
        return MockResponseHelperPreview({"error": "Unexpected URL for helper"}, 404)

    monkeypatch.setattr("requests.post", mock_post_helper_preview)

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
        assert preview_result.results[0].score == 0.65
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.unlink(sample_file)


def test_create_evaluation_helper(monkeypatch):
    tmp_dir = create_test_folder()
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_account")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")

    original_cwd = os.getcwd()

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self.json_data

        def raise_for_status(self):  # pragma: no cover
            if self.status_code != 200:
                raise Exception("API Error")

    create_called = False
    upload_endpoint_called = False
    validate_called = False

    def mock_post(*args, **kwargs):
        nonlocal create_called, upload_endpoint_called, validate_called
        url = args[0]
        payload = kwargs.get("json", {})

        # Handle different endpoints in the upload flow
        if "getUploadEndpoint" in url:
            upload_endpoint_called = True
            # Dynamically create signed URLs for whatever filenames are requested
            filename_to_size = payload.get("filename_to_size", {})
            signed_urls = {}
            for filename in filename_to_size.keys():
                signed_urls[filename] = f"https://storage.googleapis.com/test-bucket/{filename}?signed=true"
            return MockResponse({"filenameToSignedUrls": signed_urls})
        elif "validateUpload" in url:
            validate_called = True
            return MockResponse({"success": True, "valid": True})
        else:
            # Create evaluator endpoint
            create_called = True
            assert "evaluator" in payload
            assert "evaluatorId" in payload
            evaluator_data = payload["evaluator"]
            assert "criteria" in evaluator_data
            criteria = evaluator_data["criteria"]
            assert len(criteria) > 0
            criterion = criteria[0]
            assert criterion["type"] == "CODE_SNIPPETS"
            # Code is now uploaded as tar.gz, not in criteria

            return MockResponse(
                {
                    "name": "accounts/test_account/evaluators/test-eval",
                    "displayName": "Test Evaluator",
                    "description": "Test description",
                    "multiMetrics": False,
                }
            )

    # Mock GCS upload
    from unittest.mock import MagicMock

    mock_session = MagicMock()
    mock_gcs_response = MagicMock()
    mock_gcs_response.status_code = 200
    mock_gcs_response.raise_for_status = MagicMock()
    mock_session.send.return_value = mock_gcs_response

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("requests.Session", lambda: mock_session)

    try:
        os.chdir(tmp_dir)
        api_response = create_evaluation(
            evaluator_id="test-eval",
            metric_folders=[f"test_metric={tmp_dir}"],
            display_name="Test Evaluator",
            description="Test description",
        )

        # Verify response
        assert api_response["name"] == "accounts/test_account/evaluators/test-eval"
        assert api_response["displayName"] == "Test Evaluator"
        assert api_response["description"] == "Test description"

        # Verify full upload flow was executed
        assert create_called, "Create endpoint should be called"
        assert upload_endpoint_called, "GetUploadEndpoint should be called"
        assert validate_called, "ValidateUpload should be called"
        assert mock_session.send.called, "GCS upload should happen"

    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)
