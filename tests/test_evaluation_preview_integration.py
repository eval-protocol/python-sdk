import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Load the evaluation_preview_example module directly from the examples folder
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Spec for module {name} has no loader")
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluation_preview_example():
    # Path to the evaluation_preview_example.py file
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "evaluation_preview_example.py",
    )

    # Load the module
    return load_module_from_path("evaluation_preview_example", file_path)


@pytest.fixture
def mock_env_variables(monkeypatch):
    """Set environment variables for testing"""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_account")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")


@pytest.fixture
def mock_preview_api():
    """Mock the preview API calls"""
    with patch("requests.post") as mock_post:
        # Set up mock for preview API
        preview_response = {
            "totalSamples": 2,
            "totalRuntimeMs": 1234,
            "results": [
                {
                    "success": True,
                    "score": 0.26,
                    "perMetricEvals": {
                        "word_count": {
                            "score": 0.26,
                            "reason": "Word count: 26",
                        }
                    },
                },
                {
                    "success": True,
                    "score": 0.22,
                    "perMetricEvals": {
                        "word_count": {
                            "score": 0.22,
                            "reason": "Word count: 22",
                        }
                    },
                },
            ],
        }

        mock_post.return_value = MagicMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = preview_response

        yield mock_post


@pytest.fixture
def mock_create_api():
    """Mock the create API calls"""
    with patch("requests.post") as mock_post:
        create_response = {
            "name": "accounts/test_account/evaluators/word-count-eval",
            "displayName": "Word Count Evaluator",
            "description": "Evaluates responses based on word count",
        }

        def side_effect(*args, **kwargs):
            url = args[0]
            payload = kwargs.get("json", {})
            response = mock_post.return_value

            if "getUploadEndpoint" in url:
                # Return signed URL for upload
                filename_to_size = payload.get("filename_to_size", {})
                signed_urls = {}
                for filename in filename_to_size.keys():
                    signed_urls[filename] = f"https://storage.googleapis.com/test-bucket/{filename}?signed=true"
                response.json.return_value = {"filenameToSignedUrls": signed_urls}
            elif "validateUpload" in url:
                response.json.return_value = {"success": True, "valid": True}
            else:
                response.json.return_value = create_response

            response.status_code = 200
            return response

        mock_post.side_effect = side_effect
        mock_post.return_value = MagicMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = create_response
        mock_post.return_value.raise_for_status = MagicMock()

        yield mock_post


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
def mock_word_count_metric():
    """Create a temporary directory with a word count metric"""
    tmp_dir = tempfile.mkdtemp()

    # Create the metrics/word_count directory
    os.makedirs(os.path.join(tmp_dir, "metrics", "word_count"), exist_ok=True)

    # Create main.py in the word_count directory
    with open(os.path.join(tmp_dir, "metrics", "word_count", "main.py"), "w") as f:
        f.write(
            """
def evaluate(messages, ground_truth=None, tools=None, **kwargs):
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

    # Create a samples directory and sample file
    os.makedirs(os.path.join(tmp_dir, "samples"), exist_ok=True)

    # Create a sample file
    with open(os.path.join(tmp_dir, "samples", "samples.jsonl"), "w") as f:
        f.write(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {
                            "role": "assistant",
                            "content": "Hi there! How can I help you today?",
                        },
                    ]
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "What is AI?"},
                        {
                            "role": "assistant",
                            "content": "AI stands for Artificial Intelligence.",
                        },
                    ]
                }
            )
            + "\n"
        )

    yield tmp_dir

    # Clean up
    import shutil

    shutil.rmtree(tmp_dir)


def test_preview_evaluation(mock_env_variables, mock_preview_api, monkeypatch):
    """Test the preview_evaluation function in isolation"""
    from eval_protocol.evaluation import preview_evaluation

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a metrics directory with word_count
        os.makedirs(os.path.join(tmp_dir, "word_count"), exist_ok=True)

        # Create main.py in the word_count directory
        with open(os.path.join(tmp_dir, "word_count", "main.py"), "w") as f:
            f.write(
                """
def evaluate(messages, ground_truth=None, tools=None, **kwargs):
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

        # Create a temporary sample file
        sample_fd, sample_path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(sample_fd, "w") as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {
                                "role": "assistant",
                                "content": "Hi there! How can I help you today?",
                            },
                        ]
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "What is AI?"},
                            {
                                "role": "assistant",
                                "content": "AI stands for Artificial Intelligence.",
                            },
                        ]
                    }
                )
                + "\n"
            )

        # Set used_preview_api flag to simulate successful preview
        import eval_protocol.evaluation

        eval_protocol.evaluation.used_preview_api = True

        # Call preview_evaluation
        result = preview_evaluation(
            metric_folders=[f"word_count={os.path.join(tmp_dir, 'word_count')}"],
            sample_file=sample_path,
            max_samples=2,
        )

        # Clean up
        os.unlink(sample_path)

        # Verify results
        assert result.total_samples == 2
        assert len(result.results) == 2
        # Assuming result.results[0] is an object, use attribute access
        assert result.results[0].score == 0.26
        assert hasattr(result.results[0], "per_metric_evals")
        assert "word_count" in result.results[0].per_metric_evals


def test_create_evaluation(mock_env_variables, mock_create_api, mock_gcs_upload, monkeypatch):
    """Test the create_evaluation function in isolation"""
    from eval_protocol.evaluation import create_evaluation

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a metrics directory with word_count
        os.makedirs(os.path.join(tmp_dir, "word_count"), exist_ok=True)

        # Create main.py in the word_count directory
        with open(os.path.join(tmp_dir, "word_count", "main.py"), "w") as f:
            f.write(
                """
def evaluate(messages, ground_truth=None, tools=None, **kwargs):
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

        # Create requirements.txt
        with open(os.path.join(tmp_dir, "requirements.txt"), "w") as f:
            f.write("eval-protocol>=0.1.0\n")

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_dir)

        try:
            # Call create_evaluation
            result = create_evaluation(
                evaluator_id="word-count-eval",
                metric_folders=[f"word_count={os.path.join(tmp_dir, 'word_count')}"],
                display_name="Word Count Evaluator",
                description="Evaluates responses based on word count",
                force=True,
            )

            # Verify results
            assert result["name"] == "accounts/test_account/evaluators/word-count-eval"
            assert result["displayName"] == "Word Count Evaluator"
            assert result["description"] == "Evaluates responses based on word count"
        finally:
            os.chdir(original_cwd)


def test_preview_then_create(monkeypatch, mock_env_variables, mock_preview_api, mock_create_api, mock_gcs_upload):
    """Test the full example flow (simulated)"""
    # Patch input to always return 'y'
    monkeypatch.setattr("builtins.input", lambda _: "y")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a metrics directory with word_count
        os.makedirs(os.path.join(tmp_dir, "word_count"), exist_ok=True)

        # Create main.py in the word_count directory
        with open(os.path.join(tmp_dir, "word_count", "main.py"), "w") as f:
            f.write(
                """
def evaluate(messages, ground_truth=None, tools=None, **kwargs):
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

        # Create requirements.txt
        with open(os.path.join(tmp_dir, "requirements.txt"), "w") as f:
            f.write("eval-protocol>=0.1.0\n")

        # Create a temporary sample file
        sample_fd, sample_path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(sample_fd, "w") as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {
                                "role": "assistant",
                                "content": "Hi there! How can I help you today?",
                            },
                        ]
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "What is AI?"},
                            {
                                "role": "assistant",
                                "content": "AI stands for Artificial Intelligence.",
                            },
                        ]
                    }
                )
                + "\n"
            )

        # Create a patched example module with modified paths
        from eval_protocol.evaluation import create_evaluation, preview_evaluation

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_dir)

        try:
            # Define a patched main function
            def patched_main():
                # Preview the evaluation using metrics folder and samples file
                print("Previewing evaluation...")
                preview_result = preview_evaluation(
                    metric_folders=[f"word_count={os.path.join(tmp_dir, 'word_count')}"],
                    sample_file=sample_path,
                    max_samples=2,
                )

                preview_result.display()

                # Check if 'used_preview_api' attribute exists and is True
                import eval_protocol.evaluation as evaluation_module

                # For testing, always assume the API was used successfully
                evaluation_module.used_preview_api = True

                print("\nCreating evaluation...")
                try:
                    evaluator = create_evaluation(
                        evaluator_id="word-count-eval",
                        metric_folders=[f"word_count={os.path.join(tmp_dir, 'word_count')}"],
                        display_name="Word Count Evaluator",
                        description="Evaluates responses based on word count",
                        force=True,
                    )
                    print(f"Created evaluator: {evaluator['name']}")
                    return evaluator
                except Exception as e:
                    print(f"Error creating evaluator: {str(e)}")
                    print("Make sure you have proper Fireworks API credentials set up.")
                    return None

            # Run the patched main function
            result = patched_main()

            # Clean up
            os.unlink(sample_path)

            # Verify the result
            assert result is not None
            assert result["name"] == "accounts/test_account/evaluators/word-count-eval"
        finally:
            os.chdir(original_cwd)
