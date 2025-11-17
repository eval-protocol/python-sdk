"""
End-to-end tests for ep upload command.

Tests the complete upload workflow:
1. Discovery of @evaluation_test decorated functions
2. Upload command execution
3. API calls (create, getUploadEndpoint, validateUpload)
4. Tar.gz creation and GCS upload
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def create_test_project_with_evaluation_test(test_content: str, filename: str = "test_eval.py"):
    """
    Helper to create a proper test project structure for pytest discovery.

    Creates:
        project_dir/
            requirements.txt
            {filename}  <- test_content goes here (at root, not in subdirectory)

    Returns:
        tuple: (project_dir, test_file_path)
    """
    test_project_dir = tempfile.mkdtemp()

    # Put test file at root (not in subdirectory) to avoid import issues
    test_file_path = Path(test_project_dir) / filename
    test_file_path.write_text(test_content)

    # Create requirements.txt (required for upload)
    (Path(test_project_dir) / "requirements.txt").write_text("eval-protocol>=0.1.0\n")

    # Add to sys.path for imports
    if test_project_dir not in sys.path:
        sys.path.insert(0, test_project_dir)

    return test_project_dir, test_file_path


@pytest.fixture
def mock_env_variables(monkeypatch):
    """Set up test environment variables"""
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
    """Mock requests.post for all API endpoints"""
    with patch("requests.post") as mock_post:
        validate_response = {"success": True, "valid": True}
        create_response = {
            "name": "accounts/test_account/evaluators/test-eval",
            "displayName": "Test Evaluator",
            "description": "Test description",
        }

        def side_effect(*args, **kwargs):
            url = args[0]
            payload = kwargs.get("json", {})
            response = mock_post.return_value

            if "getUploadEndpoint" in url:
                # Dynamically create signed URLs for whatever filenames are requested
                filename_to_size = payload.get("filename_to_size", {})
                signed_urls = {}
                for filename in filename_to_size.keys():
                    signed_urls[filename] = f"https://storage.googleapis.com/test-bucket/{filename}?signed=true"
                response.json.return_value = {"filenameToSignedUrls": signed_urls}
            elif "validateUpload" in url:
                response.json.return_value = validate_response
            else:
                # Create evaluator endpoint
                response.json.return_value = create_response

            response.status_code = 200
            return response

        mock_post.side_effect = side_effect
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        yield mock_post


def test_ep_upload_discovers_and_uploads_evaluation_test(
    mock_env_variables, mock_requests_post, mock_requests_get, mock_gcs_upload, monkeypatch
):
    """
    Test the complete ep upload flow:
    - Create a test file with @evaluation_test
    - Discover it using _discover_tests
    - Upload via upload_command
    - Verify all API calls
    """
    from eval_protocol.cli_commands.upload import upload_command, _discover_tests

    # 1. CREATE TEST PROJECT STRUCTURE
    test_content = """
from typing import List
from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test

@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[Message(role="user", content="Hello")]),
        EvaluationRow(messages=[Message(role="user", content="Test message")]),
    ]],
    mode="pointwise"
)
async def test_simple_evaluation(row: EvaluationRow) -> EvaluationRow:
    '''Simple test evaluator'''
    content = row.messages[-1].content if row.messages else ""
    word_count = len(content.split())
    score = min(word_count / 10.0, 1.0)

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=f"Words: {word_count}",
        metrics={"words": {"score": score, "is_score_valid": True}}
    )
    return row
"""

    test_project_dir, test_file_path = create_test_project_with_evaluation_test(test_content, "test_simple_eval.py")

    # Save current directory
    original_cwd = os.getcwd()

    try:
        # Change to test project directory - all operations happen from here
        os.chdir(test_project_dir)

        # 2. TEST DISCOVERY
        discovered_tests = _discover_tests(test_project_dir)

        # Verify discovery
        assert len(discovered_tests) == 1, f"Expected 1 test, found {len(discovered_tests)}"
        discovered_test = discovered_tests[0]
        assert "test_simple_evaluation" in discovered_test.qualname
        assert str(test_file_path) in discovered_test.file_path
        # input_rows automatically creates parametrization, so has_parametrize is True
        assert discovered_test.has_parametrize is True

        # 3. RUN EP UPLOAD COMMAND
        args = argparse.Namespace(
            path=test_project_dir,
            entry=None,  # Discover all tests
            id="test-simple-eval",  # Explicit ID
            display_name="Simple Word Count Eval",
            description="E2E test evaluator",
            force=False,
            yes=True,  # Non-interactive
        )

        # Mock the selection (auto-select the discovered test)
        with patch("eval_protocol.cli_commands.upload._prompt_select") as mock_select:
            mock_select.return_value = discovered_tests

            # Execute upload command
            exit_code = upload_command(args)

        # 4. VERIFY SUCCESS
        assert exit_code == 0, "Upload command should return 0 (success)"

        # 5. VERIFY ALL API CALLS IN UPLOAD FLOW
        post_calls = [call[0][0] for call in mock_requests_post.call_args_list]

        # Step 1: Create evaluator (V2 endpoint)
        create_calls = [url for url in post_calls if "evaluatorsV2" in url]
        assert len(create_calls) >= 1, "Should call V2 create endpoint"

        # Step 2: Get upload endpoint
        upload_endpoint_calls = [url for url in post_calls if "getUploadEndpoint" in url]
        assert len(upload_endpoint_calls) >= 1, "Should call getUploadEndpoint"

        # Step 3: Validate upload
        validate_calls = [url for url in post_calls if "validateUpload" in url]
        assert len(validate_calls) >= 1, "Should call validateUpload"

        # Step 4: GCS upload
        assert mock_gcs_upload.send.called, "Should upload tar.gz to GCS"
        gcs_request = mock_gcs_upload.send.call_args[0][0]
        assert gcs_request.method == "PUT", "GCS upload should use PUT"
        assert "storage.googleapis.com" in gcs_request.url, "Should upload to GCS"

        # 6. VERIFY CREATE PAYLOAD STRUCTURE
        create_payload = None
        for call in mock_requests_post.call_args_list:
            url = call[0][0]
            if "evaluatorsV2" in url:
                create_payload = call[1].get("json")
                break

        assert create_payload is not None
        assert "evaluator" in create_payload
        assert create_payload["evaluatorId"] == "test-simple-eval"

        evaluator_data = create_payload["evaluator"]
        assert evaluator_data["displayName"] == "Simple Word Count Eval"
        assert evaluator_data["description"] == "E2E test evaluator"

        # Verify entry point is included
        assert "entryPoint" in evaluator_data, "Should include entry point"
        entry_point = evaluator_data["entryPoint"]
        assert "test_simple_eval.py::test_simple_evaluation" in entry_point

        # Verify criteria structure (minimal, no embedded code)
        criteria = evaluator_data["criteria"]
        assert len(criteria) > 0
        assert criteria[0]["type"] == "CODE_SNIPPETS"
        # Code is uploaded as tar.gz, not embedded in criteria

    finally:
        # Restore original directory
        os.chdir(original_cwd)

        # Cleanup
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)


def test_ep_upload_with_parametrized_test(
    mock_env_variables,
    mock_requests_post,
    mock_requests_get,
    mock_gcs_upload,
):
    """
    Test ep upload with a parametrized @evaluation_test
    Verifies that parametrized tests are discovered and uploaded as single evaluator
    """
    from eval_protocol.cli_commands.upload import upload_command, _discover_tests

    test_content = """
import pytest
from typing import List
from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test

@pytest.mark.parametrize("completion_params", [
    {"model": "model-a", "temperature": 0.0},
    {"model": "model-b", "temperature": 0.5},
])
@evaluation_test(
    input_rows=[[EvaluationRow(messages=[Message(role="user", content="Test")])]],
    mode="pointwise"
)
async def test_multi_model_eval(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = EvaluateResult(score=1.0, reason="Pass")
    return row
"""

    test_project_dir, test_file_path = create_test_project_with_evaluation_test(test_content, "test_parametrized.py")

    original_cwd = os.getcwd()

    try:
        os.chdir(test_project_dir)

        # Discovery should find it as 1 test (with 2 variants)
        discovered_tests = _discover_tests(test_project_dir)

        assert len(discovered_tests) == 1
        discovered_test = discovered_tests[0]
        assert "test_multi_model_eval" in discovered_test.qualname
        assert discovered_test.has_parametrize is True
        assert discovered_test.param_count == 2

        # Upload should work for parametrized tests
        args = argparse.Namespace(
            path=test_project_dir,
            entry=None,
            id="test-param-eval",
            display_name="Parametrized Eval",
            description="Test parametrized evaluator",
            force=False,
            yes=True,
        )

        with patch("eval_protocol.cli_commands.upload._prompt_select") as mock_select:
            mock_select.return_value = discovered_tests
            exit_code = upload_command(args)

        assert exit_code == 0

        # Verify upload flow completed
        post_calls = [call[0][0] for call in mock_requests_post.call_args_list]
        assert any("evaluatorsV2" in url for url in post_calls)
        assert any("getUploadEndpoint" in url for url in post_calls)
        assert any("validateUpload" in url for url in post_calls)
        assert mock_gcs_upload.send.called

    finally:
        os.chdir(original_cwd)
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)


def test_ep_upload_discovery_skips_problematic_files(mock_env_variables):
    """
    Test that discovery properly skips files like setup.py, versioneer.py
    that would cause issues during pytest collection
    """
    from eval_protocol.cli_commands.upload import _discover_tests

    test_content = """
from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluationRow

@evaluation_test(input_rows=[[EvaluationRow()]])
async def test_good_eval(row: EvaluationRow) -> EvaluationRow:
    return row
"""

    test_project_dir, test_file_path = create_test_project_with_evaluation_test(test_content, "test_good.py")

    original_cwd = os.getcwd()

    try:
        os.chdir(test_project_dir)

        # Create problematic files that should be ignored
        setup_py = Path(test_project_dir) / "setup.py"
        setup_py.write_text("""
from setuptools import setup
setup(name='test')
""")

        versioneer_py = Path(test_project_dir) / "versioneer.py"
        versioneer_py.write_text("# versioneer content")

        # Discovery should find only the good test
        discovered_tests = _discover_tests(test_project_dir)

        assert len(discovered_tests) == 1
        assert "test_good_eval" in discovered_tests[0].qualname
        assert "setup.py" not in discovered_tests[0].file_path
        assert "versioneer.py" not in discovered_tests[0].file_path

    finally:
        os.chdir(original_cwd)
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)


def test_ep_upload_discovers_non_test_prefixed_files(mock_env_variables):
    """
    Test that discovery finds @evaluation_test in files like quickstart.py
    (files that don't start with 'test_')
    """
    from eval_protocol.cli_commands.upload import _discover_tests

    test_content = """
from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluationRow

@evaluation_test(input_rows=[[EvaluationRow()]])
async def test_quickstart_eval(row: EvaluationRow) -> EvaluationRow:
    return row
"""

    test_project_dir, test_file_path = create_test_project_with_evaluation_test(
        test_content,
        "ep_upload_non_test_prefixed_eval.py",  # Non test_* filename
    )

    original_cwd = os.getcwd()

    try:
        os.chdir(test_project_dir)

        # Discovery should find it
        discovered_tests = _discover_tests(test_project_dir)

        assert len(discovered_tests) == 1
        assert "test_quickstart_eval" in discovered_tests[0].qualname
        # Verify we discovered a non-test-prefixed file (our unique filename)
        assert "ep_upload_non_test_prefixed_eval.py" in discovered_tests[0].file_path

    finally:
        os.chdir(original_cwd)
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)


def test_ep_upload_complete_workflow_with_entry_point_validation(
    mock_env_variables,
    mock_requests_post,
    mock_requests_get,
    mock_gcs_upload,
):
    """
    Complete workflow test validating:
    - Test file discovery
    - Entry point generation
    - Upload command execution
    - Full 5-step upload flow
    - Payload structure
    """
    from eval_protocol.cli_commands.upload import upload_command, _discover_tests

    test_content = """
from typing import List
from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test

@evaluation_test(
    input_rows=[[
        EvaluationRow(
            messages=[Message(role="user", content="What is 2+2?")],
            ground_truth="4"
        )
    ]],
    mode="pointwise"
)
async def test_math_correctness(row: EvaluationRow) -> EvaluationRow:
    '''Evaluates math responses'''
    response = row.messages[-1].content if len(row.messages) > 1 else ""
    ground_truth = row.ground_truth or ""

    score = 1.0 if response.strip() == ground_truth.strip() else 0.0

    row.evaluation_result = EvaluateResult(
        score=score,
        reason="Match" if score == 1.0 else "Mismatch",
        metrics={"correctness": {"score": score, "is_score_valid": True}}
    )
    return row
"""

    test_project_dir, test_file_path = create_test_project_with_evaluation_test(test_content, "test_math_eval.py")

    original_cwd = os.getcwd()

    try:
        os.chdir(test_project_dir)

        # 1. TEST DISCOVERY
        discovered_tests = _discover_tests(test_project_dir)

        assert len(discovered_tests) == 1
        test = discovered_tests[0]
        assert "test_math_correctness" in test.qualname
        assert test.lineno is not None

        # 2. RUN UPLOAD COMMAND
        args = argparse.Namespace(
            path=test_project_dir,
            entry=None,
            id=None,  # Auto-generate from test name
            display_name=None,  # Auto-generate
            description=None,  # Auto-generate
            force=False,
            yes=True,
        )

        with patch("eval_protocol.cli_commands.upload._prompt_select") as mock_select:
            mock_select.return_value = discovered_tests
            exit_code = upload_command(args)

        assert exit_code == 0

        # 3. VERIFY 5-STEP UPLOAD FLOW
        post_calls = [call[0][0] for call in mock_requests_post.call_args_list]

        # Step 1: Create evaluator
        assert any("evaluatorsV2" in url for url in post_calls), "Missing create call"

        # Step 2: Get upload endpoint
        assert any("getUploadEndpoint" in url for url in post_calls), "Missing getUploadEndpoint call"

        # Step 3: Upload to GCS
        assert mock_gcs_upload.send.called, "Missing GCS upload"
        gcs_request = mock_gcs_upload.send.call_args[0][0]
        assert gcs_request.method == "PUT"
        assert "storage.googleapis.com" in gcs_request.url

        # Step 4: Validate
        assert any("validateUpload" in url for url in post_calls), "Missing validateUpload call"

        # 4. VERIFY PAYLOAD DETAILS
        create_payload = None
        for call in mock_requests_post.call_args_list:
            url = call[0][0]
            if "evaluatorsV2" in url:
                create_payload = call[1].get("json")
                break

        assert create_payload is not None

        # Verify evaluator ID auto-generated from filename + test name
        evaluator_id = create_payload["evaluatorId"]
        assert "test-math-eval" in evaluator_id or "math-correctness" in evaluator_id

        # Verify entry point is path-based (not module-based)
        evaluator_data = create_payload["evaluator"]
        assert "entryPoint" in evaluator_data, "Should include entry point"
        entry_point = evaluator_data["entryPoint"]
        assert "test_math_eval.py::test_math_correctness" in entry_point

        # Verify criteria is minimal
        criteria = evaluator_data["criteria"]
        assert len(criteria) > 0
        assert criteria[0]["type"] == "CODE_SNIPPETS"
        # Code is in tar.gz, not in payload

        # 5. VERIFY TAR.GZ WAS CREATED AND UPLOADED
        # Check getUploadEndpoint call payload
        upload_endpoint_payload = None
        for call in mock_requests_post.call_args_list:
            url = call[0][0]
            if "getUploadEndpoint" in url:
                upload_endpoint_payload = call[1].get("json")
                break

        assert upload_endpoint_payload is not None
        assert "filename_to_size" in upload_endpoint_payload
        # Tar filename is dynamic (based on directory name)
        tar_files = list(upload_endpoint_payload["filename_to_size"].keys())
        assert len(tar_files) == 1, "Should have exactly one tar file"
        tar_filename = tar_files[0]
        assert tar_filename.endswith(".tar.gz"), "Should be a tar.gz file"
        tar_size = upload_endpoint_payload["filename_to_size"][tar_filename]
        assert tar_size > 0, "Tar file should have non-zero size"

    finally:
        os.chdir(original_cwd)
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)


def test_ep_upload_force_flag_triggers_delete_flow(
    mock_env_variables,
    mock_requests_post,
    mock_gcs_upload,
):
    """
    Test that --force flag triggers the check/delete/recreate flow
    """
    from eval_protocol.cli_commands.upload import upload_command, _discover_tests

    test_content = """
from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluationRow

@evaluation_test(input_rows=[[EvaluationRow()]])
async def test_force_eval(row: EvaluationRow) -> EvaluationRow:
    return row
"""

    test_project_dir, test_file_path = create_test_project_with_evaluation_test(test_content, "test_force.py")

    original_cwd = os.getcwd()

    try:
        os.chdir(test_project_dir)

        # Mock requests.get to return 200 (evaluator exists)
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.raise_for_status = MagicMock()

            # Mock requests.delete
            with patch("requests.delete") as mock_delete:
                mock_delete.return_value.status_code = 200
                mock_delete.return_value.raise_for_status = MagicMock()

                discovered_tests = _discover_tests(test_project_dir)

                args = argparse.Namespace(
                    path=test_project_dir,
                    entry=None,
                    id="test-force",
                    display_name=None,
                    description=None,
                    force=True,  # Force flag enabled
                    yes=True,
                )

                with patch("eval_protocol.cli_commands.upload._prompt_select") as mock_select:
                    mock_select.return_value = discovered_tests
                    exit_code = upload_command(args)

                assert exit_code == 0

                # Verify check happened
                assert mock_get.called, "Should check if evaluator exists"

                # Verify delete happened (since mock_get returned 200)
                assert mock_delete.called, "Should delete existing evaluator"

    finally:
        os.chdir(original_cwd)
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)
