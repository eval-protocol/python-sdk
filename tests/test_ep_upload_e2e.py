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

import eval_protocol.cli_commands.utils as cli_utils


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
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")

    monkeypatch.setattr(cli_utils, "verify_api_key_and_get_account_id", lambda *a, **k: "test_account")
    # Upload ultimately calls into eval_protocol.evaluation for API calls; keep it offline too.
    monkeypatch.setattr("eval_protocol.evaluation.get_fireworks_account_id", lambda: "test_account")


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
def mock_fireworks_client():
    """Mock the Fireworks SDK client used in evaluation.py"""
    with patch("eval_protocol.evaluation.Fireworks") as mock_fw_class:
        mock_client = MagicMock()
        mock_fw_class.return_value = mock_client

        # Mock evaluators.create response
        mock_create_response = MagicMock()
        mock_create_response.name = "accounts/test_account/evaluators/test-eval"
        mock_create_response.display_name = "Test Evaluator"
        mock_create_response.description = "Test description"
        mock_client.evaluators.create.return_value = mock_create_response

        # Mock evaluators.get_upload_endpoint response - will be set dynamically
        def get_upload_endpoint_side_effect(evaluator_id, filename_to_size):
            response = MagicMock()
            signed_urls = {}
            for filename in filename_to_size.keys():
                signed_urls[filename] = f"https://storage.googleapis.com/test-bucket/{filename}?signed=true"
            response.filename_to_signed_urls = signed_urls
            return response

        mock_client.evaluators.get_upload_endpoint.side_effect = get_upload_endpoint_side_effect

        # Mock evaluators.validate_upload response
        mock_validate_response = MagicMock()
        mock_validate_response.success = True
        mock_validate_response.valid = True
        mock_client.evaluators.validate_upload.return_value = mock_validate_response

        # Mock evaluators.get (for force flow - raises NotFoundError by default)
        import fireworks

        mock_client.evaluators.get.side_effect = fireworks.NotFoundError(
            "Evaluator not found",
            response=MagicMock(status_code=404),
            body={"error": "not found"},
        )

        # Mock evaluators.delete
        mock_client.evaluators.delete.return_value = None

        yield mock_client


@pytest.fixture
def mock_platform_api_client():
    """Mock the Fireworks SDK client used in platform_api.py for secrets"""
    with patch("eval_protocol.platform_api.Fireworks") as mock_fw_class:
        mock_client = MagicMock()
        mock_fw_class.return_value = mock_client

        # Mock secrets.get - raise NotFoundError to simulate secret doesn't exist
        from fireworks import NotFoundError

        mock_client.secrets.get.side_effect = NotFoundError(
            "Secret not found",
            response=MagicMock(status_code=404),
            body={"error": "not found"},
        )

        # Mock secrets.create - successful
        mock_create_response = MagicMock()
        mock_create_response.name = "accounts/test_account/secrets/test-secret"
        mock_client.secrets.create.return_value = mock_create_response

        yield mock_client


def test_ep_upload_discovers_and_uploads_evaluation_test(
    mock_env_variables, mock_fireworks_client, mock_platform_api_client, mock_gcs_upload, monkeypatch
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

        # 5. VERIFY ALL API CALLS IN UPLOAD FLOW via Fireworks SDK
        # Step 1: Create evaluator
        assert mock_fireworks_client.evaluators.create.called, "Should call evaluators.create"

        # Step 2: Get upload endpoint
        assert mock_fireworks_client.evaluators.get_upload_endpoint.called, (
            "Should call evaluators.get_upload_endpoint"
        )

        # Step 3: Validate upload
        assert mock_fireworks_client.evaluators.validate_upload.called, "Should call evaluators.validate_upload"

        # Step 4: GCS upload
        assert mock_gcs_upload.send.called, "Should upload tar.gz to GCS"
        gcs_request = mock_gcs_upload.send.call_args[0][0]
        assert gcs_request.method == "PUT", "GCS upload should use PUT"
        assert "storage.googleapis.com" in gcs_request.url, "Should upload to GCS"

        # 6. VERIFY CREATE PAYLOAD STRUCTURE
        create_call = mock_fireworks_client.evaluators.create.call_args
        assert create_call is not None

        # Check evaluator_id
        assert create_call.kwargs.get("evaluator_id") == "test-simple-eval"

        # Check evaluator params
        evaluator_params = create_call.kwargs.get("evaluator", {})
        assert evaluator_params.get("display_name") == "Simple Word Count Eval"
        assert evaluator_params.get("description") == "E2E test evaluator"

        # Verify entry point is included
        assert "entry_point" in evaluator_params, "Should include entry point"
        entry_point = evaluator_params["entry_point"]
        assert "test_simple_eval.py::test_simple_evaluation" in entry_point

    finally:
        # Restore original directory
        os.chdir(original_cwd)

        # Cleanup
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)


def test_ep_upload_with_parametrized_test(
    mock_env_variables,
    mock_fireworks_client,
    mock_platform_api_client,
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

        # Verify upload flow completed via Fireworks SDK
        assert mock_fireworks_client.evaluators.create.called
        assert mock_fireworks_client.evaluators.get_upload_endpoint.called
        assert mock_fireworks_client.evaluators.validate_upload.called
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
    mock_fireworks_client,
    mock_platform_api_client,
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

        # 3. VERIFY 5-STEP UPLOAD FLOW via Fireworks SDK
        # Step 1: Create evaluator
        assert mock_fireworks_client.evaluators.create.called, "Missing create call"

        # Step 2: Get upload endpoint
        assert mock_fireworks_client.evaluators.get_upload_endpoint.called, "Missing getUploadEndpoint call"

        # Step 3: Upload to GCS
        assert mock_gcs_upload.send.called, "Missing GCS upload"
        gcs_request = mock_gcs_upload.send.call_args[0][0]
        assert gcs_request.method == "PUT"
        assert "storage.googleapis.com" in gcs_request.url

        # Step 4: Validate
        assert mock_fireworks_client.evaluators.validate_upload.called, "Missing validateUpload call"

        # 4. VERIFY PAYLOAD DETAILS
        create_call = mock_fireworks_client.evaluators.create.call_args
        assert create_call is not None

        # Verify evaluator ID auto-generated from filename + test name
        evaluator_id = create_call.kwargs.get("evaluator_id", "")
        assert "test-math-eval" in evaluator_id or "math-correctness" in evaluator_id

        # Verify entry point is path-based (not module-based)
        evaluator_params = create_call.kwargs.get("evaluator", {})
        assert "entry_point" in evaluator_params, "Should include entry point"
        entry_point = evaluator_params["entry_point"]
        assert "test_math_eval.py::test_math_correctness" in entry_point

        # 5. VERIFY TAR.GZ WAS CREATED AND UPLOADED
        # Check getUploadEndpoint call payload
        upload_call = mock_fireworks_client.evaluators.get_upload_endpoint.call_args
        assert upload_call is not None
        filename_to_size = upload_call.kwargs.get("filename_to_size", {})
        assert filename_to_size, "Should have filename_to_size"
        # Tar filename is dynamic (based on directory name)
        tar_files = list(filename_to_size.keys())
        assert len(tar_files) == 1, "Should have exactly one tar file"
        tar_filename = tar_files[0]
        assert tar_filename.endswith(".tar.gz"), "Should be a tar.gz file"
        tar_size = int(filename_to_size[tar_filename])
        assert tar_size > 0, "Tar file should have non-zero size"

    finally:
        os.chdir(original_cwd)
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)


def test_ep_upload_force_flag_triggers_delete_flow(
    mock_env_variables,
    mock_gcs_upload,
    mock_platform_api_client,
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

        # Mock the Fireworks client with evaluator existing (for force flow)
        with patch("eval_protocol.evaluation.Fireworks") as mock_fw_class:
            mock_client = MagicMock()
            mock_fw_class.return_value = mock_client

            # Mock evaluators.get to return an existing evaluator (not raise NotFoundError)
            mock_existing_evaluator = MagicMock()
            mock_existing_evaluator.name = "accounts/test_account/evaluators/test-force"
            mock_client.evaluators.get.return_value = mock_existing_evaluator

            # Mock evaluators.delete
            mock_client.evaluators.delete.return_value = None

            # Mock evaluators.create response
            mock_create_response = MagicMock()
            mock_create_response.name = "accounts/test_account/evaluators/test-force"
            mock_client.evaluators.create.return_value = mock_create_response

            # Mock get_upload_endpoint
            def get_upload_endpoint_side_effect(evaluator_id, filename_to_size):
                response = MagicMock()
                signed_urls = {}
                for filename in filename_to_size.keys():
                    signed_urls[filename] = f"https://storage.googleapis.com/test-bucket/{filename}?signed=true"
                response.filename_to_signed_urls = signed_urls
                return response

            mock_client.evaluators.get_upload_endpoint.side_effect = get_upload_endpoint_side_effect

            # Mock validate_upload
            mock_client.evaluators.validate_upload.return_value = MagicMock()

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

            # Verify check happened (evaluators.get was called)
            assert mock_client.evaluators.get.called, "Should check if evaluator exists"

            # Verify delete happened (since evaluator existed)
            assert mock_client.evaluators.delete.called, "Should delete existing evaluator"

            # Verify create happened after delete
            assert mock_client.evaluators.create.called, "Should create evaluator after delete"

    finally:
        os.chdir(original_cwd)
        if test_project_dir in sys.path:
            sys.path.remove(test_project_dir)
        shutil.rmtree(test_project_dir, ignore_errors=True)
