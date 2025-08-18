import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from eval_protocol.cli_commands.deploy import deploy_command
from eval_protocol.config import GCPCloudRunConfig, RewardKitConfig

# Constants for a dummy reward function module
# This module will be created and deleted by tests needing it.
DUMMY_DEPLOY_TEST_MODULE_NAME = "dummy_deploy_test_module"
DUMMY_DEPLOY_TEST_MODULE_FILENAME = f"{DUMMY_DEPLOY_TEST_MODULE_NAME}.py"
DUMMY_DEPLOY_TEST_FUNCTION_NAME = "my_dummy_deploy_reward_func"
DUMMY_DEPLOY_FUNCTION_REF = f"{DUMMY_DEPLOY_TEST_MODULE_NAME}.{DUMMY_DEPLOY_TEST_FUNCTION_NAME}"
DUMMY_DEPLOY_REQUIREMENTS = "requests==2.25.0\nfastapi==0.70.0"

DUMMY_DEPLOY_MODULE_CONTENT = f"""
from eval_protocol.typed_interface import reward_function

@reward_function(id="test-deploy-func", requirements='''{DUMMY_DEPLOY_REQUIREMENTS}''')
def {DUMMY_DEPLOY_TEST_FUNCTION_NAME}(messages, ground_truth=None, **kwargs):
    return {{"score": 0.5, "reason": "Deployed dummy"}}
"""

# Ensure the CWD (project root) is in sys.path for module loading during tests
if Path.cwd().as_posix() not in sys.path:
    sys.path.insert(0, Path.cwd().as_posix())


@pytest.fixture(scope="function")
def create_dummy_reward_module_for_deploy():
    # Create the dummy module file
    with open(DUMMY_DEPLOY_TEST_MODULE_FILENAME, "w") as f:
        f.write(DUMMY_DEPLOY_MODULE_CONTENT)

    # Ensure the module can be imported by clearing any cached versions
    if DUMMY_DEPLOY_TEST_MODULE_NAME in sys.modules:
        del sys.modules[DUMMY_DEPLOY_TEST_MODULE_NAME]

    yield DUMMY_DEPLOY_FUNCTION_REF  # Provide the function reference to the test

    # Cleanup: remove the dummy module file
    if os.path.exists(DUMMY_DEPLOY_TEST_MODULE_FILENAME):
        os.remove(DUMMY_DEPLOY_TEST_MODULE_FILENAME)
    # Cleanup: remove from sys.modules if it was loaded
    if DUMMY_DEPLOY_TEST_MODULE_NAME in sys.modules:
        del sys.modules[DUMMY_DEPLOY_TEST_MODULE_NAME]


# Load the deploy_example module directly from the examples folder
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
def deploy_example():
    # Path to the deploy_example.py file
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "deploy_example.py",
    )

    # Load the module
    return load_module_from_path("deploy_example", file_path)


@pytest.fixture
def mock_env_variables(monkeypatch):
    """Set environment variables for testing"""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_account")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")


@pytest.fixture
def mock_requests_post():
    """Mock requests.post method"""
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "name": "accounts/test_account/evaluators/informativeness-v1",
            "displayName": "informativeness-v1",
            "description": "Evaluates response informativeness based on specificity and content density",
        }
        yield mock_post


@pytest.fixture
def mock_requests_get():
    """Mock requests.get method"""
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock()
        mock_get.return_value.status_code = 404  # Evaluator doesn't exist
        yield mock_get


def test_deploy_gcp_with_inline_requirements(
    mock_env_variables,  # Ensures FIREWORKS_ACCOUNT_ID etc. are set
    create_dummy_reward_module_for_deploy,  # Creates and cleans up the dummy module
):
    """
    Test the deploy_command with --target gcp-cloud-run, ensuring inline requirements
    from the @reward_function decorator are passed to generate_dockerfile_content.
    """
    function_ref = create_dummy_reward_module_for_deploy
    evaluator_id = "test-gcp-evaluator-with-reqs"

    args = argparse.Namespace(
        id=evaluator_id,
        target="gcp-cloud-run",
        function_ref=function_ref,
        metrics_folders=None,
        remote_url=None,
        display_name=None,
        description=None,
        force=False,
        huggingface_dataset=None,
        huggingface_split=None,
        huggingface_key_map=None,  # This is what argparse would create from --huggingface-key-map
        huggingface_prompt_key=None,
        huggingface_response_key=None,
        local_port=8001,  # Default, not used for GCP
        runtime="python3.10",  # Example runtime
        gcp_project="test-gcp-project",
        gcp_region="us-central1",
        gcp_ar_repo=None,  # Will default
        gcp_auth_mode="api-key",
    )

    # Mock all external dependencies of _deploy_to_gcp_cloud_run and deploy_command
    with (
        patch("eval_protocol.cli_commands.deploy.check_environment", return_value=True) as mock_check_env,
        patch("eval_protocol.cli_commands.deploy.get_config") as mock_get_config,
        patch(
            "eval_protocol.cli_commands.deploy.ensure_artifact_registry_repo_exists",
            return_value=True,
        ) as mock_ensure_ar,
        patch(
            "eval_protocol.cli_commands.deploy.generate_dockerfile_content",
            return_value="DOCKERFILE CONTENT",
        ) as mock_gen_dockerfile,
        patch(
            "eval_protocol.cli_commands.deploy.build_and_push_docker_image",
            return_value=True,
        ) as mock_build_push,
        patch(
            "eval_protocol.cli_commands.deploy.ensure_gcp_secret",
            return_value="projects/p/secrets/s/versions/1",
        ) as mock_ensure_secret,
        patch(
            "eval_protocol.cli_commands.deploy.create_or_update_fireworks_secret",
            return_value=True,
        ) as mock_fw_secret,
        patch(
            "eval_protocol.cli_commands.deploy.deploy_to_cloud_run",
            return_value="https://service-url.run.app",
        ) as mock_deploy_cr,
        patch(
            "eval_protocol.cli_commands.deploy.create_evaluation",
            return_value={"name": evaluator_id},
        ) as mock_create_eval,
    ):
        # Configure mock_get_config to return a basic config
        mock_config_instance = RewardKitConfig(
            gcp_cloud_run=GCPCloudRunConfig(
                project_id="test-gcp-project-yaml",  # Test CLI override
                region="us-west1-yaml",  # Test CLI override
                default_auth_mode="api-key",
            ),
            evaluator_endpoint_keys={},
        )
        mock_get_config.return_value = mock_config_instance

        # Call the deploy command
        result_code = deploy_command(args)

        assert result_code == 0
        mock_check_env.assert_called_once()

        # Key assertion: generate_dockerfile_content was called with the correct inline_requirements_content
        mock_gen_dockerfile.assert_called_once()
        call_args, call_kwargs = mock_gen_dockerfile.call_args
        assert call_kwargs.get("function_ref") == function_ref
        assert call_kwargs.get("inline_requirements_content") == DUMMY_DEPLOY_REQUIREMENTS
        assert call_kwargs.get("user_requirements_path") is None  # Ensure it's not trying to use both

        mock_ensure_ar.assert_called_once_with(
            project_id=args.gcp_project,
            region=args.gcp_region,
            repo_name="eval-protocol-evaluators",  # Default repo name
        )
        mock_build_push.assert_called_once()
        mock_deploy_cr.assert_called_once()
        mock_create_eval.assert_called_once()

        # Check that the dynamically loaded module's requirements were used
        # This is implicitly tested by checking mock_gen_dockerfile's call_kwargs

    # Ensure the dummy module is cleaned up by the fixture
    # No explicit cleanup needed here due to yield in fixture
