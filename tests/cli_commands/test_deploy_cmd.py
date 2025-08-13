import json
from unittest.mock import MagicMock, patch

import pytest

# Module to be tested
from eval_protocol.cli_commands.deploy import deploy_command
from eval_protocol.platform_api import PlatformAPIError  # Import for exception testing


# --- Mocking argparse.Namespace to simulate parsed CLI arguments ---
class MockArgs:
    def __init__(self, **kwargs):
        self.verbose = False
        self.id = None
        self.metrics_folders = None
        self.display_name = None
        self.description = None
        self.force = False
        self.huggingface_dataset = None
        self.huggingface_split = "train"
        self.huggingface_prompt_key = "prompt"
        self.huggingface_response_key = "response"
        self.huggingface_key_map = None
        self.remote_url = None
        # For GCP
        self.target = "fireworks"  # Default target
        self.function_ref = None
        self.gcp_project = None
        self.gcp_region = None
        self.gcp_ar_repo = None
        self.service_account = None
        self.entry_point = "reward_function"
        self.runtime = "python311"
        self.gcp_auth_mode = None
        self.__dict__.update(kwargs)


@pytest.fixture
def mock_check_environment():
    with patch("eval_protocol.cli_commands.deploy.check_environment", return_value=True) as mock_check:
        yield mock_check


@pytest.fixture
def mock_gcp_tools():
    with (
        patch("eval_protocol.cli_commands.deploy.ensure_artifact_registry_repo_exists") as mock_ensure_repo,
        patch("eval_protocol.cli_commands.deploy.generate_dockerfile_content") as mock_gen_dockerfile,
        patch("eval_protocol.cli_commands.deploy.build_and_push_docker_image") as mock_build_push,
        patch("eval_protocol.cli_commands.deploy.deploy_to_cloud_run") as mock_deploy_run,
        patch("eval_protocol.cli_commands.deploy.ensure_gcp_secret") as mock_ensure_gcp_secret,
    ):
        mock_ensure_repo.return_value = True
        mock_gen_dockerfile.return_value = "DOCKERFILE CONTENT"
        mock_build_push.return_value = True
        mock_deploy_run.return_value = "http://mock-cloud-run-url.com/service"
        mock_ensure_gcp_secret.return_value = "projects/test-proj/secrets/mocksecret/versions/1"
        yield {
            "ensure_repo": mock_ensure_repo,
            "gen_dockerfile": mock_gen_dockerfile,
            "build_push": mock_build_push,
            "deploy_run": mock_deploy_run,
            "ensure_gcp_secret": mock_ensure_gcp_secret,
        }


class TestDeployCommandRemoteUrl:
    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_remote_url_success(self, mock_create_evaluation_call, mock_check_environment, capsys):
        """Test successful registration of a remote URL via create_evaluation."""
        args = MockArgs(
            id="my-remote-eval",
            remote_url="http://my-evaluator.com/evaluate",
            display_name="My Remote Eval",
            description="A cool remote evaluator.",
            target="fireworks",  # Explicitly set target for this path
        )
        mock_create_evaluation_call.return_value = {
            "name": args.id,  # Simulate platform API returning full name
            "id": args.id,  # Simulate platform API returning id
        }

        return_code = deploy_command(args)
        assert return_code == 0

        mock_create_evaluation_call.assert_called_once_with(
            evaluator_id=args.id,
            remote_url=args.remote_url,
            display_name=args.display_name or args.id,
            description=args.description
            or f"Evaluator for {args.id} at {args.remote_url}",  # Updated description format
            force=args.force,
            huggingface_dataset=args.huggingface_dataset,
            huggingface_split=args.huggingface_split,
            huggingface_message_key_map=None,
            huggingface_prompt_key=args.huggingface_prompt_key,
            huggingface_response_key=args.huggingface_response_key,
        )

        captured = capsys.readouterr()
        assert (
            f"Registering remote URL: {args.remote_url} for evaluator '{args.id}'"  # Updated initial message
            in captured.out
        )
        assert (
            f"Successfully registered evaluator '{args.id}' on Fireworks AI, pointing to '{args.remote_url}'."  # Updated success message
            in captured.out
        )

    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_remote_url_with_metrics_folders_warning(self, mock_create_eval, mock_check_environment, capsys):
        args = MockArgs(
            id="my-remote-eval",
            remote_url="http://my-evaluator.com/evaluate",
            metrics_folders=["mf=path"],
            target="fireworks",  # Explicitly set target
        )
        mock_create_eval.return_value = {"name": args.id}
        deploy_command(args)
        captured = capsys.readouterr()
        assert (
            "Info: --metrics-folders are ignored when deploying with --remote-url."  # Updated "not packaged" to "ignored"
            in captured.out
        )

    def test_deploy_remote_url_invalid_url_format(self, mock_check_environment, capsys):
        args = MockArgs(id="my-eval", remote_url="ftp://invalid.com", target="fireworks")
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error: Invalid --remote-url 'ftp://invalid.com'" in captured.out

    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_remote_url_platform_api_error(self, mock_create_eval, mock_check_environment, capsys):
        args = MockArgs(
            id="my-remote-eval-fail",
            remote_url="http://my-evaluator.com/evaluate",
            target="fireworks",
        )
        # Simulate the full error string from PlatformAPIError's __str__
        error_message = "Platform connection failed (Status: 500, Response: N/A)"
        mock_create_eval.side_effect = PlatformAPIError(
            "Platform connection failed", status_code=500, response_text="N/A"
        )

        return_code = deploy_command(args)
        assert return_code == 1

        captured = capsys.readouterr()
        # Updated error message to match common registration block
        assert f"Error registering URL with Fireworks AI: {error_message}" in captured.out

    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_remote_url_unexpected_error(self, mock_create_eval, mock_check_environment, capsys):
        args = MockArgs(
            id="my-remote-eval-generic-fail",
            remote_url="http://my-evaluator.com/evaluate",
            target="fireworks",
        )
        mock_create_eval.side_effect = Exception("Something broke")

        return_code = deploy_command(args)
        assert return_code == 1

        captured = capsys.readouterr()
        # Updated error message to match common registration block
        assert "An unexpected error occurred during Fireworks AI registration: Something broke" in captured.out


class TestDeployCommandLocalMode:  # This class tests the "fireworks" target (packaging metrics)
    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_local_mode_success(  # Renaming to reflect it tests "fireworks" target
        self, mock_create_eval, mock_check_environment, capsys
    ):
        mock_create_eval.return_value = {"name": "my-fireworks-eval"}  # Adjusted for clarity
        args = MockArgs(
            id="my-fireworks-eval",
            metrics_folders=["mf=./path"],
            display_name="My Fireworks Eval",
            description="A packaged one.",
            target="fireworks",  # Explicitly "fireworks" target
        )
        return_code = deploy_command(args)
        assert return_code == 0
        expected_hf_message_key_map = None
        mock_create_eval.assert_called_once_with(
            evaluator_id=args.id,
            metric_folders=args.metrics_folders,
            display_name=args.display_name or args.id,
            description=args.description or f"Evaluator: {args.id}",
            force=args.force,
            huggingface_dataset=args.huggingface_dataset,
            huggingface_split=args.huggingface_split,
            huggingface_message_key_map=expected_hf_message_key_map,
            huggingface_prompt_key=args.huggingface_prompt_key,
            huggingface_response_key=args.huggingface_response_key,
        )
        captured = capsys.readouterr()
        assert "Packaging and deploying metrics for evaluator 'my-fireworks-eval' to Fireworks AI..." in captured.out
        assert "Successfully created/updated evaluator: my-fireworks-eval" in captured.out

    def test_deploy_local_mode_missing_metrics_folders(  # Renaming to reflect "fireworks" target
        self, mock_check_environment, capsys
    ):
        args = MockArgs(
            id="my-fireworks-eval-fail", target="fireworks", remote_url=None
        )  # Explicit target, no remote_url
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        # Updated error message to be specific to "fireworks" target
        assert (
            "Error: --metrics-folders are required for 'fireworks' target if --remote-url is not provided."
            in captured.out
        )

    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_local_mode_create_evaluation_fails(  # Renaming
        self, mock_create_eval, mock_check_environment, capsys
    ):
        error_message = "Platform API error (Status: 503, Response: N/A)"
        mock_create_eval.side_effect = PlatformAPIError("Platform API error", status_code=503, response_text="N/A")
        args = MockArgs(id="my-fireworks-eval", metrics_folders=["mf=./path"], target="fireworks")
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert f"Error creating/updating evaluator 'my-fireworks-eval': {error_message}" in captured.out

    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_local_mode_create_evaluation_fails_generic_exception(  # Renaming
        self, mock_create_eval, mock_check_environment, capsys
    ):
        mock_create_eval.side_effect = Exception("Generic error")
        args = MockArgs(id="my-fireworks-eval", metrics_folders=["mf=./path"], target="fireworks")
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error creating/updating evaluator 'my-fireworks-eval': Generic error" in captured.out


class TestDeployCommandGCPMode:
    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_gcp_mode_success(
        self,
        mock_create_evaluation_final_step,
        mock_check_environment,
        mock_gcp_tools,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval-test",
            function_ref="my_module.my_func",
            gcp_project="test-proj",
            gcp_region="us-central1",
            gcp_ar_repo="test-repo",
            runtime="python310",
            gcp_auth_mode="api-key",
        )
        mock_create_evaluation_final_step.return_value = {"name": args.id}  # Simulate platform API returning full name

        return_code = deploy_command(args)
        assert return_code == 0

        mock_gcp_tools["ensure_repo"].assert_called_once()
        mock_gcp_tools["gen_dockerfile"].assert_called_once()
        mock_gcp_tools["build_push"].assert_called_once()
        mock_gcp_tools["ensure_gcp_secret"].assert_called_once()
        mock_gcp_tools["deploy_run"].assert_called_once()
        mock_create_evaluation_final_step.assert_called_once()

        captured = capsys.readouterr()
        # Check initial message from helper
        assert f"Starting GCP Cloud Run deployment for evaluator '{args.id}'..." in captured.out
        assert "Successfully built and pushed Docker image" in captured.out
        assert (
            f"Successfully deployed to Cloud Run. Service URL: {mock_gcp_tools['deploy_run'].return_value}"
            in captured.out
        )
        # Check common registration success message
        assert (
            f"Successfully registered evaluator '{args.id}' on Fireworks AI, pointing to '{mock_gcp_tools['deploy_run'].return_value}'."
            in captured.out
        )

    @patch("eval_protocol.cli_commands.deploy.get_config")
    def test_deploy_gcp_mode_missing_args(self, mock_get_config, mock_check_environment, capsys):
        # Mock empty config to test missing project/region scenarios
        from eval_protocol.config import RewardKitConfig

        mock_get_config.return_value = RewardKitConfig()

        args = MockArgs(target="gcp-cloud-run", id="gcp-eval-incomplete")
        # function_ref is missing, gcp_project, gcp_region also

        # Test missing function_ref
        temp_args_dict = args.__dict__.copy()
        temp_args_dict.pop("function_ref", None)
        current_args = MockArgs(**temp_args_dict)
        return_code = deploy_command(current_args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error: --function-ref is required for GCP Cloud Run deployment." in captured.out

        # Test missing gcp_project
        temp_args_dict = args.__dict__.copy()
        temp_args_dict["function_ref"] = "a.b"
        temp_args_dict.pop("gcp_project", None)
        current_args = MockArgs(**temp_args_dict)
        return_code = deploy_command(current_args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error: GCP Project ID must be provided" in captured.out

        # Test missing gcp_region
        temp_args_dict["gcp_project"] = "proj"
        temp_args_dict.pop("gcp_region", None)
        current_args = MockArgs(**temp_args_dict)
        return_code = deploy_command(current_args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error: GCP Region must be provided" in captured.out

    @patch(
        "eval_protocol.cli_commands.deploy.ensure_artifact_registry_repo_exists",
        return_value=False,
    )
    def test_deploy_gcp_mode_ensure_repo_fails(self, mock_ensure_repo_fails, mock_check_environment, capsys):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Failed to ensure Artifact Registry repository" in captured.out

    @patch(
        "eval_protocol.cli_commands.deploy.ensure_artifact_registry_repo_exists",
        return_value=True,
    )
    @patch(
        "eval_protocol.cli_commands.deploy.generate_dockerfile_content",
        return_value=None,
    )
    def test_deploy_gcp_mode_gen_dockerfile_fails(
        self,
        mock_gen_dockerfile_fails,
        mock_ensure_repo,
        mock_check_environment,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Failed to generate Dockerfile content. Aborting." in captured.out

    @patch(
        "eval_protocol.cli_commands.deploy.ensure_artifact_registry_repo_exists",
        return_value=True,
    )
    @patch(
        "eval_protocol.cli_commands.deploy.generate_dockerfile_content",
        return_value="Dockerfile",
    )
    @patch(
        "eval_protocol.cli_commands.deploy.build_and_push_docker_image",
        return_value=False,
    )
    def test_deploy_gcp_mode_build_fails(
        self,
        mock_build_fails,
        mock_gen_dockerfile,
        mock_ensure_repo,
        mock_check_environment,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Failed to build and push Docker image" in captured.out

    @patch(
        "eval_protocol.cli_commands.deploy.ensure_artifact_registry_repo_exists",
        return_value=True,
    )
    @patch(
        "eval_protocol.cli_commands.deploy.generate_dockerfile_content",
        return_value="Dockerfile",
    )
    @patch(
        "eval_protocol.cli_commands.deploy.build_and_push_docker_image",
        return_value=True,
    )
    @patch("eval_protocol.cli_commands.deploy.deploy_to_cloud_run", return_value=None)
    @patch(
        "eval_protocol.cli_commands.deploy.ensure_gcp_secret",
        return_value="projects/p/secrets/mocksecret/versions/1",
    )
    def test_deploy_gcp_mode_cloud_run_deploy_fails(
        self,
        mock_ensure_gcp_secret_individual,
        mock_deploy_run_fails,
        mock_build_push,
        mock_gen_dockerfile,
        mock_ensure_repo,
        mock_check_environment,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
            gcp_auth_mode="api-key",
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Failed to deploy to Cloud Run or retrieve service URL. Aborting." in captured.out

    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    @patch(
        "eval_protocol.cli_commands.deploy.ensure_gcp_secret",
        return_value="projects/p/secrets/mocksecret/versions/1",
    )
    def test_deploy_gcp_mode_final_registration_fails_platform_error(
        self,
        mock_ensure_gcp_secret_individual,
        mock_create_evaluation_final_step,
        mock_check_environment,
        mock_gcp_tools,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval-reg-fail",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
            gcp_auth_mode="api-key",
        )
        error_message = "Registration failed (Status: 400, Response: N/A)"
        mock_create_evaluation_final_step.side_effect = PlatformAPIError(
            "Registration failed", status_code=400, response_text="N/A"
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        # Updated error message to match common registration block
        assert f"Error registering URL with Fireworks AI: {error_message}" in captured.out

    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    @patch(
        "eval_protocol.cli_commands.deploy.ensure_gcp_secret",
        return_value="projects/p/secrets/mocksecret/versions/1",
    )
    def test_deploy_gcp_mode_final_registration_fails_generic_error(
        self,
        mock_ensure_gcp_secret_individual,
        mock_create_evaluation_final_step,
        mock_check_environment,
        mock_gcp_tools,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval-reg-fail-gen",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
            gcp_auth_mode="api-key",
        )
        mock_create_evaluation_final_step.side_effect = Exception("Unexpected registration issue")
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        # Updated error message to match common registration block
        assert (
            "An unexpected error occurred during Fireworks AI registration: Unexpected registration issue"
            in captured.out
        )
