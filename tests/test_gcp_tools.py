import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from eval_protocol.gcp_tools import (
    _run_gcloud_command,
    build_and_push_docker_image,
    deploy_to_cloud_run,
    ensure_artifact_registry_repo_exists,
    ensure_gcp_secret,
    logger,
)


class TestGCPTools(unittest.TestCase):
    def setUp(self):
        self.mock_project_id = "test-project"
        self.mock_region = "us-central1"
        self.mock_image_name_tag = f"gcr.io/{self.mock_project_id}/test-image:latest"
        self.mock_service_name = "test-service"
        self.mock_secret_id = "test-secret"
        self.mock_secret_value = "supersecret"
        self.mock_repo_name = "test-repo"

    @patch("eval_protocol.gcp_tools.subprocess.run")
    def test_run_gcloud_command_success(self, mock_subprocess_run):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Success!"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        success, stdout, stderr = _run_gcloud_command(["info"])

        self.assertTrue(success)
        self.assertEqual(stdout, "Success!")
        self.assertEqual(stderr, "")
        mock_subprocess_run.assert_called_once_with(["gcloud", "info"], capture_output=True, text=True, check=False)

    @patch("eval_protocol.gcp_tools.subprocess.run")
    def test_run_gcloud_command_failure(self, mock_subprocess_run):
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Error!"
        mock_subprocess_run.return_value = mock_process

        success, stdout, stderr = _run_gcloud_command(["info"])

        self.assertFalse(success)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "Error!")

    @patch("eval_protocol.gcp_tools.subprocess.run")
    def test_run_gcloud_command_file_not_found(self, mock_subprocess_run):
        mock_subprocess_run.side_effect = FileNotFoundError

        success, stdout, stderr = _run_gcloud_command(["info"])

        self.assertFalse(success)
        self.assertEqual(stderr, "gcloud command not found.")

    def test_run_gcloud_command_dry_run(self):
        command = ["projects", "list"]
        success, stdout, stderr = _run_gcloud_command(command, dry_run=True)

        self.assertTrue(success)
        self.assertEqual(stdout, "Dry run: gcloud projects list")
        self.assertEqual(stderr, "")

    @patch("eval_protocol.gcp_tools.subprocess.run")
    def test_run_gcloud_command_success_with_stderr(self, mock_subprocess_run):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Main output"
        mock_process.stderr = "Informational message on stderr"
        mock_subprocess_run.return_value = mock_process

        success, stdout, stderr = _run_gcloud_command(["some", "command"])

        self.assertTrue(success)
        self.assertEqual(stdout, "Main output")
        self.assertEqual(stderr, "Informational message on stderr")
        mock_subprocess_run.assert_called_once()

    @patch("eval_protocol.gcp_tools.Path", autospec=True)
    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    @patch(
        "eval_protocol.gcp_tools.os.remove"
    )  # os.path.exists mock removed as not directly used by SUT for this path
    @patch("builtins.open", new_callable=mock_open)
    def test_build_and_push_docker_image_success(self, mock_open_file, mock_os_remove, mock_run_gcloud, MockGCPPath):
        mock_path_instance = MockGCPPath.return_value

        mock_dockerfile_path_obj = MagicMock(spec=Path)
        mock_dockerfile_path_obj.exists.return_value = True  # For the finally block check
        # Make the mock object usable by open() and os.remove() by giving it a name attribute or ensuring it's Path-like
        # For open(), it will be passed directly. For os.remove(), it will be passed directly.
        # Let's ensure it has a proper string representation for logging if any, though not strictly needed for mocks.
        mock_dockerfile_path_obj.__str__.return_value = str(Path(".") / "Dockerfile")

        mock_path_instance.__truediv__.return_value = mock_dockerfile_path_obj

        mock_run_gcloud.return_value = (True, "Build success", "")

        dockerfile_content = "FROM ubuntu"
        build_context_dir_str = "."

        result = build_and_push_docker_image(
            self.mock_image_name_tag,
            dockerfile_content,
            build_context_dir_str,
            self.mock_project_id,
        )

        self.assertTrue(result)
        MockGCPPath.assert_called_once_with(build_context_dir_str)
        mock_path_instance.__truediv__.assert_called_once_with("Dockerfile")

        mock_dockerfile_path_obj.exists.assert_called_once_with()

        mock_open_file.assert_called_once_with(mock_dockerfile_path_obj, "w")
        mock_open_file().write.assert_called_once_with(dockerfile_content)
        mock_run_gcloud.assert_called_once_with(
            [
                "builds",
                "submit",
                build_context_dir_str,
                "--tag",
                self.mock_image_name_tag,
                "--project",
                self.mock_project_id,
            ],
            dry_run=False,
        )
        mock_os_remove.assert_called_once_with(mock_dockerfile_path_obj)

    @patch("eval_protocol.gcp_tools.Path", autospec=True)
    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    @patch("eval_protocol.gcp_tools.os.remove")  # os.path.exists mock removed
    @patch("builtins.open", new_callable=mock_open)
    def test_build_and_push_docker_image_build_fails(
        self, mock_open_file, mock_os_remove, mock_run_gcloud, MockGCPPath
    ):
        mock_path_instance = MockGCPPath.return_value

        mock_dockerfile_path_obj = MagicMock(spec=Path)
        mock_dockerfile_path_obj.exists.return_value = True  # For the finally block check
        mock_dockerfile_path_obj.__str__.return_value = str(Path(".") / "Dockerfile")
        mock_path_instance.__truediv__.return_value = mock_dockerfile_path_obj

        mock_run_gcloud.return_value = (False, "", "Build failed error")

        dockerfile_content = "FROM ubuntu"
        build_context_dir_str = "."

        result = build_and_push_docker_image(
            self.mock_image_name_tag,
            dockerfile_content,
            build_context_dir_str,
            self.mock_project_id,
        )

        self.assertFalse(result)
        MockGCPPath.assert_called_once_with(build_context_dir_str)
        mock_path_instance.__truediv__.assert_called_once_with("Dockerfile")

        mock_dockerfile_path_obj.exists.assert_called_once_with()

        mock_open_file.assert_called_once_with(mock_dockerfile_path_obj, "w")
        mock_open_file().write.assert_called_once_with(dockerfile_content)
        mock_run_gcloud.assert_called_once()
        mock_os_remove.assert_called_once_with(mock_dockerfile_path_obj)

    def test_build_and_push_docker_image_no_project_id(self):
        dockerfile_content = "FROM ubuntu"
        build_context_dir_str = "."

        # logger is imported from eval_protocol.gcp_tools
        with patch.object(logger, "error") as mock_logger_error:
            result = build_and_push_docker_image(
                self.mock_image_name_tag,
                dockerfile_content,
                build_context_dir_str,
                gcp_project_id=None,  # Key part of the test
            )

        self.assertFalse(result)
        mock_logger_error.assert_called_once_with("GCP Project ID is required for Google Cloud Build.")

    @patch("eval_protocol.gcp_tools.Path", autospec=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("eval_protocol.gcp_tools.os.remove")  # Added for this test
    def test_build_and_push_docker_image_open_fails(
        self, mock_os_remove, mock_open_file, MockGCPPath
    ):  # mock_os_remove added
        mock_path_instance = MockGCPPath.return_value
        mock_dockerfile_path_obj = MagicMock(spec=Path)
        mock_dockerfile_path_obj.name = "mock_dockerfile_path_for_open_fail_test"
        mock_dockerfile_path_obj.__str__.return_value = mock_dockerfile_path_obj.name
        mock_path_instance.__truediv__.return_value = mock_dockerfile_path_obj

        mock_open_file.side_effect = IOError("Failed to open Dockerfile for writing")
        # When open fails, the file isn't created, so .exists() in finally should be False.
        mock_dockerfile_path_obj.exists.return_value = False

        dockerfile_content = "FROM ubuntu"
        build_context_dir_str = "."

        with patch.object(logger, "error") as mock_logger_error:
            result = build_and_push_docker_image(
                self.mock_image_name_tag,
                dockerfile_content,
                build_context_dir_str,
                self.mock_project_id,
            )

        self.assertFalse(result)
        MockGCPPath.assert_called_once_with(build_context_dir_str)
        mock_path_instance.__truediv__.assert_called_once_with("Dockerfile")
        mock_open_file.assert_called_once_with(mock_dockerfile_path_obj, "w")

        mock_logger_error.assert_called_once()
        logged_error_message = mock_logger_error.call_args[0][0]
        self.assertIn(
            "An error occurred during Dockerfile creation or gcloud command preparation",
            logged_error_message,
        )
        self.assertIn("Failed to open Dockerfile for writing", logged_error_message)

        # Check finally block behavior
        mock_dockerfile_path_obj.exists.assert_called_once_with()
        mock_os_remove.assert_not_called()  # Because .exists() returned False

    @patch("eval_protocol.gcp_tools.Path", autospec=True)
    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    @patch("eval_protocol.gcp_tools.os.remove")
    @patch("builtins.open", new_callable=mock_open)
    def test_build_and_push_docker_image_success_dockerfile_vanishes(
        self, mock_open_file, mock_os_remove, mock_run_gcloud, MockGCPPath
    ):
        mock_path_instance = MockGCPPath.return_value
        mock_dockerfile_path_obj = MagicMock(spec=Path)
        mock_dockerfile_path_obj.name = "mock_dockerfile_path_vanishes_test"
        mock_dockerfile_path_obj.__str__.return_value = mock_dockerfile_path_obj.name
        mock_path_instance.__truediv__.return_value = mock_dockerfile_path_obj

        # Key difference: Dockerfile doesn't exist when finally block checks
        mock_dockerfile_path_obj.exists.return_value = False

        mock_run_gcloud.return_value = (True, "Build success", "")

        dockerfile_content = "FROM ubuntu"
        build_context_dir_str = "."

        result = build_and_push_docker_image(
            self.mock_image_name_tag,
            dockerfile_content,
            build_context_dir_str,
            self.mock_project_id,
        )

        self.assertTrue(result)  # Build itself should succeed
        MockGCPPath.assert_called_once_with(build_context_dir_str)
        mock_path_instance.__truediv__.assert_called_once_with("Dockerfile")

        mock_open_file.assert_called_once_with(mock_dockerfile_path_obj, "w")
        mock_open_file().write.assert_called_once_with(dockerfile_content)
        mock_run_gcloud.assert_called_once()

        # Check finally block behavior
        mock_dockerfile_path_obj.exists.assert_called_once_with()
        mock_os_remove.assert_not_called()  # Because .exists() returned False

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    def test_deploy_to_cloud_run_success(self, mock_run_gcloud):
        # Mock for deploy command
        mock_run_gcloud.side_effect = [
            (True, "Deploy success", ""),  # For initial deploy
            (
                True,
                f"https://{self.mock_service_name}-mock-url.a.run.app",
                "",
            ),  # For describe to get URL
        ]

        result_url = deploy_to_cloud_run(
            self.mock_service_name,
            self.mock_image_name_tag,
            self.mock_project_id,
            self.mock_region,
            env_vars={"KEY": "VALUE"},
            secrets_to_mount={"MY_SECRET": f"projects/{self.mock_project_id}/secrets/my-secret-id/versions/latest"},
        )

        self.assertEqual(result_url, f"https://{self.mock_service_name}-mock-url.a.run.app")
        self.assertEqual(mock_run_gcloud.call_count, 2)
        deploy_call_args = mock_run_gcloud.call_args_list[0][0][0]
        self.assertIn("--set-env-vars", deploy_call_args)
        self.assertIn("KEY=VALUE", deploy_call_args)
        self.assertIn("--update-secrets", deploy_call_args)
        self.assertIn("MY_SECRET=my-secret-id:latest", deploy_call_args)

    def test_deploy_to_cloud_run_no_project_id(self):
        with patch.object(logger, "error") as mock_logger_error:
            result_url = deploy_to_cloud_run(
                self.mock_service_name,
                self.mock_image_name_tag,
                gcp_project_id=None,  # Key
                gcp_region=self.mock_region,
            )
        self.assertIsNone(result_url)
        mock_logger_error.assert_called_once_with("GCP Project ID is required for deploying to Cloud Run.")

    def test_deploy_to_cloud_run_no_region(self):
        with patch.object(logger, "error") as mock_logger_error:
            result_url = deploy_to_cloud_run(
                self.mock_service_name,
                self.mock_image_name_tag,
                gcp_project_id=self.mock_project_id,
                gcp_region=None,  # Key
            )
        self.assertIsNone(result_url)
        mock_logger_error.assert_called_once_with("GCP Region is required for deploying to Cloud Run.")

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    def test_deploy_to_cloud_run_deploy_fails(self, mock_run_gcloud):
        mock_run_gcloud.return_value = (
            False,
            "",
            "Deploy command failed",
        )  # First call fails

        with patch.object(logger, "error") as mock_logger_error:
            result_url = deploy_to_cloud_run(
                self.mock_service_name,
                self.mock_image_name_tag,
                self.mock_project_id,
                self.mock_region,
            )

        self.assertIsNone(result_url)
        mock_run_gcloud.assert_called_once()  # Only deploy command should be called
        mock_logger_error.assert_called_once()
        self.assertIn(
            f"Failed to deploy service {self.mock_service_name}",
            mock_logger_error.call_args[0][0],
        )

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    def test_deploy_to_cloud_run_describe_fails(self, mock_run_gcloud):
        mock_run_gcloud.side_effect = [
            (True, "Deploy success", ""),  # First call (deploy) succeeds
            (False, "", "Describe command failed"),  # Second call (describe) fails
        ]

        with patch.object(logger, "error") as mock_logger_error:
            result_url = deploy_to_cloud_run(
                self.mock_service_name,
                self.mock_image_name_tag,
                self.mock_project_id,
                self.mock_region,
            )

        self.assertIsNone(result_url)
        self.assertEqual(mock_run_gcloud.call_count, 2)
        mock_logger_error.assert_called_once()
        self.assertEqual(
            mock_logger_error.call_args[0][0],
            f"Deployed service {self.mock_service_name}, but failed to retrieve its URL. Stderr: Describe command failed",
        )

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    def test_deploy_to_cloud_run_bad_url(self, mock_run_gcloud):
        mock_run_gcloud.side_effect = [
            (True, "Deploy success", ""),
            (True, "http://bad-url-no-https.a.run.app", ""),  # Describe returns bad URL
        ]

        with patch.object(logger, "error") as mock_logger_error:
            result_url = deploy_to_cloud_run(
                self.mock_service_name,
                self.mock_image_name_tag,
                self.mock_project_id,
                self.mock_region,
            )

        self.assertIsNone(result_url)
        self.assertEqual(mock_run_gcloud.call_count, 2)
        mock_logger_error.assert_called_once()
        self.assertIn("Service URL is not valid", mock_logger_error.call_args[0][0])

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    def test_deploy_to_cloud_run_gcloud_exception(self, mock_run_gcloud):
        mock_run_gcloud.side_effect = Exception("gcloud exploded")

        with patch.object(logger, "error") as mock_logger_error:
            result_url = deploy_to_cloud_run(
                self.mock_service_name,
                self.mock_image_name_tag,
                self.mock_project_id,
                self.mock_region,
            )

        self.assertIsNone(result_url)
        mock_run_gcloud.assert_called_once()  # Should fail on the first call
        mock_logger_error.assert_called_once()
        logged_error_message = mock_logger_error.call_args[0][0]
        self.assertIn("An error occurred during Cloud Run deployment", logged_error_message)
        self.assertIn("gcloud exploded", logged_error_message)

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    def test_ensure_artifact_registry_repo_exists_already_exists(self, mock_run_gcloud):
        mock_run_gcloud.return_value = (True, "Repo exists", "")  # Describe success

        result = ensure_artifact_registry_repo_exists(self.mock_project_id, self.mock_region, self.mock_repo_name)
        self.assertTrue(result)
        mock_run_gcloud.assert_called_once_with(
            [
                "artifacts",
                "repositories",
                "describe",
                self.mock_repo_name,
                "--project",
                self.mock_project_id,
                "--location",
                self.mock_region,
            ],
            dry_run=False,
        )

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    def test_ensure_artifact_registry_repo_exists_create_new(self, mock_run_gcloud):
        mock_run_gcloud.side_effect = [
            (False, "", "NOT_FOUND"),  # Describe fails (not found)
            (True, "Create success", ""),  # Create success
        ]

        result = ensure_artifact_registry_repo_exists(self.mock_project_id, self.mock_region, self.mock_repo_name)
        self.assertTrue(result)
        self.assertEqual(mock_run_gcloud.call_count, 2)
        create_call_args = mock_run_gcloud.call_args_list[1][0][0]
        self.assertIn("create", create_call_args)
        self.assertIn(self.mock_repo_name, create_call_args)

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    def test_ensure_artifact_registry_repo_exists_exception(self, mock_run_gcloud):
        mock_run_gcloud.side_effect = Exception("Kaboom")
        with patch.object(logger, "error") as mock_logger_error:
            result = ensure_artifact_registry_repo_exists(self.mock_project_id, self.mock_region, self.mock_repo_name)
        self.assertFalse(result)
        mock_run_gcloud.assert_called_once()  # Describe should be called
        mock_logger_error.assert_called_once_with(
            f"An unexpected error occurred while ensuring Artifact Registry repository '{self.mock_repo_name}': Kaboom"
        )

    def test_ensure_gcp_secret_no_project_id(self):
        with patch.object(logger, "error") as mock_logger_error:
            result_version = ensure_gcp_secret(
                project_id=None,  # Changed here
                secret_id=self.mock_secret_id,
                secret_value=self.mock_secret_value,
            )
        self.assertIsNone(result_version)
        mock_logger_error.assert_called_once_with("GCP Project ID is required to manage secrets.")

    def test_ensure_gcp_secret_no_secret_id(self):
        with patch.object(logger, "error") as mock_logger_error:
            result_version = ensure_gcp_secret(
                project_id=self.mock_project_id,  # Changed here
                secret_id=None,
                secret_value=self.mock_secret_value,
            )
        self.assertIsNone(result_version)
        mock_logger_error.assert_called_once_with("Secret ID is required to manage secrets.")

    def test_ensure_gcp_secret_no_secret_value(self):
        with patch.object(logger, "error") as mock_logger_error:
            result_version = ensure_gcp_secret(
                project_id=self.mock_project_id,  # Changed here
                secret_id=self.mock_secret_id,
                secret_value=None,
            )
        self.assertIsNone(result_version)
        mock_logger_error.assert_called_once_with("Secret value is required to create or update a secret.")

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    @patch("eval_protocol.gcp_tools.tempfile.NamedTemporaryFile")
    @patch("eval_protocol.gcp_tools.os.remove")
    @patch(
        "eval_protocol.gcp_tools.os.path.exists"
    )  # Changed from Path.exists to os.path.exists as used in gcp_tools.py
    def test_ensure_gcp_secret_create_and_add_version(
        self, mock_os_path_exists, mock_os_remove, mock_named_temp_file, mock_run_gcloud
    ):
        # Setup mock for NamedTemporaryFile
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.write = MagicMock()
        mock_temp_file_instance.name = "mock_temp_file_path"
        # __enter__ should return the mock instance itself for context manager
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance

        mock_run_gcloud.side_effect = [
            (False, "", "NOT_FOUND"),  # Describe secret fails (not found)
            (True, "Create secret success", ""),  # Create secret success
            (
                True,
                f"projects/{self.mock_project_id}/secrets/{self.mock_secret_id}/versions/1",
                "",
            ),  # Add version success
            (
                True,
                f"projects/{self.mock_project_id}/secrets/{self.mock_secret_id}/versions/1",
                "",
            ),  # Describe latest version success
        ]
        # mock_os_path_exists for os.remove(tmp_secret_file_path)
        mock_os_path_exists.return_value = True

        result_secret_version = ensure_gcp_secret(self.mock_project_id, self.mock_secret_id, self.mock_secret_value)

        self.assertEqual(
            result_secret_version,
            f"projects/{self.mock_project_id}/secrets/{self.mock_secret_id}/versions/1",
        )
        self.assertEqual(
            mock_run_gcloud.call_count, 4
        )  # Describe secret (fail), Create secret, Add version, Describe version
        mock_named_temp_file.assert_called_once_with(mode="w", delete=False)
        mock_temp_file_instance.write.assert_called_once_with(self.mock_secret_value)
        # Ensure os.remove is called with the name attribute of the mock temp file
        mock_os_remove.assert_called_once_with("mock_temp_file_path")
        mock_os_path_exists.assert_called_with("mock_temp_file_path")

    @patch("eval_protocol.gcp_tools._run_gcloud_command")
    @patch("eval_protocol.gcp_tools.tempfile.NamedTemporaryFile")
    @patch("eval_protocol.gcp_tools.os.remove")
    @patch("eval_protocol.gcp_tools.os.path.exists")  # Changed from Path.exists
    def test_ensure_gcp_secret_add_version_to_existing(
        self, mock_os_path_exists, mock_os_remove, mock_named_temp_file, mock_run_gcloud
    ):
        # Setup mock for NamedTemporaryFile
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.write = MagicMock()
        mock_temp_file_instance.name = "mock_temp_file_path_existing"  # Different name to avoid collision
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance

        mock_run_gcloud.side_effect = [
            (True, "Describe success", ""),  # Describe secret success (exists)
            (
                True,
                f"projects/{self.mock_project_id}/secrets/{self.mock_secret_id}/versions/2",
                "",
            ),  # Add version success
            (
                True,
                f"projects/{self.mock_project_id}/secrets/{self.mock_secret_id}/versions/2",
                "",
            ),  # Describe latest version success
        ]
        mock_os_path_exists.return_value = True  # For os.remove

        result_secret_version = ensure_gcp_secret(self.mock_project_id, self.mock_secret_id, self.mock_secret_value)

        self.assertEqual(
            result_secret_version,
            f"projects/{self.mock_project_id}/secrets/{self.mock_secret_id}/versions/2",
        )
        self.assertEqual(mock_run_gcloud.call_count, 3)  # Describe secret, Add version, Describe version
        mock_named_temp_file.assert_called_once_with(mode="w", delete=False)
        mock_temp_file_instance.write.assert_called_once_with(self.mock_secret_value)
        mock_os_remove.assert_called_once_with("mock_temp_file_path_existing")
        mock_os_path_exists.assert_called_with("mock_temp_file_path_existing")


if __name__ == "__main__":
    unittest.main()
