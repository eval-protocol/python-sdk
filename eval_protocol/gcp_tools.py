import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _run_gcloud_command(command: List[str], dry_run: bool = False) -> Tuple[bool, str, str]:
    """
    Helper to run a gcloud command.
    In a real scenario, this would interact with subprocess.
    Returns: (success_status, stdout, stderr)
    """
    command_str_for_print = " ".join(["gcloud"] + command)
    logger.info(f"Executing: {command_str_for_print}")
    if dry_run:
        logger.info(f"Dry run mode. Command not executed: {command_str_for_print}")
        return True, f"Dry run: {command_str_for_print}", ""

    try:
        process = subprocess.run(
            ["gcloud"] + command,
            capture_output=True,
            text=True,
            check=False,  # Handle non-zero exit codes manually
        )
        if process.returncode == 0:
            if process.stderr:  # gcloud sometimes prints informational messages to stderr on success
                logger.info(f"Command successful with stderr output:\n{process.stderr}")
            return True, process.stdout.strip(), process.stderr.strip()
        else:
            logger.error(f"Command failed. Return code: {process.returncode}")
            logger.error(f"Stdout:\n{process.stdout}")
            logger.error(f"Stderr:\n{process.stderr}")
            return False, process.stdout.strip(), process.stderr.strip()
    except FileNotFoundError:
        logger.error("gcloud command not found. Is it installed and in PATH?")
        return False, "", "gcloud command not found."
    except Exception as e:
        logger.error(f"An unexpected error occurred while running gcloud command: {e}")
        return False, "", str(e)


def build_and_push_docker_image(
    image_name_tag: str,  # e.g., gcr.io/my-project/my-reward-func:latest
    dockerfile_content: str,
    build_context_dir: str,  # Directory where Dockerfile and user code are (usually CWD)
    gcp_project_id: Optional[str] = None,  # Required if using gcloud builds submit without local Docker
    dry_run: bool = False,
) -> bool:
    """
    Builds a Docker image using the provided Dockerfile content and pushes it to a registry (e.g., GCR, Artifact Registry).
    Can use local Docker or 'gcloud builds submit'.

    Args:
        image_name_tag: Full name and tag for the image (e.g., "gcr.io/project-id/image-name:tag").
        dockerfile_content: String content of the Dockerfile.
        build_context_dir: The build context directory for Docker.
        gcp_project_id: GCP Project ID, used for 'gcloud builds submit'.
        dry_run: If True, prints commands instead of executing them.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Attempting to build and push Docker image using Google Cloud Build: {image_name_tag}")

    if not gcp_project_id:
        logger.error("GCP Project ID is required for Google Cloud Build.")
        return False

    # Create Dockerfile in the build_context_dir. It must be named "Dockerfile".
    dockerfile_path_in_context = Path(build_context_dir) / "Dockerfile"

    try:
        with open(dockerfile_path_in_context, "w") as f:
            f.write(dockerfile_content)
        logger.info(f"Dockerfile created at: {dockerfile_path_in_context}")

        # Command for gcloud builds submit
        # The build_context_dir (e.g., ".") is where gcloud looks for the Dockerfile and other source files.
        build_cmd_gcloud = [
            "builds",
            "submit",
            build_context_dir,  # Source code to upload (can be "." for CWD)
            "--tag",
            image_name_tag,
            "--project",
            gcp_project_id,
        ]

        success, stdout, stderr = _run_gcloud_command(build_cmd_gcloud, dry_run=dry_run)

        if not success:
            logger.error(f"Google Cloud Build failed. Stdout: {stdout}, Stderr: {stderr}")
            return False

    except Exception as e:
        logger.error(f"An error occurred during Dockerfile creation or gcloud command preparation: {e}")
        return False
    finally:
        if dockerfile_path_in_context.exists():
            os.remove(dockerfile_path_in_context)
            logger.info(f"Temporary Dockerfile {dockerfile_path_in_context} removed.")

    if success:
        logger.info(f"Successfully built and pushed image {image_name_tag}")
    else:
        logger.error(f"Failed to build and push image {image_name_tag}")
    return success


def deploy_to_cloud_run(
    service_name: str,
    image_name_tag: str,
    gcp_project_id: str,
    gcp_region: str,
    allow_unauthenticated: bool = True,  # For --auth api-key, the service itself is open, auth is app-level
    env_vars: Optional[Dict[str, str]] = None,
    secrets_to_mount: Optional[Dict[str, str]] = None,
    service_port: int = 8080,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Deploys a container image to Google Cloud Run.

    Args:
        service_name: Name for the Cloud Run service.
        image_name_tag: Full name of the Docker image to deploy (e.g., "gcr.io/project/image:tag").
        gcp_project_id: GCP Project ID.
        gcp_region: GCP Region for the service.
        allow_unauthenticated: Whether to allow unauthenticated invocations (publicly accessible).
        env_vars: Environment variables to set for the service.
        secrets_to_mount: Secrets from GCP Secret Manager to mount as environment variables.
        service_port: Port the container exposes.
        dry_run: If True, prints commands instead of executing them.

    Returns:
        The URL of the deployed service if successful, else None.
    """

    if not gcp_project_id:
        logger.error("GCP Project ID is required for deploying to Cloud Run.")
        return None
    if not gcp_region:
        logger.error("GCP Region is required for deploying to Cloud Run.")
        return None

    try:
        logger.info(
            f"Deploying image {image_name_tag} to Cloud Run service {service_name} in {gcp_region} (Project: {gcp_project_id})"
        )

        deploy_cmd_list = [
            "run",
            "deploy",
            service_name,
            "--image",
            image_name_tag,
            "--region",
            gcp_region,
            "--project",
            gcp_project_id,
            "--port",
            str(service_port),
            # "--platform", "managed",
        ]

        if allow_unauthenticated:
            deploy_cmd_list.append("--allow-unauthenticated")
        else:
            # For IAM based auth, would be --no-allow-unauthenticated and then set IAM policy
            deploy_cmd_list.append("--no-allow-unauthenticated")
            logger.info("Note: --no-allow-unauthenticated set. Further IAM configuration might be needed.")

        if env_vars:
            env_vars_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
            deploy_cmd_list.extend(["--set-env-vars", env_vars_str])

        if secrets_to_mount:
            # Format: ENV_VAR_NAME=secret_name:version,...
            # Secret name here is just the short ID, not the full path.
            # gcloud will resolve it within the project.
            # Example: MY_API_KEY=my-api-key-secret:latest
            secrets_str_list = []
            for env_var_name, secret_manager_full_id in secrets_to_mount.items():
                # Parse projects/PROJECT_ID/secrets/SECRET_ID/versions/VERSION
                parts = secret_manager_full_id.split("/")
                if len(parts) == 6 and parts[0] == "projects" and parts[2] == "secrets" and parts[4] == "versions":
                    secret_id = parts[3]
                    secret_version = parts[5]
                    secrets_str_list.append(f"{env_var_name}={secret_id}:{secret_version}")
                else:
                    logger.warning(
                        f"Invalid secret manager full ID format: {secret_manager_full_id}. Skipping secret mount for {env_var_name}."
                    )

            if secrets_str_list:
                deploy_cmd_list.extend(["--update-secrets", ",".join(secrets_str_list)])

        success, stdout, stderr = _run_gcloud_command(deploy_cmd_list, dry_run=dry_run)

        if success:
            if dry_run:
                service_url_placeholder = f"https://{service_name}-mock-url.a.run.app"
                logger.info(
                    f"Successfully deployed service {service_name} (dry run). URL (placeholder): {service_url_placeholder}"
                )
                return service_url_placeholder

            # Get the service URL after successful deployment
            get_url_cmd = [
                "run",
                "services",
                "describe",
                service_name,
                "--region",
                gcp_region,
                "--project",
                gcp_project_id,
                "--format",
                "value(status.url)",
            ]
            url_success, url_stdout, url_stderr = _run_gcloud_command(
                get_url_cmd, dry_run=False
            )  # Always try to get URL if deploy was not dry_run

            if url_success and url_stdout:
                service_url = url_stdout.strip()
                if not service_url.startswith("https://"):
                    logger.error(f"Service URL is not valid (must be HTTPS): {service_url}")
                    return None
                logger.info(f"Successfully deployed service {service_name}. URL: {service_url}")
                return service_url
            else:
                logger.error(f"Deployed service {service_name}, but failed to retrieve its URL. Stderr: {url_stderr}")
                return None  # Consider deployment failed if URL cannot be retrieved
        else:
            logger.error(f"Failed to deploy service {service_name}. Stderr: {stderr}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during Cloud Run deployment for service {service_name}: {e}")
        return None


def ensure_artifact_registry_repo_exists(project_id: str, region: str, repo_name: str, dry_run: bool = False) -> bool:
    """
    Checks if an Artifact Registry repository exists, and creates it if it doesn't.
    """
    logger.info(
        f"Ensuring Artifact Registry repository '{repo_name}' exists in project '{project_id}', region '{region}'."
    )

    try:
        describe_cmd = [
            "artifacts",
            "repositories",
            "describe",
            repo_name,
            "--project",
            project_id,
            "--location",
            region,
        ]

        # Don't use dry_run for describe, as we need to know if it exists
        success, stdout, stderr = _run_gcloud_command(describe_cmd, dry_run=False)

        if success:
            logger.info(f"Artifact Registry repository '{repo_name}' already exists.")
            return True

        # If describe failed, check if it's because the repo was not found
        # gcloud typically returns non-zero exit code and an error message to stderr for "not found"
        if "NOT_FOUND" in stderr.upper() or "failed to find" in stderr.lower():  # Heuristic check
            logger.info(f"Artifact Registry repository '{repo_name}' not found. Attempting to create it.")
            create_cmd = [
                "artifacts",
                "repositories",
                "create",
                repo_name,
                "--project",
                project_id,
                "--repository-format",
                "docker",
                "--location",
                region,
                "--description",
                "Repository for Eval Protocol evaluators (auto-created by Eval Protocol CLI)",
            ]
            create_success, create_stdout, create_stderr = _run_gcloud_command(create_cmd, dry_run=dry_run)
            if create_success:
                logger.info(f"Successfully created Artifact Registry repository '{repo_name}'.")
                return True
            else:
                logger.error(f"Failed to create Artifact Registry repository '{repo_name}'. Stderr: {create_stderr}")
                return False
        else:
            # Describe failed for a reason other than "not found"
            logger.error(f"Error describing Artifact Registry repository '{repo_name}'. Stderr: {stderr}")
            return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while ensuring Artifact Registry repository '{repo_name}': {e}")
        return False


def ensure_gcp_secret(
    project_id: str,
    secret_id: str,
    secret_value: str,
    region: Optional[str] = None,  # For replication policy if needed, or if secrets are regional
    labels: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Ensures a secret exists in GCP Secret Manager and adds the given value as a new version.
    Returns the full resource name of the new secret version if successful, else None.
    e.g., projects/PROJECT_ID/secrets/SECRET_ID/versions/VERSION
    """
    if not project_id:
        logger.error("GCP Project ID is required to manage secrets.")
        return None
    if not secret_id:
        logger.error("Secret ID is required to manage secrets.")
        return None
    if secret_value is None:
        logger.error("Secret value is required to create or update a secret.")
        return None

    logger.info(f"Ensuring secret '{secret_id}' in project '{project_id}'.")
    describe_cmd = ["secrets", "describe", secret_id, "--project", project_id]
    secret_exists, _, describe_stderr = _run_gcloud_command(describe_cmd, dry_run=False)

    if not secret_exists:
        if "NOT_FOUND" in describe_stderr.upper() or "failed to find" in describe_stderr.lower():
            logger.info(f"Secret '{secret_id}' not found. Attempting to create it.")
            create_cmd_list = [
                "secrets",
                "create",
                secret_id,
                "--project",
                project_id,
            ]
            # Replication policy: automatic is default (global).
            # If a region is provided, could set --replication-policy user-managed --locations <region>
            # For simplicity, using automatic for now.
            # TODO: Consider if region-specific replication is needed.
            # if region:
            #     create_cmd_list.extend(["--replication-policy", "user-managed", "--locations", region])
            # else:
            create_cmd_list.extend(["--replication-policy", "automatic"])

            if labels:
                labels_str = ",".join([f"{k}={v}" for k, v in labels.items()])
                create_cmd_list.extend(["--labels", labels_str])

            create_success, _, create_stderr = _run_gcloud_command(create_cmd_list, dry_run=dry_run)
            if not create_success:
                logger.error(f"Failed to create secret '{secret_id}'. Stderr: {create_stderr}")
                return None
            logger.info(f"Successfully created secret '{secret_id}'.")
            secret_exists = True  # Now it exists
        else:
            # Describe failed for another reason
            logger.error(f"Error describing secret '{secret_id}'. Stderr: {describe_stderr}")
            return None

    # Add a new version to the secret
    # Create a temporary file for the secret value
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_secret_file:
            tmp_secret_file.write(secret_value)
            tmp_secret_file_path = tmp_secret_file.name

        add_version_cmd = [
            "secrets",
            "versions",
            "add",
            secret_id,
            "--project",
            project_id,
            "--data-file",
            tmp_secret_file_path,
        ]
        version_success, version_stdout, version_stderr = _run_gcloud_command(add_version_cmd, dry_run=dry_run)

        if tmp_secret_file_path and os.path.exists(tmp_secret_file_path):
            os.remove(tmp_secret_file_path)

        if not version_success:
            logger.error(f"Failed to add version to secret '{secret_id}'. Stderr: {version_stderr}")
            return None

        # The stdout of 'versions add' usually contains the version name, but it's safer to describe.
        # Let's parse the version from the output if available, or describe to get the latest.
        # For simplicity, if dry_run, we can't get a real version.
        if dry_run:
            logger.info(f"Successfully added version to secret '{secret_id}' (dry run).")
            return f"projects/{project_id}/secrets/{secret_id}/versions/latest-dry-run"

        # Get the full name of the newly added version
        # 'gcloud secrets versions describe latest --secret=SECRET_ID --format="value(name)"' gets the name
        describe_version_cmd = [
            "secrets",
            "versions",
            "describe",
            "latest",
            "--secret",
            secret_id,
            "--project",
            project_id,
            "--format",
            "value(name)",
        ]
        desc_ver_success, desc_ver_stdout, desc_ver_stderr = _run_gcloud_command(describe_version_cmd, dry_run=False)
        if desc_ver_success and desc_ver_stdout:
            secret_version_name = desc_ver_stdout.strip()
            logger.info(f"Successfully added version to secret '{secret_id}'. Version name: {secret_version_name}")
            return secret_version_name
        else:
            logger.error(
                f"Added version to secret '{secret_id}', but failed to retrieve new version name. Stderr: {desc_ver_stderr}"
            )
            return None

    except Exception as e:
        logger.error(f"An error occurred while adding secret version: {e}")
        if "tmp_secret_file_path" in locals() and os.path.exists(tmp_secret_file_path):
            os.remove(tmp_secret_file_path)
        return None


if __name__ == "__main__":
    # Basic setup for logger to see output when run directly
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger.info("--- GCP Tools Module (Placeholder Examples) ---")

    # Note: Dockerfile content would come from packaging.py
    dummy_dockerfile = 'FROM python:3.10-slim\nCMD ["echo", "hello"]'
    img_name = "gcr.io/my-test-project/my-test-reward-eval:latest"  # Old GCR name, update for AR
    # Example AR image name: us-central1-docker.pkg.dev/my-test-project/my-ar-repo/my-test-reward-eval:latest
    ar_img_name = "us-central1-docker.pkg.dev/my-test-project/eval-protocol-images/my-test-reward-eval:latest"

    print(f"\n1. Simulating build and push for {ar_img_name} (dry_run=True)")
    build_and_push_docker_image(
        image_name_tag=ar_img_name,
        dockerfile_content=dummy_dockerfile,
        build_context_dir=".",  # Assumes CWD is build context
        gcp_project_id="my-test-project",
        dry_run=True,
    )

    print("\n2. Simulating deploy to Cloud Run (dry_run=True)")
    deploy_to_cloud_run(
        service_name="my-reward-service",
        image_name_tag=ar_img_name,  # Use AR image name
        gcp_project_id="my-test-project",
        gcp_region="us-central1",
        allow_unauthenticated=True,
        env_vars={"MY_ENV_VAR": "my_value"},
        secrets_to_mount={"API_KEY_SECRET": "projects/my-test-project/secrets/my-api-key/versions/latest"},
        dry_run=True,
    )

    print("\n3. Simulating ensure_artifact_registry_repo_exists (dry_run=True)")
    ensure_artifact_registry_repo_exists(
        project_id="my-test-project",
        region="us-central1",
        repo_name="eval-protocol-evaluators",
        dry_run=True,
    )

    print("\n4. Simulating ensure_gcp_secret (dry_run=True)")
    ensure_gcp_secret(
        project_id="my-test-project",
        secret_id="my-test-api-key-secret",
        secret_value="supersecretvalue123",
        labels={"managed-by": "eval-protocol-test"},
        dry_run=True,
    )
    print("\nNote: These are placeholder executions. Real implementation requires gcloud CLI and Docker.")
