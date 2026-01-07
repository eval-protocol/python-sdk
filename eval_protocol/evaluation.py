import logging
import os
import time
from typing import List, Optional

import fireworks
import requests
from fireworks import Fireworks

from eval_protocol.auth import (
    get_fireworks_account_id,
    get_fireworks_api_key,
    verify_api_key_and_get_account_id,
)
from eval_protocol.get_pep440_version import get_pep440_version

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        account_id: Optional[str] = None,
        api_key: Optional[str] = None,
        entry_point: Optional[str] = None,
    ):
        self.account_id = account_id
        self.api_key = api_key
        self.description = ""
        self.display_name = ""
        self.api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
        self.entry_point: Optional[str] = entry_point

    @staticmethod
    def _parse_ignore_file(ignore_path: str) -> List[str]:
        """Parse .gitignore and return patterns."""
        patterns = []
        if not os.path.exists(ignore_path):
            return patterns

        try:
            with open(ignore_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except Exception:
            pass

        return patterns

    @staticmethod
    def _ensure_requirements_present(source_dir: str) -> None:
        req_path = os.path.join(source_dir, "requirements.txt")
        if not os.path.isfile(req_path):
            logger.error("Missing requirements.txt in upload directory: %s", source_dir)
            raise ValueError(
                "Upload requires requirements.txt in the project root. "
                "Create a requirements.txt (it can be empty) and rerun 'eval-protocol upload' "
                "or 'eval-protocol create rft'. If you're running in a notebook (e.g., Colab), "
                f"create the file in your working directory (e.g., {source_dir}/requirements.txt)."
            )

    @staticmethod
    def _should_ignore(path: str, ignore_patterns: List[str]) -> bool:
        """Check if path matches any ignore pattern."""
        from pathlib import Path
        import fnmatch

        default_ignores = [
            ".git",
            ".github",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".venv",
            "venv",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".ipynb_checkpoints",
            ".idea",
            ".vscode",
            ".cache",
            "node_modules",
            "vendor",
            "dist",
            "build",
            "*.egg-info",
            "*.egg",
            "*.whl",
            "*.tar.gz",
            "*.zip",
            "*.log",
            "*.tmp",
            "*.swp",
            ".DS_Store",
            "coverage",
            "htmlcov",
            ".coverage",
            "coverage.xml",
            ".env",
            ".env.*",
            "*.so",
            "*.dylib",
            ".pytest_cache/",
            "env/",
        ]
        all_patterns = default_ignores + ignore_patterns

        path_obj = Path(path)
        for pattern in all_patterns:
            if pattern.endswith("/"):
                if path_obj.is_dir() and fnmatch.fnmatch(path_obj.name, pattern.rstrip("/")):
                    return True
            elif fnmatch.fnmatch(path_obj.name, pattern) or fnmatch.fnmatch(str(path_obj), pattern):
                return True

        return False

    @staticmethod
    def _create_tar_gz_with_ignores(output_path: str, source_dir: str) -> int:
        """Create tar.gz of source_dir with parent directory included."""
        import tarfile
        from pathlib import Path

        source_path = Path(source_dir)
        gitignore_patterns = Evaluator._parse_ignore_file(str(source_path / ".gitignore"))
        all_ignore_patterns = gitignore_patterns

        logger.info(f"Creating tar.gz with {len(all_ignore_patterns)} ignore patterns")

        # Get directory name for the archive root
        dir_name = os.path.basename(source_dir)
        parent_dir = os.path.dirname(source_dir)

        with tarfile.open(output_path, "w:gz") as tar:
            for root, dirs, files in os.walk(source_dir):
                dirs[:] = [d for d in dirs if not Evaluator._should_ignore(os.path.join(root, d), all_ignore_patterns)]

                for file in files:
                    file_path = os.path.join(root, file)
                    if Evaluator._should_ignore(file_path, all_ignore_patterns):
                        continue

                    # Include parent directory in archive path
                    rel_path = os.path.relpath(file_path, parent_dir)  # Relative to parent
                    tar.add(file_path, arcname=rel_path)  # Keeps "python-sdk/..." structure

        size_bytes = os.path.getsize(output_path)
        logger.info(f"Created {output_path} ({size_bytes:,} bytes)")
        return size_bytes

    def create(self, evaluator_id, display_name=None, description=None, force=False):
        auth_token = self.api_key or get_fireworks_api_key()
        account_id = self.account_id or get_fireworks_account_id()
        if not account_id and auth_token:
            # Attempt to verify the API key and derive account id from server headers
            account_id = verify_api_key_and_get_account_id(api_key=auth_token, api_base=self.api_base)
        if not auth_token or not account_id:
            logger.error("Authentication error: API credentials appear to be invalid or incomplete.")
            raise ValueError("Invalid or missing API credentials.")

        client = Fireworks(api_key=auth_token, base_url=self.api_base, account_id=account_id)

        self.display_name = display_name or evaluator_id
        self.description = description or f"Evaluator created from {evaluator_id}"

        try:
            version_str = get_pep440_version()
        except Exception:
            version_str = None

        # Build evaluator params for SDK
        from fireworks.types import evaluator_create_params

        evaluator_params: evaluator_create_params.Evaluator = {
            "display_name": self.display_name,
            "description": self.description,
        }
        if version_str:
            evaluator_params["commit_hash"] = version_str
        if self.entry_point:
            evaluator_params["entry_point"] = self.entry_point
            logger.info(f"Including entryPoint in payload: {self.entry_point}")

        # Debug log the create payload structure
        try:
            logger.info(f"Create API Request: evaluator_id={evaluator_id}, evaluator={evaluator_params}")
        except Exception:
            pass

        self._ensure_requirements_present(os.getcwd())

        logger.info(f"Creating evaluator '{evaluator_id}' for account '{account_id}'...")

        try:
            if force:
                try:
                    logger.info("Checking if evaluator exists")
                    existing_evaluator = client.evaluators.get(evaluator_id=evaluator_id)
                    if existing_evaluator:
                        logger.info(f"Evaluator '{evaluator_id}' already exists, deleting and recreating...")
                        try:
                            client.evaluators.delete(evaluator_id=evaluator_id)
                            logger.info(f"Successfully deleted evaluator '{evaluator_id}'")
                        except fireworks.NotFoundError:
                            logger.info(f"Evaluator '{evaluator_id}' not found, creating...")
                        except fireworks.APIError as e:
                            logger.warning(f"Error deleting evaluator: {str(e)}")
                except fireworks.NotFoundError:
                    logger.info(f"Evaluator '{evaluator_id}' does not exist, creating...")

            # Create evaluator using SDK
            result = client.evaluators.create(
                evaluator_id=evaluator_id,
                evaluator=evaluator_params,
            )
            logger.info(f"Successfully created evaluator '{evaluator_id}'")

            # Upload code as tar.gz to GCS
            evaluator_name = result.name  # e.g., "accounts/pyroworks/evaluators/test-123"

            if not evaluator_name:
                raise ValueError(
                    "Create evaluator response missing 'name' field. "
                    f"Cannot proceed with code upload. Response: {result}"
                )

            try:
                # Create tar.gz of current directory
                cwd = os.getcwd()
                dir_name = os.path.basename(cwd)
                tar_filename = f"{dir_name}.tar.gz"
                tar_path = os.path.join(cwd, tar_filename)

                tar_size = self._create_tar_gz_with_ignores(tar_path, cwd)

                # Call GetEvaluatorUploadEndpoint using SDK
                logger.info(f"Requesting upload endpoint for {tar_filename}")
                upload_response = client.evaluators.get_upload_endpoint(
                    evaluator_id=evaluator_id,
                    filename_to_size={tar_filename: str(tar_size)},
                )

                # Check for signed URLs
                signed_urls = upload_response.filename_to_signed_urls or {}

                if not signed_urls:
                    raise ValueError(f"GetUploadEndpoint returned no signed URLs. Response: {upload_response}")

                signed_url = signed_urls.get(tar_filename)

                if not signed_url:
                    raise ValueError(
                        f"No signed URL received for {tar_filename}. Available files: {list(signed_urls.keys())}"
                    )

                # Upload to GCS
                logger.info(f"Uploading {tar_filename} to GCS...")

                file_size = os.path.getsize(tar_path)

                # Retry configuration
                max_retries = 3
                retry_delay = 2  # seconds

                for attempt in range(max_retries):
                    try:
                        with open(tar_path, "rb") as f:
                            # Create request exactly like Golang
                            req = requests.Request(
                                "PUT",
                                signed_url,
                                data=f,
                                headers={
                                    "Content-Type": "application/octet-stream",
                                    "X-Goog-Content-Length-Range": f"{file_size},{file_size}",
                                },
                            )
                            prepared = req.prepare()

                            # Don't let requests add extra headers
                            session = requests.Session()
                            gcs_response = session.send(prepared, timeout=600)
                            gcs_response.raise_for_status()

                        logger.info(f"Successfully uploaded {tar_filename}")
                        break  # Success, exit retry loop

                    except (requests.exceptions.RequestException, IOError) as e:
                        if attempt < max_retries - 1:
                            # Check if it's a retryable error
                            is_retryable = False
                            if isinstance(e, requests.exceptions.RequestException):
                                if hasattr(e, "response") and e.response is not None:
                                    # Retry on 5xx errors or 408 (timeout)
                                    is_retryable = e.response.status_code >= 500 or e.response.status_code == 408
                                else:
                                    # Network errors (no response) are retryable
                                    is_retryable = True
                            else:
                                # IOError is retryable
                                is_retryable = True

                            if is_retryable:
                                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                                logger.warning(
                                    f"Upload attempt {attempt + 1}/{max_retries} failed: {e}. "
                                    f"Retrying in {wait_time}s..."
                                )
                                time.sleep(wait_time)
                            else:
                                # Non-retryable error, raise immediately
                                raise
                        else:
                            # Last attempt failed
                            logger.error(f"Upload failed after {max_retries} attempts")
                            raise

                # Step 3: Validate upload using SDK
                client.evaluators.validate_upload(
                    evaluator_id=evaluator_id,
                    body={},
                )
                logger.info("Upload validated successfully")

                # Clean up tar file
                if os.path.exists(tar_path):
                    os.remove(tar_path)

            except Exception as upload_error:
                logger.warning(f"Code upload failed (evaluator created but code not uploaded): {upload_error}")
                # Don't fail - evaluator is created, just code upload failed

            return result  # Return after attempting upload
        except fireworks.APIStatusError as e:
            logger.error(f"Error creating evaluator: {str(e)}")
            logger.error(f"Status code: {e.status_code}, Response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error creating evaluator: {str(e)}")
            raise

    def _get_authentication(self):
        account_id = get_fireworks_account_id()
        auth_token = get_fireworks_api_key()
        if not account_id:
            logger.error("Authentication error: Fireworks Account ID not found.")
            raise ValueError("Fireworks Account ID not found.")
        if not auth_token:
            logger.error("Authentication error: Fireworks API Key not found.")
            raise ValueError("Fireworks API Key not found.")
        return account_id, auth_token


# Helper functions for CLI commands
def create_evaluation(
    evaluator_id: str,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    force: bool = False,
    account_id: Optional[str] = None,
    api_key: Optional[str] = None,
    entry_point: Optional[str] = None,
):
    """
    Create an evaluator on the Fireworks platform.

    Args:
        evaluator_id: Unique identifier for the evaluator
        display_name: Display name for the evaluator
        description: Description for the evaluator
        force: If True, delete and recreate if evaluator exists
        account_id: Optional Fireworks account ID
        api_key: Optional Fireworks API key
        entry_point: Optional entry point (module::function or path::function)
    """
    evaluator = Evaluator(
        account_id=account_id,
        api_key=api_key,
        entry_point=entry_point,
    )

    return evaluator.create(evaluator_id, display_name, description, force)
