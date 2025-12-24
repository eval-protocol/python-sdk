import ast  # Added for AST parsing
import importlib.util  # Added for dynamic module loading
import json
import logging
import os
import sys  # Added for path manipulation
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

if TYPE_CHECKING:
    # For type checking only
    import datasets

from fireworks import Fireworks
import fireworks
import requests

from eval_protocol.auth import (
    get_fireworks_account_id,
    get_fireworks_api_key,
    verify_api_key_and_get_account_id,
)
from eval_protocol.typed_interface import EvaluationMode

from eval_protocol.get_pep440_version import get_pep440_version

logger = logging.getLogger(__name__)


def huggingface_dataset_to_jsonl(
    dataset_name: str,
    split: str = "train",
    output_file: Optional[str] = None,
    max_samples: int = 100,
    message_key_map: Optional[Dict[str, str]] = None,
    response_key: str = "response",
    prompt_key: str = "prompt",
) -> str:
    """
    Converts a HuggingFace dataset to JSONL format suitable for Eval Protocol evaluation.

    Args:
        dataset_name: The name of the HuggingFace dataset (e.g., "deepseek-ai/DeepSeek-ProverBench")
        split: The dataset split to use (default: "train")
        output_file: Optional file path to save the JSONL output (if None, generates a temp file)
        max_samples: Maximum number of samples to include
        message_key_map: Optional mapping of dataset keys to Eval Protocol message keys
        response_key: Key in the dataset containing the response text (default: "response")
        prompt_key: Key in the dataset containing the prompt text (default: "prompt")

    Returns:
        Path to the generated JSONL file
    """
    try:
        from datasets import load_dataset  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to use this function. "
            "Please install it with 'pip install \"eval-protocol[deepseek]\"'"
        )

    import tempfile

    logger.info(f"Loading dataset {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)

    if not output_file:
        temp_dir = tempfile.gettempdir()
        dataset_basename = dataset_name.split("/")[-1]
        output_file = os.path.join(temp_dir, f"{dataset_basename}_{split}_{int(time.time())}.jsonl")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    if message_key_map is None:
        message_key_map = {}

    processed_samples = 0
    # Initialize i to handle empty dataset case for logging
    i = -1
    with open(output_file, "w") as f:
        for i, item in enumerate(dataset):
            if processed_samples >= max_samples:
                break

            if prompt_key not in item and "statement" not in item:
                logger.debug(f"Skipping sample {i} due to missing prompt/statement key.")
                continue

            prompt_text = item.get(prompt_key, item.get("statement", ""))
            response_text = item.get(
                response_key,
                item.get("reference_solution", item.get("expected_proof", "")),
            )

            if not prompt_text or not response_text:
                logger.debug(f"Skipping sample {i} due to missing prompt or response text.")
                continue

            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text},
            ]
            entry = {"messages": messages}

            for ds_key, rk_key in message_key_map.items():
                if ds_key in item:
                    entry[rk_key] = item[ds_key]

            for key, value in item.items():
                if key not in [prompt_key, response_key] and key not in message_key_map:
                    entry[key] = value

            f.write(json.dumps(entry) + "\n")
            processed_samples += 1

        if processed_samples == 0 and i == -1:
            logger.info(f"No samples converted to JSONL format: {output_file}")
        else:
            logger.info(f"Converted {processed_samples} samples to JSONL format: {output_file}")
    return output_file


class Evaluator:
    def __init__(
        self,
        multi_metrics=False,  # Relates to output structure (dict of metrics vs single)
        remote_url: Optional[str] = None,
        ts_mode_config: Optional[Dict[str, Any]] = None,
        reward_function_mode: EvaluationMode = "pointwise",  # New parameter for input processing mode
        account_id: Optional[str] = None,
        api_key: Optional[str] = None,
        entry_point: Optional[str] = None,
    ):
        self.multi_metrics = multi_metrics
        self.remote_url = remote_url
        self.ts_mode_config = ts_mode_config
        self.reward_function_mode = reward_function_mode
        self.code_files = {}
        self.metric_folders: Dict[str, Dict[str, Any]] = {}  # Changed to store path and requirements
        self.account_id = account_id
        self.api_key = api_key
        self.description = ""
        self.display_name = ""
        self.api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
        # Optional requirements string for multi-metric mode (when loaded differently)
        self._loaded_multi_metric_requirements_str: Optional[str] = None
        # Optional entry point metadata (module::function or path::function)
        self.entry_point: Optional[str] = entry_point

        if self.ts_mode_config:
            python_code = self.ts_mode_config.get("python_code")
            file_name = self.ts_mode_config.get("file_name", "main.py")
            if not python_code:
                raise ValueError("python_code is required in ts_mode_config")
            self.code_files[file_name] = python_code

    def _should_include_file(self, filename: str) -> bool:
        """Check if a file should be included in the evaluator upload."""
        return (
            filename.endswith(".py")
            or filename.endswith(".txt")
            or filename.endswith(".toml")
            or os.path.basename(filename) == "Dockerfile"
        )

    def _load_python_files_from_folder(self, folder_path: str) -> Dict[str, str]:
        """
        Recursively loads Python, text, and TOML files from a given folder (excluding common ignored dirs).

        Args:
            folder_path: Absolute path to the folder.

        Returns:
            A dictionary mapping relative file paths (within folder) to their content.

        Raises:
            ValueError: If folder_path is invalid or not a directory.
        """
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        files: Dict[str, str] = {}
        ignored_dirs = {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", "vendor"}
        base_path = Path(folder_path)
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # prune ignored directories
            dirnames[:] = [d for d in dirnames if d not in ignored_dirs and not d.startswith(".")]
            for name in filenames:
                if not self._should_include_file(name):
                    continue
                abs_path = Path(dirpath) / name
                rel_path = str(abs_path.relative_to(base_path))
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                files[rel_path] = content
        if not files:
            raise ValueError(f"No Python, text, or TOML files found in {folder_path}")
        return files

    def load_metric_folder(self, metric_name, folder_path):
        """
        Load code files from a metric folder

        Args:
            metric_name: Name of the metric
            folder_path: Path to the folder containing code files

        Returns:
            Dict mapping filenames to their contents
        """
        folder_path = os.path.abspath(folder_path)
        files = self._load_python_files_from_folder(folder_path)  # Reads all .py files into a dict
        metric_requirements_list: Optional[List[str]] = None

        main_py_content = files.get("main.py")
        if main_py_content:
            try:
                tree = ast.parse(main_py_content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                        for decorator_node in node.decorator_list:
                            if (
                                isinstance(decorator_node, ast.Call)
                                and isinstance(decorator_node.func, ast.Name)
                                and decorator_node.func.id == "reward_function"
                            ):
                                for keyword in decorator_node.keywords:
                                    if keyword.arg == "requirements":
                                        if isinstance(keyword.value, ast.List):
                                            reqs: List[str] = []
                                            for elt in keyword.value.elts:
                                                if isinstance(elt, ast.Constant):  # Python 3.8+
                                                    if isinstance(elt.value, str):
                                                        reqs.append(cast(str, elt.value))
                                                elif isinstance(elt, ast.Str):  # Python < 3.8
                                                    reqs.append(cast(str, elt.s))
                                            if reqs:
                                                metric_requirements_list = cast(List[str], reqs)
                                        elif isinstance(keyword.value, ast.Constant) and isinstance(
                                            keyword.value.value, str
                                        ):  # Python 3.8+ (single req string)
                                            metric_requirements_list = [cast(str, keyword.value.value)]
                                        elif isinstance(keyword.value, ast.Str):  # Python < 3.8 (single req string)
                                            metric_requirements_list = [cast(str, keyword.value.s)]
                                        break
                                if metric_requirements_list:
                                    break
                        if metric_requirements_list:
                            logger.info(
                                f"Found requirements for metric '{metric_name}' via AST: {metric_requirements_list}"
                            )
                            break
            except SyntaxError as e:
                logger.error(f"Syntax error parsing main.py for metric '{metric_name}' to find requirements: {e}")
            except Exception as e:
                logger.error(f"Error parsing main.py AST for metric '{metric_name}': {e}")

        self.metric_folders[metric_name] = {
            "path": folder_path,
            "requirements": metric_requirements_list,  # This is now a list of strings or None
        }

        for filename, content in files.items():
            self.code_files[f"{metric_name}/{filename}"] = content

        logger.info(f"Loaded {len(files)} files for metric '{metric_name}' from {folder_path}")
        return files

    def load_multi_metrics_folder(self, folder_path):
        """
        Load code files from a folder with multiple metrics

        Args:
            folder_path: Path to the folder containing code files

        Returns:
            Dict mapping filenames to their contents
        """
        folder_path = os.path.abspath(folder_path)
        files = self._load_python_files_from_folder(folder_path)

        self.code_files = files
        logger.info(f"Loaded {len(files)} files from {folder_path} for multi-metrics evaluation")
        return files

    def load_samples_from_jsonl(self, sample_file, max_samples=5):
        if not os.path.exists(sample_file):
            raise ValueError(f"Sample file does not exist: {sample_file}")
        samples = []
        with open(sample_file, "r") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {i + 1}, skipping")
        logger.info(f"Loaded {len(samples)} samples from {sample_file}")
        return samples

    @staticmethod
    def _parse_ignore_file(ignore_path: str) -> List[str]:
        """Parse .gitignore or .dockerignore and return patterns."""
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
        dockerignore_patterns = Evaluator._parse_ignore_file(str(source_path / ".dockerignore"))
        all_ignore_patterns = gitignore_patterns + dockerignore_patterns

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
        if not self.remote_url and not self.ts_mode_config and not self.code_files:
            raise ValueError("No code files loaded. Load metric folder(s) or provide ts_mode_config/remote_url first.")

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
    metric_folders: Optional[List[str]] = None,
    multi_metrics: bool = False,  # Original folder-based multi_metrics flag
    folder: Optional[str] = None,
    python_code_to_evaluate: Optional[str] = None,
    python_file_name_for_code: str = "main.py",
    criterion_name_for_code: str = "default_code_criterion",
    criterion_description_for_code: str = "Python code execution",
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    force: bool = False,
    huggingface_dataset: Optional[str] = None,
    huggingface_split: str = "train",
    huggingface_message_key_map: Optional[Dict[str, str]] = None,
    huggingface_response_key: str = "response",
    huggingface_prompt_key: str = "prompt",
    remote_url: Optional[str] = None,
    reward_function_mode: EvaluationMode = "pointwise",  # Added
    account_id: Optional[str] = None,
    api_key: Optional[str] = None,
    entry_point: Optional[str] = None,
):
    ts_mode_config = None
    if python_code_to_evaluate:
        if metric_folders or folder:  # Removed multi_metrics from this check
            raise ValueError("Cannot use python_code_to_evaluate with folder-based parameters.")
        ts_mode_config = {
            "python_code": python_code_to_evaluate,
            "file_name": python_file_name_for_code,
            "criterion_name": criterion_name_for_code,
            "description": criterion_description_for_code,
        }

    evaluator = Evaluator(
        multi_metrics=multi_metrics,
        remote_url=remote_url,
        ts_mode_config=ts_mode_config,
        reward_function_mode=reward_function_mode,
        account_id=account_id,
        api_key=api_key,
        entry_point=entry_point,
    )

    if remote_url:
        logger.info(f"Configuring evaluator to use remote URL: {remote_url}")
        if (
            metric_folders or folder or python_code_to_evaluate
        ):  # If remote_url, other code sources are ignored for execution
            logger.warning(
                "When remote_url is provided, other code sources (folders, python_code_to_evaluate) are ignored for execution logic by the platform."
            )
    elif ts_mode_config:
        # ts_mode_config already handled in Evaluator.__init__ for self.code_files
        logger.info("Configuring evaluator with direct Python code snippet (ts_mode).")
    elif multi_metrics:  # Folder-based multi_metrics
        if not folder:
            raise ValueError("`folder` must be specified for folder-based multi_metrics mode.")
        evaluator.load_multi_metrics_folder(folder)
    else:  # Folder-based single/multiple metrics (non-multi_metrics structure)
        if not metric_folders:
            raise ValueError("At least one metric_folder must be specified.")
        for pair in metric_folders:
            if "=" not in pair:
                raise ValueError(f"Invalid metric-folder format: {pair}.")
            metric_name, folder_path = pair.split("=", 1)
            evaluator.load_metric_folder(metric_name, folder_path)

    if huggingface_dataset:
        logger.info(f"HuggingFace dataset specified: {huggingface_dataset}")

    return evaluator.create(evaluator_id, display_name, description, force)
