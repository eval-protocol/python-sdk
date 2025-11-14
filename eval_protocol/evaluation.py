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

import requests

from eval_protocol.auth import (
    get_fireworks_account_id,
    get_fireworks_api_key,
    verify_api_key_and_get_account_id,
)
from eval_protocol.common_utils import get_user_agent
from eval_protocol.typed_interface import EvaluationMode

from eval_protocol.get_pep440_version import get_pep440_version

logger = logging.getLogger(__name__)

# Flag to track if the preview API was successfully used
used_preview_api = False


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


class EvaluatorPreviewResult:
    def __init__(self):
        self.results = []
        self.total_samples = 0
        self.total_runtime_ms = 0

    def add_result(self, sample_index, success, score, per_metric_evals):
        result_obj = types.SimpleNamespace(
            index=sample_index,
            success=success,
            score=score,
            per_metric_evals=per_metric_evals,
        )
        self.results.append(result_obj)

    def display(self):
        print("Evaluation Preview Results")
        print("------------------------")
        print(f"Total Samples: {self.total_samples}")
        print(f"Total Runtime: {self.total_runtime_ms} ms\n")
        print("Individual Results:")
        print("------------------")
        for i, result_obj in enumerate(self.results):
            print(f"Sample {result_obj.index + 1}:")
            print(f"  Success: {result_obj.success}")
            print(f"  Score: {result_obj.score}")
            if hasattr(result_obj, "per_metric_evals") and isinstance(result_obj.per_metric_evals, dict):
                for metric, value in result_obj.per_metric_evals.items():
                    print(f"  {metric}: {value}")
            elif hasattr(result_obj, "per_metric_evals"):
                print(f"  Per-Metric Evals: {result_obj.per_metric_evals}")
            if i < len(self.results) - 1:
                print()


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
            # ts_mode implies multiMetrics: true for the payload structure
            # but it's distinct from folder-based multi_metrics for loading.
            # The original self.multi_metrics flag is for folder loading.
            # The payload's multiMetrics field will be set to True if ts_mode_config is active.
            # The check for (metric_folders or folder) is not applicable in __init__ and was causing an error.
            # If ts_mode_config is active, it takes precedence for code definition.
            # The multi_metrics flag passed to __init__ is for folder-based loading if ts_mode_config is not used.

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

    def preview(self, sample_file, max_samples=5):
        if not self.remote_url and not self.ts_mode_config and not self.code_files:
            raise ValueError("No code files loaded. Load metric folder(s) or provide ts_mode_config/remote_url first.")

        # If not remote and not ts_mode, then main.py check applies to loaded code_files
        if not self.remote_url and not self.ts_mode_config:
            if "main.py" not in self.code_files and not any(k.endswith("/main.py") for k in self.code_files):
                raise ValueError("No main.py found in loaded code files for folder-based evaluation.")

        samples = self.load_samples_from_jsonl(sample_file, max_samples)
        if not samples:
            raise ValueError(f"No valid samples found in {sample_file}")

        auth_token = self.api_key or get_fireworks_api_key()
        account_id = self.account_id or get_fireworks_account_id()
        if not account_id and auth_token:
            account_id = verify_api_key_and_get_account_id(api_key=auth_token, api_base=self.api_base)
        logger.debug(f"Preview using account_id: {account_id}")

        if not account_id or not auth_token:
            logger.error("Authentication error: Missing Fireworks Account ID or API Key.")
            raise ValueError("Missing Fireworks Account ID or API Key.")

        # Keep multiMetrics/rollupSettings for backward compatibility with tests
        payload_multi_metrics = True
        payload_rollup_settings = {"skipRollup": True}

        # For preview, evaluator_id might not be as critical for shim's env var name,
        # but pass it for consistency. Use display_name as a proxy if no specific ID.
        preview_evaluator_id_for_shim = self.display_name or "preview_evaluator"
        evaluator_payload_data = {
            "displayName": self.display_name or "Preview Evaluator",
            "description": self.description or "Preview Evaluator",
            "multiMetrics": payload_multi_metrics,
            "criteria": self._construct_criteria(criteria_data={}),
            "requirements": self._get_combined_requirements(),
            "rollupSettings": payload_rollup_settings,
        }

        sample_strings = [json.dumps(sample) for sample in samples]
        payload = {
            "evaluator": evaluator_payload_data,
            "sampleData": sample_strings,
            "maxSamples": max_samples,
        }

        api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")

        if "dev.api.fireworks.ai" in api_base and account_id == "fireworks":
            account_id = "pyroworks-dev"

        url = f"{api_base}/v1/accounts/{account_id}/evaluators:previewEvaluator"
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
        }
        logger.info(f"Previewing evaluator using API endpoint: {url} with account: {account_id}")
        logger.debug(f"Preview API Request URL: {url}")
        logger.debug(f"Preview API Request Headers: {json.dumps(headers, indent=2)}")
        logger.debug(f"Preview API Request Payload: {json.dumps(payload, indent=2)}")

        global used_preview_api
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            used_preview_api = True
            preview_result_obj = EvaluatorPreviewResult()
            preview_result_obj.total_samples = result.get("totalSamples", len(samples))
            preview_result_obj.total_runtime_ms = int(result.get("totalRuntimeMs", 0))
            sample_results = result.get("results", [])
            for i, sample_result_item in enumerate(sample_results):
                preview_result_obj.add_result(
                    sample_index=i,
                    success=sample_result_item.get("success", False),
                    score=sample_result_item.get("score", 0.0),
                    per_metric_evals=sample_result_item.get("perMetricEvals", {}),
                )
            return preview_result_obj
        except Exception as e:
            logger.error(f"Error previewing evaluator: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, "response"):
                logger.error(f"Response: {e.response.text}")
            used_preview_api = False
            logger.warning("Falling back to simulated preview mode")
            return self._simulated_preview(samples)

    def _get_combined_requirements(self) -> str:
        """Combines requirements from all loaded metrics."""
        all_requirements_set = set()
        for metric_data in self.metric_folders.values():
            req_list_or_str = metric_data.get("requirements")
            if req_list_or_str:
                if isinstance(req_list_or_str, list):
                    for req_item in req_list_or_str:
                        if isinstance(req_item, str):
                            all_requirements_set.add(req_item.strip())
                elif isinstance(req_list_or_str, str):  # Fallback if somehow a string is still passed
                    items = [r.strip() for r in req_list_or_str.splitlines() if r.strip()]
                    for item in items:
                        all_requirements_set.add(item)

        # For multi_metrics loaded directly into self.code_files (not via metric_folders)
        # This part is more complex as it requires loading the 'main.py' from self.code_files
        # if self.multi_metrics and not self.metric_folders and "main.py" in self.code_files:
        # We would need a temporary way to load this main.py to get its requirements.
        # For now, focusing on metric_folders which is the primary path for --metrics-folders.
        # If a multi_metrics folder is loaded via load_multi_metrics_folder, it also needs a similar
        # dynamic import logic to fetch requirements from its main 'evaluate' function.
        # This part is NOT YET IMPLEMENTED for multi_metrics folders.

        if not all_requirements_set and hasattr(self, "_loaded_multi_metric_requirements_str"):
            # Fallback for multi_metrics if requirements were loaded differently (hypothetical)
            # This attribute doesn't exist yet, placeholder for future enhancement if needed.
            if self._loaded_multi_metric_requirements_str:  # type: ignore
                requirements_list = [
                    r.strip() for r in self._loaded_multi_metric_requirements_str.splitlines() if r.strip()
                ]  # type: ignore
                for req_item in requirements_list:
                    all_requirements_set.add(req_item)

        logger.info(f"Combined unique requirements: {all_requirements_set}")
        return "\n".join(sorted(list(all_requirements_set)))

    def _simulated_preview(self, samples):
        preview_result = EvaluatorPreviewResult()
        preview_result.total_samples = len(samples)
        start_time = time.time()
        for i, sample in enumerate(samples):
            try:
                if "messages" not in sample:
                    raise ValueError(f"Sample {i + 1} is missing 'messages' field")
                _ = sample.get("messages", [])
                _ = sample.get("ground_truth", [])
                _ = sample.get("tools", [])
                _ = {
                    k: v
                    for k, v in sample.items()
                    if k
                    not in [
                        "messages",
                        "ground_truth",
                        "tools",
                    ]
                }

                if self.multi_metrics or self.ts_mode_config:  # ts_mode also implies a single set of results
                    per_metric_evals = {"quality": 0.8, "relevance": 0.7, "safety": 0.9}
                else:
                    per_metric_evals = {metric_name: 0.75 for metric_name in self.metric_folders}

                score = sum(per_metric_evals.values()) / len(per_metric_evals) if per_metric_evals else 0.0
                preview_result.add_result(
                    sample_index=i,
                    success=True,
                    score=score,
                    per_metric_evals=per_metric_evals,
                )
            except Exception as e:
                logger.error(f"Error processing sample {i + 1}: {str(e)}")
                preview_result.add_result(
                    sample_index=i,
                    success=False,
                    score=0.0,
                    per_metric_evals={"error": str(e)},
                )
        end_time = time.time()
        preview_result.total_runtime_ms = max(1, int((end_time - start_time) * 1000))
        return preview_result

    def _build_minimal_criteria(self) -> List[Dict[str, str]]:
        """Build minimal criteria (name, type, description) without code snippets."""

        # Remote URL mode
        if self.remote_url:
            return [
                {
                    "name": "remote_eval_proxy",
                    "type": "CODE_SNIPPETS",
                    "description": f"Proxies evaluation to remote URL: {self.remote_url}",
                }
            ]

        # TS mode (direct code snippet)
        elif self.ts_mode_config:
            criterion_name = self.ts_mode_config.get("criterion_name", "default_code_criterion")
            description = self.ts_mode_config.get("description", "Python code execution")
            return [
                {
                    "name": criterion_name,
                    "type": "CODE_SNIPPETS",
                    "description": description,
                }
            ]

        # Multi-metrics mode
        elif self.multi_metrics:
            return [
                {
                    "name": "eval",
                    "type": "CODE_SNIPPETS",
                    "description": self.description or "Multi-metric evaluation",
                }
            ]

        # Single metric folders
        else:
            criteria = []
            for metric_name in self.metric_folders:
                criteria.append(
                    {
                        "name": metric_name,
                        "type": "CODE_SNIPPETS",
                        "description": self.description or f"Evaluation metric: {metric_name}",
                    }
                )
            return criteria

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

        self.display_name = display_name or evaluator_id
        self.description = description or f"Evaluator created from {evaluator_id}"

        # Keep multiMetrics/rollupSettings for backward compatibility with tests
        payload_multi_metrics = True
        payload_rollup_settings = {"skipRollup": True}
        parent = f"accounts/{account_id}"

        try:
            version_str = get_pep440_version()
        except Exception:
            version_str = None

        payload_data = {
            "parent": parent,
            "evaluator": {
                "displayName": self.display_name,
                "description": self.description,
                "multiMetrics": payload_multi_metrics,
                "commitHash": version_str,
                "criteria": self._build_minimal_criteria(),
                "requirements": "",
                "rollupSettings": payload_rollup_settings,
            },
            "evaluatorId": evaluator_id,
        }

        # Include optional entry point when provided
        if self.entry_point:
            payload_data["evaluator"]["entryPoint"] = self.entry_point
            logger.info(f"Including entryPoint in payload: {self.entry_point}")

        # Debug log the create payload structure (without sample data)
        try:
            logger.info(f"Create API Request Payload: {json.dumps(payload_data, indent=2)}")
        except Exception:
            # If serialization fails for any reason, skip debug dump
            pass

        if "dev.api.fireworks.ai" in self.api_base and account_id == "fireworks":
            account_id = "pyroworks-dev"

        base_url = f"{self.api_base}/v1/{parent}/evaluatorsV2"
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
        }

        self._ensure_requirements_present(os.getcwd())

        logger.info(f"Creating evaluator '{evaluator_id}' for account '{account_id}'...")

        try:
            if force:
                check_url = f"{self.api_base}/v1/{parent}/evaluators/{evaluator_id}"
                try:
                    logger.info(f"Checking if evaluator exists: {check_url}")
                    check_response = requests.get(check_url, headers=headers)

                    if check_response.status_code == 200:
                        logger.info(f"Evaluator '{evaluator_id}' already exists, deleting and recreating...")
                        delete_url = f"{self.api_base}/v1/{parent}/evaluators/{evaluator_id}"
                        try:
                            delete_response = requests.delete(delete_url, headers=headers)
                            if delete_response.status_code < 400:
                                logger.info(f"Successfully deleted evaluator '{evaluator_id}'")
                            else:
                                logger.warning(
                                    f"Unable to delete evaluator '{evaluator_id}', status: {delete_response.status_code}"
                                )
                        except Exception as e_del:
                            logger.warning(f"Error deleting evaluator: {str(e_del)}")
                        response = requests.post(base_url, json=payload_data, headers=headers)
                    else:
                        response = requests.post(base_url, json=payload_data, headers=headers)
                except requests.exceptions.RequestException:
                    response = requests.post(base_url, json=payload_data, headers=headers)
            else:
                logger.info(f"Creating evaluator at: {base_url}")
                response = requests.post(base_url, json=payload_data, headers=headers)

            response.raise_for_status()
            result = response.json()
            logger.info(f"Successfully created evaluator '{evaluator_id}'")

            # Upload code as tar.gz to GCS
            evaluator_name = result.get("name")  # e.g., "accounts/pyroworks/evaluators/test-123"

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

                # Call GetEvaluatorUploadEndpoint
                upload_endpoint_url = f"{self.api_base}/v1/{evaluator_name}:getUploadEndpoint"
                upload_payload = {"name": evaluator_name, "filename_to_size": {tar_filename: tar_size}}

                logger.info(f"Requesting upload endpoint for {tar_filename}")
                upload_response = requests.post(upload_endpoint_url, json=upload_payload, headers=headers)
                upload_response.raise_for_status()

                # Check for signed URLs
                upload_response_data = upload_response.json()
                signed_urls = upload_response_data.get("filenameToSignedUrls", {})

                if not signed_urls:
                    raise ValueError(f"GetUploadEndpoint returned no signed URLs. Response: {upload_response_data}")

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

                # Step 3: Validate upload
                validate_url = f"{self.api_base}/v1/{evaluator_name}:validateUpload"
                validate_payload = {"name": evaluator_name}
                validate_response = requests.post(validate_url, json=validate_payload, headers=headers)
                validate_response.raise_for_status()

                validate_data = validate_response.json()

                logger.info("Upload validated successfully")

                # Clean up tar file
                if os.path.exists(tar_path):
                    os.remove(tar_path)

            except Exception as upload_error:
                logger.warning(f"Code upload failed (evaluator created but code not uploaded): {upload_error}")
                # Don't fail - evaluator is created, just code upload failed

            return result  # Return after attempting upload
        except Exception as e:
            logger.error(f"Error creating evaluator: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, "response"):
                logger.error(f"Response: {e.response.text}")
            raise

    def _construct_criteria(self, criteria_data: Any) -> Any:
        assertions = []
        if self.remote_url:
            shim_main_py_content = f"""
import json
import os
import requests

REMOTE_EVALUATOR_URL = "{self.remote_url}"

def evaluate(messages, ground_truth: Optional[Union[str, List[Dict[str, Any]]]] = None, tools=None, **kwargs):
    payload = {{
        "messages": messages,
        "ground_truth": ground_truth,
        "tools": tools,
        "kwargs": kwargs
    }}
    headers = {{"Content-Type": "application/json"}}
    try:
        response = requests.post(REMOTE_EVALUATOR_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_info = {{
            "error": f"Failed to call remote evaluator at {{REMOTE_EVALUATOR_URL}}: {{str(e)}}",
            "status_code": getattr(e.response, 'status_code', None),
            "response_text": getattr(e.response, 'text', None)
        }}
        return {{
            "score": 0.0, "reason": f"Error calling remote evaluator: {{str(e)}}",
            "is_score_valid": False, "metrics": {{"remote_call_error": {{"score": 0.0, "is_score_valid": False, "reason": json.dumps(error_info)}}}}
        }}
    except Exception as e:
        return {{
            "score": 0.0, "reason": f"Unexpected error in remote evaluator shim: {{str(e)}}",
            "is_score_valid": False, "metrics": {{"shim_error": {{"score": 0.0, "is_score_valid": False, "reason": str(e)}}}}
        }}
"""
            file_contents = {"main.py": shim_main_py_content}
            assertions.append(
                {
                    "codeSnippets": {
                        "language": "python",
                        "fileContents": file_contents,
                    },
                    "name": "remote_eval_proxy",
                    "type": "CODE_SNIPPETS",
                    "description": f"Proxies evaluation to remote URL: {self.remote_url}",
                }
            )
        elif self.ts_mode_config:
            python_code = self.ts_mode_config.get("python_code")
            file_name = self.ts_mode_config.get("file_name", "main.py")
            criterion_name = self.ts_mode_config.get("criterion_name", "default_code_criterion")
            description = self.ts_mode_config.get("description", "Python code execution")
            if not python_code:
                raise ValueError("python_code is required in ts_mode_config")
            entry_func = "evaluate"
            try:
                if self.entry_point and "::" in self.entry_point:
                    entry_func = self.entry_point.split("::", 1)[1]
            except Exception:
                entry_func = "evaluate"
            assertions.append(
                {
                    "type": "CODE_SNIPPETS",
                    "name": criterion_name,
                    "description": description,
                    "codeSnippets": {
                        "language": "python",
                        "fileContents": {file_name: python_code},
                        "entryFile": file_name,
                        "entryFunc": entry_func,
                    },
                }
            )
        elif self.multi_metrics:
            file_contents = {}
            for filename, content in self.code_files.items():
                if filename.endswith(".py"):
                    file_contents[filename] = self._update_evaluate_signature(content)
                elif self._should_include_file(filename) and not filename.endswith(".py"):
                    file_contents[filename] = content
            if not file_contents:
                raise ValueError("No files found for multi-metrics mode.")
            # Determine entry file from entry_point if provided; otherwise detect
            entry_file = None
            if self.entry_point and "::" in self.entry_point:
                try:
                    ep_file = self.entry_point.split("::", 1)[0]
                    if ep_file in file_contents:
                        entry_file = ep_file
                    else:
                        ep_base = os.path.basename(ep_file)
                        for fname in file_contents.keys():
                            if os.path.basename(fname) == ep_base:
                                entry_file = fname
                                break
                except Exception:
                    entry_file = None
            if not entry_file:
                try:
                    for fname, content in file_contents.items():
                        for line in content.split("\n"):
                            s = line.lstrip()
                            if s.startswith("def evaluate(") or s.startswith("async def evaluate("):
                                entry_file = fname
                                break
                        if entry_file:
                            break
                except Exception:
                    entry_file = None
                if not entry_file:
                    entry_file = "main.py" if "main.py" in file_contents else list(file_contents.keys())[0]
            entry_func = "evaluate"
            try:
                if self.entry_point and "::" in self.entry_point:
                    entry_func = self.entry_point.split("::", 1)[1]
            except Exception:
                entry_func = "evaluate"
            assertions.append(
                {
                    "codeSnippets": {
                        "language": "python",
                        "fileContents": file_contents,
                        "entryFile": entry_file,
                        "entryFunc": entry_func,
                    },
                    "name": "eval",
                    "type": "CODE_SNIPPETS",
                    "description": self.description or "Multi-metric evaluation",
                }
            )
        else:  # Folder-based, non-multi_metrics
            for metric_name in self.metric_folders:
                file_contents = {}
                # Include all discovered files for this metric folder, preserving filenames
                for filename, content in self.code_files.items():
                    if filename.startswith(f"{metric_name}/"):
                        # Use the file name within the metric folder for clarity
                        short_name = filename.split(f"{metric_name}/", 1)[1]
                        if filename.endswith(".py"):
                            file_contents[short_name] = self._update_evaluate_signature(content)
                        elif self._should_include_file(filename) and not filename.endswith(".py"):
                            file_contents[short_name] = content
                if not file_contents:
                    logger.warning(f"No files prepared for metric '{metric_name}', skipping this metric for criteria.")
                    continue
                # Determine entry file within this metric's files using entry_point if present
                entry_file = None
                if self.entry_point and "::" in self.entry_point:
                    try:
                        ep_file = self.entry_point.split("::", 1)[0]
                        if ep_file in file_contents:
                            entry_file = ep_file
                        else:
                            ep_base = os.path.basename(ep_file)
                            for fname in file_contents.keys():
                                if os.path.basename(fname) == ep_base:
                                    entry_file = fname
                                    break
                    except Exception:
                        entry_file = None
                if not entry_file:
                    try:
                        for fname, content in file_contents.items():
                            for line in content.split("\n"):
                                s = line.lstrip()
                                if s.startswith("def evaluate(") or s.startswith("async def evaluate("):
                                    entry_file = fname
                                    break
                            if entry_file:
                                break
                    except Exception:
                        entry_file = None
                    if not entry_file:
                        entry_file = "main.py" if "main.py" in file_contents else list(file_contents.keys())[0]

                entry_func = "evaluate"
                try:
                    if self.entry_point and "::" in self.entry_point:
                        entry_func = self.entry_point.split("::", 1)[1]
                except Exception:
                    entry_func = "evaluate"
                assertions.append(
                    {
                        "codeSnippets": {
                            "language": "python",
                            "fileContents": file_contents,
                            "entryFile": entry_file,
                            "entryFunc": entry_func,
                        },
                        "name": metric_name,
                        "type": "CODE_SNIPPETS",
                        "description": f"Metric: {metric_name}",
                    }
                )

        if not assertions:
            raise ValueError("No valid criteria could be constructed.")
        return assertions

    def _update_evaluate_signature(self, content):
        import re

        # Simple regex to match the old evaluate function signature
        old_pattern = r"def\s+evaluate\s*\(\s*entry\s*(?::\s*dict)?\s*\)"
        # Regex to match the signature we are changing from (original_messages)
        current_signature_pattern = (
            r"def\s+evaluate\s*\(\s*messages,\s*original_messages\s*=\s*None,\s*tools\s*=\s*None,\s*\*\*kwargs\s*\)"
        )
        new_signature = "def evaluate(messages, ground_truth: Optional[Union[str, List[Dict[str, Any]]]] = None, tools=None, **kwargs)"

        # Check if the old pattern (entry-based) exists
        if re.search(old_pattern, content):
            updated_content = re.sub(old_pattern, new_signature, content, count=1)

            # Add a compatibility layer for the 'entry' style
            compat_layer = """
    # Compatibility layer for old 'entry' format
    if ground_truth is None: # Default ground_truth from messages if not provided
        ground_truth = messages
    # Assuming 'entry' dict was constructed from messages, original_messages (now ground_truth), tools, kwargs
    # This part might need more context on how 'entry' was used.
    # For now, we'll assume ground_truth takes precedence or is derived.
"""
        # Check if the current signature (with original_messages) exists
        elif re.search(current_signature_pattern, content):
            updated_content = re.sub(current_signature_pattern, new_signature, content, count=1)
            # No specific compatibility layer needed here as it's a direct parameter rename
            compat_layer = ""  # No additional layer for this direct change
        else:
            # If neither known signature is found, return content as is
            return content

        # Find the function body indent level if a change was made
        if "updated_content" in locals() and compat_layer:  # Only add layer if it's defined
            func_match = re.search(r"def\s+evaluate.*?:\s*\n(\s+)", updated_content, re.DOTALL)
            if func_match:
                indent = func_match.group(1)
                # Adjust indentation of compatibility layer
                indented_compat_layer = "\n".join(indent + line for line in compat_layer.strip().split("\n"))

                # Insert compatibility layer after function definition
                updated_content = re.sub(
                    re.escape(new_signature) + r"\s*:",
                    new_signature + ":" + indented_compat_layer,
                    updated_content,
                    count=1,
                )
            return updated_content
        elif "updated_content" in locals():
            return updated_content
        return content

    def _get_combined_code(self):  # This method seems unused now, consider removal
        # ... (implementation unchanged, but likely dead code)
        pass

    def _get_code_from_files(self, files):  # This method seems unused now, consider removal
        # ... (implementation unchanged, but likely dead code)
        pass

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
def preview_evaluation(
    metric_folders: Optional[List[str]] = None,
    multi_metrics: bool = False,
    folder: Optional[str] = None,
    python_code_to_evaluate: Optional[str] = None,
    python_file_name_for_code: str = "main.py",
    criterion_name_for_code: str = "default_code_criterion",
    criterion_description_for_code: str = "Python code execution",
    sample_file: Optional[str] = None,
    max_samples: int = 5,
    huggingface_dataset: Optional[str] = None,
    huggingface_split: str = "train",
    huggingface_message_key_map: Optional[Dict[str, str]] = None,
    huggingface_response_key: str = "response",
    huggingface_prompt_key: str = "prompt",
    reward_function_mode: EvaluationMode = "pointwise",  # Added for consistency
    account_id: Optional[str] = None,
    api_key: Optional[str] = None,
):
    ts_mode_config = None
    if python_code_to_evaluate:
        if metric_folders or folder:  # Removed multi_metrics from this check as it's handled by Evaluator init
            raise ValueError(
                "Cannot use python_code_to_evaluate with folder-based parameters (metric_folders, folder)."
            )
        ts_mode_config = {
            "python_code": python_code_to_evaluate,
            "file_name": python_file_name_for_code,
            "criterion_name": criterion_name_for_code,
            "description": criterion_description_for_code,
        }
        # When python_code_to_evaluate is used, multi_metrics in Evaluator constructor is effectively True
        # due to how ts_mode_config is handled (sets self.multi_metrics = True for payload).
        # The multi_metrics flag passed to Evaluator here should be the original one for folder logic.
        evaluator = Evaluator(
            multi_metrics=multi_metrics,
            ts_mode_config=ts_mode_config,
            reward_function_mode=reward_function_mode,
            account_id=account_id,
            api_key=api_key,
        )
    else:
        evaluator = Evaluator(
            multi_metrics=multi_metrics,
            reward_function_mode=reward_function_mode,
            account_id=account_id,
            api_key=api_key,
        )  # Pass mode to Evaluator
        if multi_metrics:
            if not folder:
                raise ValueError("`folder` must be specified for multi_metrics mode.")
            evaluator.load_multi_metrics_folder(folder)
        else:
            if not metric_folders:
                raise ValueError("At least one metric_folder must be specified.")
            for pair in metric_folders:
                if "=" not in pair:
                    raise ValueError(f"Invalid metric-folder format: {pair}.")
                metric_name, folder_path = pair.split("=", 1)
                evaluator.load_metric_folder(metric_name, folder_path)

    if huggingface_dataset:
        if sample_file:
            logger.warning("Both sample_file and huggingface_dataset specified. Using HuggingFace dataset.")
        sample_file = huggingface_dataset_to_jsonl(
            dataset_name=huggingface_dataset,
            split=huggingface_split,
            max_samples=max_samples,
            message_key_map=huggingface_message_key_map,
            response_key=huggingface_response_key,
            prompt_key=huggingface_prompt_key,
        )
        logger.info(f"Converted dataset saved to: {sample_file}")

    if not sample_file:
        raise ValueError("Either sample_file or huggingface_dataset must be specified.")
    return evaluator.preview(sample_file, max_samples)


def preview_folder_evaluation(  # This function might become redundant or need to align with the new preview_evaluation
    evaluator_folder,
    sample_file=None,
    max_samples=5,
    multi_metrics=False,  # original multi_metrics
    huggingface_dataset=None,
    huggingface_split="train",
    huggingface_message_key_map=None,
    huggingface_response_key="response",
    huggingface_prompt_key="prompt",
):
    evaluator_folder = os.path.abspath(evaluator_folder)
    if not os.path.exists(evaluator_folder):
        raise ValueError(f"Evaluator folder does not exist: {evaluator_folder}")
    if not os.path.isdir(evaluator_folder):
        raise ValueError(f"Not a directory: {evaluator_folder}")

    has_main_py = os.path.exists(os.path.join(evaluator_folder, "main.py"))
    # Auto-detect multi_metrics if not specified by caller
    detected_multi_metrics = multi_metrics
    if has_main_py and not multi_metrics:
        py_files = list(Path(evaluator_folder).glob("*.py"))
        if len(py_files) > 1:
            logger.info("Auto-detecting multi-metrics mode based on folder structure for preview_folder_evaluation")
            detected_multi_metrics = True

    # Call the unified preview_evaluation
    # This function doesn't directly support ts_mode_config, so python_code_to_evaluate is None
    return preview_evaluation(
        metric_folders=(
            None if detected_multi_metrics else [f"{os.path.basename(evaluator_folder)}={evaluator_folder}"]
        ),  # Simplified for now
        multi_metrics=detected_multi_metrics,
        folder=evaluator_folder if detected_multi_metrics else None,
        python_code_to_evaluate=None,  # Not applicable for this helper
        sample_file=sample_file,
        max_samples=max_samples,
        huggingface_dataset=huggingface_dataset,
        huggingface_split=huggingface_split,
        huggingface_message_key_map=huggingface_message_key_map,
        huggingface_response_key=huggingface_response_key,
        huggingface_prompt_key=huggingface_prompt_key,
    )


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
        logger.info(f"HuggingFace dataset specified: {huggingface_dataset} (currently for preview only).")

    return evaluator.create(evaluator_id, display_name, description, force)


def deploy_folder_evaluation(  # This function might become redundant or need to align with the new create_evaluation
    evaluator_id,
    evaluator_folder,
    display_name=None,
    description=None,
    force=False,
    multi_metrics=False,  # original multi_metrics
    huggingface_dataset=None,
    huggingface_split="train",
    huggingface_message_key_map=None,
    huggingface_response_key="response",
    huggingface_prompt_key="prompt",
    remote_url: Optional[str] = None,
):
    evaluator_folder_abs = os.path.abspath(evaluator_folder) if evaluator_folder else None

    # If remote_url is provided, evaluator_folder is less relevant for code loading
    # but might still be used for context/metadata if the function design implies it.
    # For now, if remote_url, we don't load from folder.

    python_code_to_evaluate = None  # This helper doesn't take direct code string

    if not remote_url and not evaluator_folder_abs:
        raise ValueError("evaluator_folder must be specified if not using remote_url.")

    if evaluator_folder_abs:
        if not os.path.exists(evaluator_folder_abs):
            raise ValueError(f"Evaluator folder does not exist: {evaluator_folder_abs}")
        if not os.path.isdir(evaluator_folder_abs):
            raise ValueError(f"Not a directory: {evaluator_folder_abs}")

    # Auto-detect multi_metrics if not specified and not remote_url and folder is given
    detected_multi_metrics = multi_metrics
    folder_for_loading = None
    metric_folders_for_loading = None

    if not remote_url and evaluator_folder_abs:
        has_main_py = os.path.exists(os.path.join(evaluator_folder_abs, "main.py"))
        if has_main_py and not multi_metrics:  # If user says not multi_metrics, but main.py is at root
            py_files = list(Path(evaluator_folder_abs).glob("*.py"))
            if len(py_files) > 1:  # Heuristic: if multiple .py files at root with main.py, likely multi-metric
                logger.info("Auto-detecting multi-metrics mode for deploy_folder_evaluation.")
                detected_multi_metrics = True

        if detected_multi_metrics:
            folder_for_loading = evaluator_folder_abs
        else:  # Prepare metric_folders list
            metric_folders_for_loading = []
            if has_main_py:  # Single metric in the root folder
                metric_folders_for_loading.append(f"{os.path.basename(evaluator_folder_abs)}={evaluator_folder_abs}")
            else:  # Look for subdirectories
                for item in os.listdir(evaluator_folder_abs):
                    item_path = os.path.join(evaluator_folder_abs, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "main.py")):
                        metric_folders_for_loading.append(f"{item}={item_path}")
                if not metric_folders_for_loading:
                    raise ValueError(
                        f"No valid metrics found in {evaluator_folder_abs} for non-multi-metric deployment."
                    )

    return create_evaluation(
        evaluator_id=evaluator_id,
        metric_folders=metric_folders_for_loading,
        multi_metrics=detected_multi_metrics,  # Use the detected or passed-in multi_metrics
        folder=folder_for_loading,
        python_code_to_evaluate=python_code_to_evaluate,  # None for this helper
        display_name=display_name,
        description=description,
        force=force,
        huggingface_dataset=huggingface_dataset,
        huggingface_split=huggingface_split,
        huggingface_message_key_map=huggingface_message_key_map,
        huggingface_response_key=huggingface_response_key,
        huggingface_prompt_key=huggingface_prompt_key,
        remote_url=remote_url,
    )
