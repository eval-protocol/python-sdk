"""
Task Manager for the Agent Evaluation Framework V2.
Coordinates multiple tasks and their associated resources.
"""

import asyncio
import importlib
import json
import logging
import os
import shlex
import socket
import statistics
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import requests

from ..models import TaskDefinitionModel
from .orchestrator import Orchestrator
from .resource_abc import ForkableResource
from .resource_pool import ResourcePool


class TaskManager:
    """
    Manages the execution of multiple agent evaluation tasks.
    Coordinates resources, orchestrators, and execution flows.
    """

    def __init__(self):
        """Initialize the TaskManager with an empty task registry."""
        self.tasks: Dict[str, TaskDefinitionModel] = {}
        self.resource_pool = ResourcePool()
        self.logger = logging.getLogger("TaskManager")
        self.orchestrators: Dict[str, Orchestrator] = {}
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.server_ports: Dict[str, int] = {}
        self.all_server_pids: Set[int] = set()

    def register_task(self, task_definition_or_name, task_definition=None) -> str:
        """
        Register a task with the manager.

        Args:
            task_definition_or_name: Either a TaskDefinitionModel instance (legacy) or task name (new)
            task_definition: TaskDefinitionModel instance when first arg is task name

        Returns:
            task_id: A unique identifier for the registered task
        """
        # Handle both calling patterns for backward compatibility
        if task_definition is None:
            # Legacy call: register_task(task_definition)
            task_def = task_definition_or_name
            task_id = task_def.name
        else:
            # New call: register_task(task_name, task_definition)
            task_id = task_definition_or_name
            task_def = task_definition

        if task_id in self.tasks:
            self.logger.warning(f"Task '{task_id}' is already registered. Overwriting.")

        self.tasks[task_id] = task_def
        self.logger.info(f"Registered task: {task_id}")
        return task_id

    def register_tasks_from_directory(self, directory_path: str) -> List[str]:
        """
        Register all task definition files from a directory.

        Args:
            directory_path: Path to directory containing task definition files

        Returns:
            task_ids: List of task IDs that were successfully registered
        """
        task_ids: List[str] = []
        dir_path = Path(directory_path)

        if not dir_path.exists() or not dir_path.is_dir():
            self.logger.error(f"Directory not found or not a directory: {directory_path}")
            return task_ids

        for file_path in dir_path.glob("*.y*ml"):
            try:
                task_def = self._load_task_from_file(str(file_path))
                if task_def:
                    task_id = self.register_task(task_def)
                    task_ids.append(task_id)
            except Exception as e:
                self.logger.error(f"Error loading task from {file_path}: {e}")

        for file_path in dir_path.glob("*.json"):
            try:
                task_def = self._load_task_from_file(str(file_path))
                if task_def:
                    task_id = self.register_task(task_def)
                    task_ids.append(task_id)
            except Exception as e:
                self.logger.error(f"Error loading task from {file_path}: {e}")

        self.logger.info(f"Registered {len(task_ids)} tasks from {directory_path}")
        return task_ids

    def _load_task_from_file(self, file_path: str) -> Optional[TaskDefinitionModel]:
        """
        Load and validate a task definition from a file.

        Args:
            file_path: Path to the task definition file

        Returns:
            task_def: A validated TaskDefinitionModel instance or None if loading fails
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists() or not file_path_obj.is_file():
            self.logger.error(f"File not found or not a file: {file_path}")
            return None

        try:
            # Try to load as YAML first
            try:
                import yaml

                with open(file_path, "r") as f:
                    task_data = yaml.safe_load(f)
            except ImportError:
                # If PyYAML is not available, try JSON
                with open(file_path, "r") as f:
                    task_data = json.load(f)
            except Exception:
                # If YAML loading fails, try JSON
                with open(file_path, "r") as f:
                    task_data = json.load(f)

            # Store the original file path for downstream use
            task_data["task_def_path"] = str(file_path_obj.resolve())

            # Validate with Pydantic model
            task_def = TaskDefinitionModel.model_validate(task_data)
            return task_def
        except Exception as e:
            self.logger.error(f"Error loading task definition from {file_path}: {e}")
            return None

    def _find_free_port(self) -> int:
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _wait_for_server_health(self, health_url: str, timeout: int = 30) -> bool:
        """Wait for a server to become healthy by polling its health endpoint."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"Server is healthy at {health_url}")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        self.logger.error(f"Server failed to become healthy at {health_url} within {timeout} seconds")
        return False

    def _start_resource_server(self, task_id: str, task_def: TaskDefinitionModel) -> Optional[int]:
        """Start a resource server for a task and return the allocated port."""
        if not task_def.resource_server:
            return None

        # Find a free port
        port = self._find_free_port()

        # Replace {port} placeholder in start command
        start_command = task_def.resource_server.start_command.replace("{port}", str(port))

        # Start the server process
        try:
            self.logger.info(f"Starting resource server for task '{task_id}' on port {port}: {start_command}")
            process = subprocess.Popen(
                shlex.split(start_command),
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

            # Store the process and port
            self.server_processes[task_id] = process
            self.server_ports[task_id] = port
            self.all_server_pids.add(process.pid)

            # Wait for server to become healthy
            health_url = task_def.resource_server.health_check_url.replace("{port}", str(port))
            if self._wait_for_server_health(health_url):
                self.logger.info(f"Resource server for task '{task_id}' is ready on port {port}")
                return port
            else:
                # Server failed to start properly, clean up
                self._stop_resource_server(task_id)
                return None

        except Exception as e:
            self.logger.error(f"Failed to start resource server for task '{task_id}': {e}")
            return None

    def _stop_resource_server(self, task_id: str) -> None:
        """Stop the resource server for a task."""
        if task_id in self.server_processes:
            process = self.server_processes[task_id]
            self.all_server_pids.discard(process.pid)
            try:
                # Try to terminate gracefully first
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(process.pid), 15)  # SIGTERM
                else:
                    process.terminate()

                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't shut down gracefully
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(process.pid), 9)  # SIGKILL
                    else:
                        process.kill()
                    process.wait()

                self.logger.info(f"Stopped resource server for task '{task_id}'")
            except Exception as e:
                self.logger.error(f"Error stopping resource server for task '{task_id}': {e}")

            del self.server_processes[task_id]

        if task_id in self.server_ports:
            del self.server_ports[task_id]

    async def prepare_task(self, task_id: str) -> bool:
        """
        Prepare a task for execution by setting up its resources.

        Args:
            task_id: Identifier of the task to prepare

        Returns:
            success: True if preparation was successful, False otherwise
        """
        if task_id not in self.tasks:
            self.logger.error(f"Task '{task_id}' is not registered.")
            return False

        task_def = self.tasks[task_id]

        # Start resource server if needed
        allocated_port = None
        if task_def.resource_server:
            allocated_port = self._start_resource_server(task_id, task_def)
            if allocated_port is None:
                self.logger.error(f"Failed to start resource server for task '{task_id}'")
                return False

        # Create a modified task definition with updated base_url if a server was started
        effective_task_def = task_def
        if allocated_port is not None:
            # Create a deep copy and update the base_url
            effective_task_def = deepcopy(task_def)
            if hasattr(effective_task_def.base_resource_config, "base_url"):
                # Update existing base_url
                effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"
            elif "base_url" in effective_task_def.base_resource_config:
                # Update base_url in dict
                effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"
            else:
                # Add base_url if it doesn't exist
                effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"

        # Create an orchestrator for this specific task
        orchestrator = Orchestrator(task_definition=effective_task_def)
        self.orchestrators[task_id] = orchestrator

        # Prepare the resources for this task
        try:
            # Resource setup is handled by the orchestrator
            await orchestrator.setup_base_resource()
            return True
        except Exception as e:
            self.logger.error(f"Error preparing resources for task '{task_id}': {e}")
            # Clean up server if we started one
            if allocated_port is not None:
                self._stop_resource_server(task_id)
            return False

    async def execute_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute a registered task.

        Args:
            task_id: Identifier of the task to execute

        Returns:
            result: Dictionary containing execution results or None if execution fails
        """
        if task_id not in self.tasks:
            self.logger.error(f"Task '{task_id}' is not registered.")
            return None

        if task_id not in self.orchestrators:
            self.logger.info(f"Task '{task_id}' orchestrator not initialized. Preparing task...")
            success = await self.prepare_task(task_id)
            if not success:
                self.logger.error(f"Failed to prepare task '{task_id}'.")
                return None

        orchestrator = self.orchestrators[task_id]

        try:
            self.logger.info(f"Executing task '{task_id}'...")
            result = await orchestrator.execute_task_poc()
            self.logger.info(f"Task '{task_id}' execution completed.")
            return result
        except Exception as e:
            self.logger.error(f"Error executing task '{task_id}': {e}", exc_info=True)
            return {"error": str(e)}

    async def execute_tasks(
        self,
        task_ids: Optional[List[str]] = None,
        parallel: bool = False,
        max_concurrency: int = 3,
        num_rollouts_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute multiple tasks sequentially or in parallel.

        Args:
            task_ids: List of task IDs to execute. If None, execute all registered tasks.
            parallel: If True, execute tasks in parallel; otherwise, execute sequentially
            max_concurrency: Maximum number of tasks to execute in parallel
            num_rollouts_override: Override the number of rollouts for each task

        Returns:
            results: Dictionary mapping task IDs to execution results (aggregated if multiple rollouts)
        """
        task_ids_to_execute = task_ids if task_ids is not None else list(self.tasks.keys())

        # Validate task IDs
        valid_task_ids = [tid for tid in task_ids_to_execute if tid in self.tasks]
        if len(valid_task_ids) != len(task_ids_to_execute):
            invalid_task_ids = set(task_ids_to_execute) - set(valid_task_ids)
            self.logger.warning(f"Some task IDs are not registered: {invalid_task_ids}")

        if not valid_task_ids:
            self.logger.error("No valid tasks to execute.")
            return {}

        results: Dict[str, Any] = {}

        # For each task, determine how many rollouts to execute
        for task_id in valid_task_ids:
            task_def = self.tasks[task_id]

            # Check if this is a data-driven evaluation
            if task_def.dataset_path:
                # Data-driven evaluation: load samples from dataset
                samples = self._load_dataset_samples(task_def.dataset_path)
                if not samples:
                    results[task_id] = {"error": "Failed to load dataset or dataset is empty"}
                    continue

                self.logger.info(
                    f"Executing data-driven evaluation for task '{task_id}': {len(samples)} samples, {task_def.num_rollouts_per_sample} rollouts per sample"
                )
                rollout_results = await self._execute_data_driven_rollouts(
                    task_id, samples, task_def.num_rollouts_per_sample, max_concurrency
                )
            else:
                # Traditional evaluation: fixed number of rollouts
                num_rollouts = num_rollouts_override if num_rollouts_override is not None else task_def.num_rollouts

                if num_rollouts == 1:
                    # Single rollout - existing behavior
                    if await self.prepare_task(task_id):
                        results[task_id] = await self.execute_task(task_id)
                    else:
                        results[task_id] = {"error": "Task preparation failed"}
                    continue
                else:
                    # Multiple rollouts - batch execution
                    self.logger.info(f"Executing {num_rollouts} rollouts for task '{task_id}'")
                    rollout_results = await self._execute_batch_rollouts(task_id, num_rollouts, max_concurrency)

            # Aggregate results (for both data-driven and traditional batch execution)
            if rollout_results:
                aggregated_result = self._aggregate_results(rollout_results)
                results[task_id] = aggregated_result

                # Always save detailed results to .jsonl file (including failed rollouts for analysis)
                try:
                    detailed_file_path = self._save_detailed_results(task_id, aggregated_result)
                    self.logger.info(f"Detailed results saved to: {detailed_file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save detailed results for task '{task_id}': {e}")
            else:
                results[task_id] = {"error": "All rollouts failed"}

        return results

    async def _execute_batch_rollouts(
        self, task_id: str, num_rollouts: int, max_concurrency: int
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple rollouts for a single task in parallel.

        Args:
            task_id: The base task ID
            num_rollouts: Number of rollouts to execute
            max_concurrency: Maximum number of concurrent rollouts

        Returns:
            List of results from each rollout
        """
        task_def = self.tasks[task_id]
        rollout_results = []

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_single_rollout(rollout_index: int):
            """Execute a single rollout with its own server instance."""
            rollout_task_id = f"{task_id}_rollout_{rollout_index}"

            async with semaphore:
                try:
                    # Start resource server if needed for this rollout
                    allocated_port = None
                    if task_def.resource_server:
                        allocated_port = self._start_resource_server(rollout_task_id, task_def)
                        if allocated_port is None:
                            self.logger.error(
                                f"Failed to start resource server for rollout {rollout_index} of task '{task_id}'"
                            )
                            return {"error": f"Failed to start resource server for rollout {rollout_index}"}

                    # Create effective task definition with updated base_url if needed
                    effective_task_def = task_def
                    if allocated_port is not None:
                        effective_task_def = deepcopy(task_def)
                        if hasattr(effective_task_def.base_resource_config, "base_url"):
                            effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"
                        elif "base_url" in effective_task_def.base_resource_config:
                            effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"
                        else:
                            effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"

                    # Create orchestrator for this rollout
                    orchestrator = Orchestrator(task_definition=effective_task_def)

                    # Setup and execute
                    await orchestrator.setup_base_resource()
                    result = await orchestrator.execute_task_poc()

                    # Cleanup orchestrator resources
                    if orchestrator.base_resource:
                        await orchestrator.base_resource.close()

                    # Stop the resource server for this rollout
                    if allocated_port is not None:
                        self._stop_resource_server(rollout_task_id)

                    # Handle case where result is None
                    if result is None:
                        result = {"error": "Execution returned None"}

                    # Handle new orchestrator format that includes reward_function_inputs
                    reward_function_inputs = None
                    if isinstance(result, dict) and "evaluation_result" in result:
                        # New format with separate evaluation_result and reward_function_inputs
                        reward_function_inputs = result.get("reward_function_inputs")
                        result = result["evaluation_result"]

                    # Convert EvaluateResult to dict if needed
                    if hasattr(result, "model_dump"):
                        # Pydantic model - convert to dict
                        result = result.model_dump()  # type: ignore[call-arg]
                    elif hasattr(result, "dict"):
                        # Older pydantic models
                        result = result.dict()  # type: ignore[call-arg]
                    # If it's already a dict, leave it as is

                    # Add reward function inputs to the result for JSONL trajectory storage
                    if reward_function_inputs is not None and isinstance(result, dict):
                        result["reward_function_inputs"] = reward_function_inputs

                    score = result.get("score", "N/A") if isinstance(result, dict) else "N/A"
                    self.logger.info(f"Rollout {rollout_index} of task '{task_id}' completed with score: {score}")
                    return result

                except Exception as e:
                    error_msg = f"Error in rollout {rollout_index} of task '{task_id}': {e}"
                    self.logger.error(error_msg, exc_info=True)

                    # Capture server logs if available for debugging
                    if rollout_task_id in self.server_processes:
                        process = self.server_processes[rollout_task_id]
                        try:
                            stdout, stderr = process.communicate(timeout=1)
                            if stdout:
                                self.logger.error(f"Server stdout for rollout {rollout_index}: {stdout.decode()}")
                            if stderr:
                                self.logger.error(f"Server stderr for rollout {rollout_index}: {stderr.decode()}")
                        except Exception:
                            pass  # Ignore errors in log capture

                    # Cleanup on error
                    if allocated_port is not None:
                        self._stop_resource_server(rollout_task_id)
                    return {"error": str(e)}

        # Execute all rollouts concurrently
        rollout_tasks = [execute_single_rollout(i) for i in range(num_rollouts)]
        rollout_results_raw = await asyncio.gather(*rollout_tasks)
        # Normalize to list of dicts for typing purposes where possible
        rollout_results: List[Dict[str, Any]] = []
        for item in rollout_results_raw:
            if isinstance(item, dict):
                rollout_results.append(item)
            else:
                rollout_results.append({"result": item})

        # Log failed rollouts but return all results for comprehensive analysis
        successful_results = [r for r in rollout_results if not (isinstance(r, dict) and "error" in r)]
        failed_count = len(rollout_results) - len(successful_results)

        if failed_count > 0:
            self.logger.warning(f"{failed_count} out of {num_rollouts} rollouts failed for task '{task_id}'")

        # Return all results (successful and failed) for comprehensive logging
        return rollout_results

    def _load_dataset_samples(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load samples from a JSONL dataset file.

        Args:
            dataset_path: Path to the JSONL dataset file

        Returns:
            List of sample dictionaries loaded from the dataset
        """
        try:
            samples = []
            # Support both absolute and relative paths
            if not os.path.isabs(dataset_path):
                # Make relative paths relative to the current working directory
                dataset_path = os.path.abspath(dataset_path)

            if not os.path.exists(dataset_path):
                self.logger.error(f"Dataset file not found: {dataset_path}")
                return []

            with open(dataset_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON on line {line_num} in {dataset_path}: {e}")
                        continue

            self.logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
            return samples

        except Exception as e:
            self.logger.error(f"Error loading dataset from {dataset_path}: {e}")
            return []

    async def _execute_data_driven_rollouts(
        self,
        task_id: str,
        samples: List[Dict[str, Any]],
        rollouts_per_sample: int,
        max_concurrency: int,
    ) -> List[Dict[str, Any]]:
        """
        Execute data-driven rollouts where each sample from the dataset is used for multiple rollouts.

        Args:
            task_id: The base task ID
            samples: List of samples from the dataset
            rollouts_per_sample: Number of rollouts to execute per sample
            max_concurrency: Maximum number of concurrent rollouts

        Returns:
            List of results from all rollouts across all samples
        """
        task_def = self.tasks[task_id]
        all_rollout_results = []

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_single_rollout(sample_index: int, rollout_index: int, sample_data: Dict[str, Any]):
            """Execute a single rollout with sample data."""
            rollout_task_id = f"{task_id}_sample_{sample_index}_rollout_{rollout_index}"

            async with semaphore:
                try:
                    # Start resource server if needed for this rollout
                    allocated_port = None
                    if task_def.resource_server:
                        allocated_port = self._start_resource_server(rollout_task_id, task_def)
                        if allocated_port is None:
                            self.logger.error(
                                f"Failed to start resource server for rollout {rollout_index} of sample {sample_index} for task '{task_id}'"
                            )
                            return {
                                "error": f"Failed to start resource server for sample {sample_index}, rollout {rollout_index}",
                                "sample_data": sample_data,
                            }

                    # Create effective task definition with updated base_url if needed
                    effective_task_def = task_def
                    if allocated_port is not None:
                        effective_task_def = deepcopy(task_def)
                        if hasattr(effective_task_def.base_resource_config, "base_url"):
                            effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"
                        elif "base_url" in effective_task_def.base_resource_config:
                            effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"
                        else:
                            effective_task_def.base_resource_config["base_url"] = f"http://localhost:{allocated_port}"

                    # Create orchestrator for this rollout
                    orchestrator = Orchestrator(task_definition=effective_task_def)

                    # Setup and execute with sample data
                    await orchestrator.setup_base_resource()
                    result = await orchestrator.execute_task_poc(sample_data=sample_data)

                    # Cleanup orchestrator resources
                    if orchestrator.base_resource:
                        await orchestrator.base_resource.close()

                    # Stop the resource server for this rollout
                    if allocated_port is not None:
                        self._stop_resource_server(rollout_task_id)

                    # Handle case where result is None
                    if result is None:
                        result = {"error": "Execution returned None"}

                    # Handle new orchestrator format that includes reward_function_inputs
                    reward_function_inputs = None
                    if isinstance(result, dict) and "evaluation_result" in result:
                        # New format with separate evaluation_result and reward_function_inputs
                        reward_function_inputs = result.get("reward_function_inputs")
                        result = result["evaluation_result"]

                    # Convert EvaluateResult to dict if needed
                    if hasattr(result, "model_dump"):
                        # Pydantic model - convert to dict
                        result = result.model_dump()  # type: ignore[call-arg]
                    elif hasattr(result, "dict"):
                        # Older pydantic models
                        result = result.dict()  # type: ignore[call-arg]
                    # If it's already a dict, leave it as is

                    # Add reward function inputs to the result for JSONL trajectory storage
                    if reward_function_inputs is not None and isinstance(result, dict):
                        result["reward_function_inputs"] = reward_function_inputs

                    # Add sample metadata to the result
                    if isinstance(result, dict):
                        result = cast(Dict[str, Any], result)
                        result["sample_data"] = sample_data
                        result["sample_index"] = sample_index
                        result["rollout_index"] = rollout_index

                    score = result.get("score", "N/A") if isinstance(result, dict) else "N/A"
                    self.logger.info(
                        f"Completed rollout {rollout_index} for sample {sample_index} of task '{task_id}' with score: {score}"
                    )
                    return result

                except Exception as e:
                    self.logger.error(
                        f"Error in rollout {rollout_index} for sample {sample_index} of task '{task_id}': {e}",
                        exc_info=True,
                    )

                    # Try to capture server logs on error
                    if allocated_port is not None:
                        try:
                            process = self.server_processes.get(rollout_task_id)
                            if process:
                                stdout, stderr = process.communicate(timeout=1)
                                if stdout:
                                    self.logger.error(
                                        f"Server stdout for sample {sample_index}, rollout {rollout_index}: {stdout.decode()}"
                                    )
                                if stderr:
                                    self.logger.error(
                                        f"Server stderr for sample {sample_index}, rollout {rollout_index}: {stderr.decode()}"
                                    )
                        except Exception:
                            pass  # Ignore errors in log capture

                    # Cleanup on error
                    if allocated_port is not None:
                        self._stop_resource_server(rollout_task_id)
                    return {
                        "error": str(e),
                        "sample_data": sample_data,
                        "sample_index": sample_index,
                        "rollout_index": rollout_index,
                    }

        # Create rollout tasks for all samples
        rollout_tasks = []
        for sample_index, sample_data in enumerate(samples):
            for rollout_index in range(rollouts_per_sample):
                task = execute_single_rollout(sample_index, rollout_index, sample_data)
                rollout_tasks.append(task)

        # Execute all rollouts concurrently
        all_rollout_results = await asyncio.gather(*rollout_tasks)

        # Log summary statistics
        total_rollouts = len(all_rollout_results)
        successful_results = [r for r in all_rollout_results if not (isinstance(r, dict) and "error" in r)]
        failed_count = total_rollouts - len(successful_results)

        if failed_count > 0:
            self.logger.warning(
                f"{failed_count} out of {total_rollouts} total rollouts failed for task '{task_id}' "
                f"({len(samples)} samples x {rollouts_per_sample} rollouts per sample)"
            )

        self.logger.info(
            f"Completed data-driven evaluation for task '{task_id}': "
            f"{len(successful_results)} successful rollouts out of {total_rollouts} total"
        )

        # Return all results (successful and failed) for comprehensive logging
        return all_rollout_results

    def _aggregate_results(self, rollout_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple rollouts into a single summary.

        Args:
            rollout_results: List of individual rollout results

        Returns:
            Aggregated result dictionary
        """
        if not rollout_results:
            return {"error": "No successful rollouts to aggregate"}

        # Separate successful and failed results
        successful_results = []
        failed_results = []
        scores = []

        for result in rollout_results:
            if isinstance(result, dict) and result.get("error") is not None:
                failed_results.append(result)
            elif isinstance(result, dict) and "score" in result:
                scores.append(result["score"])
                successful_results.append(result)
            else:
                # Handle unexpected result format
                failed_results.append({"error": f"Invalid result format: {result}"})

        if not scores:
            # Even with no successful rollouts, we still want to save failed rollout data
            aggregated_result = {
                "aggregated": True,
                "num_rollouts": len(rollout_results),
                "total_rollouts": len(rollout_results),  # For compatibility with tests
                "successful_rollouts": 0,
                "failed_rollouts": len(failed_results),
                "success_rate": 0.0,
                "avg_score": 0.0,
                "average_score": 0.0,  # For compatibility with tests
                "std_dev": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "score": 0.0,  # For compatibility with existing logging
                "individual_scores": [],
                "individual_results": rollout_results,  # Include all results (failed)
                "detailed_results": rollout_results,  # For compatibility with tests
                "successful_results": [],
                "failed_results": failed_results,
                "timestamp": datetime.now().isoformat(),
                "error": "No valid scores found in rollout results",
            }
            return aggregated_result

        # Calculate aggregated statistics
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        success_rate = len(scores) / len(rollout_results) if rollout_results else 0

        # Calculate standard deviation
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        aggregated_result = {
            "aggregated": True,
            "num_rollouts": len(rollout_results),
            "total_rollouts": len(rollout_results),  # For compatibility with tests
            "successful_rollouts": len(scores),
            "failed_rollouts": len(failed_results),
            "success_rate": success_rate,
            "avg_score": avg_score,
            "average_score": avg_score,  # For compatibility with tests
            "std_dev": std_dev,
            "min_score": min_score,
            "max_score": max_score,
            "score": avg_score,  # For compatibility with existing logging
            "individual_scores": scores,
            "individual_results": rollout_results,  # Include all results (successful and failed)
            "detailed_results": rollout_results,  # For compatibility with tests
            "successful_results": successful_results,
            "failed_results": failed_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Aggregate metrics if available
        if successful_results and "metrics" in successful_results[0]:
            aggregated_metrics = {}
            for metric_name in successful_results[0]["metrics"].keys():
                metric_scores = []
                for result in successful_results:
                    if metric_name in result.get("metrics", {}):
                        metric_result = result["metrics"][metric_name]
                        if isinstance(metric_result, dict) and "score" in metric_result:
                            metric_scores.append(metric_result["score"])
                        elif isinstance(metric_result, (int, float)):
                            metric_scores.append(metric_result)

                if metric_scores:
                    aggregated_metrics[metric_name] = {
                        "avg_score": sum(metric_scores) / len(metric_scores),
                        "min_score": min(metric_scores),
                        "max_score": max(metric_scores),
                        "individual_scores": metric_scores,
                    }

            if aggregated_metrics:
                aggregated_result["aggregated_metrics"] = aggregated_metrics

        return aggregated_result

    def _save_detailed_results(
        self,
        task_id: str,
        aggregated_result: Dict[str, Any],
        output_file: Optional[str] = None,
    ) -> str:
        """
        Save detailed results to a .jsonl file for analysis.

        Args:
            task_id: The task identifier
            aggregated_result: The aggregated result dictionary
            output_file: Optional custom output file path

        Returns:
            The path to the saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prefer evaluation_logs relative to the task definition file
            chosen_dir = None

            task_def = self.tasks.get(task_id)
            if task_def is not None and hasattr(task_def, "task_def_path"):
                try:
                    task_def_path = Path(getattr(task_def, "task_def_path"))
                    base_dir = task_def_path.parent
                    eval_dir = base_dir / "evaluation_logs"
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    chosen_dir = eval_dir
                except Exception as e:
                    self.logger.warning(f"Failed to create evaluation_logs relative to task definition: {e}")

            if chosen_dir is None:
                # Look for or create common evaluation log directories
                possible_log_dirs = [
                    Path("client/evaluation_logs"),
                    Path("evaluation_logs"),
                    Path("logs"),
                    Path("."),  # Fallback to current directory
                ]

                for log_dir in possible_log_dirs:
                    try:
                        log_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        continue
                    if log_dir.exists() and log_dir.is_dir():
                        chosen_dir = log_dir
                        break

                if chosen_dir is None:
                    chosen_dir = Path(".")

            output_path = chosen_dir / f"trajectory_{task_id}_{timestamp}.jsonl"

        else:
            output_path = Path(output_file)

        try:
            self.logger.info("=== TRAJECTORY SAVE DEBUG START ===")
            self.logger.info(f"Saving trajectory data to: {output_path}")
            self.logger.info(f"Chosen directory: {chosen_dir}")
            self.logger.info(f"Individual results count: {len(aggregated_result.get('individual_results', []))}")
            self.logger.info(f"Output path parent directory exists: {output_path.parent.exists()}")

            # Ensure the directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                # Write summary line
                summary = {
                    "type": "summary",
                    "task_id": task_id,
                    "timestamp": aggregated_result.get("timestamp", datetime.now().isoformat()),
                    "num_rollouts": aggregated_result["num_rollouts"],
                    "successful_rollouts": aggregated_result["successful_rollouts"],
                    "failed_rollouts": aggregated_result.get("failed_rollouts", 0),
                    "success_rate": aggregated_result["success_rate"],
                    "avg_score": aggregated_result["avg_score"],
                    "std_dev": aggregated_result["std_dev"],
                    "min_score": aggregated_result["min_score"],
                    "max_score": aggregated_result["max_score"],
                }
                f.write(json.dumps(summary) + "\n")
                self.logger.info(f"Wrote summary line to {output_path}")

                # Write individual results
                individual_results = aggregated_result.get("individual_results", [])
                self.logger.info(f"Processing {len(individual_results)} individual results")
                for i, result in enumerate(individual_results):
                    self.logger.info(f"Processing individual result {i}: {type(result)} - {len(str(result))} chars")

                    # Clean the result for JSON serialization
                    clean_result = {}
                    for key, value in result.items():
                        if key == "reward_function_inputs" and isinstance(value, dict):
                            # Clean the reward function inputs
                            clean_inputs = {}
                            for input_key, input_value in value.items():
                                if input_key == "state" and isinstance(input_value, dict):
                                    # Clean the state by removing non-serializable objects
                                    clean_state = {}
                                    for state_key, state_value in input_value.items():
                                        if state_key == "resource":
                                            # Replace resource object with a string representation
                                            clean_state[state_key] = f"<{type(state_value).__name__}>"
                                        else:
                                            clean_state[state_key] = state_value
                                    clean_inputs[input_key] = clean_state
                                else:
                                    clean_inputs[input_key] = input_value
                            clean_result[key] = clean_inputs
                        else:
                            clean_result[key] = value

                    detailed_result = {
                        "type": "individual_result",
                        "task_id": task_id,
                        "rollout_index": i,
                        "timestamp": datetime.now().isoformat(),
                        **clean_result,
                    }
                    f.write(json.dumps(detailed_result) + "\n")
                    self.logger.info(f"Wrote individual result {i} to {output_path}")

                # Force flush to ensure data is written
                f.flush()
                import os

                os.fsync(f.fileno())

            self.logger.info(f"Successfully saved trajectory data to: {output_path}")
            self.logger.info(f"Trajectory file size: {output_path.stat().st_size} bytes")
            self.logger.info("=== TRAJECTORY SAVE DEBUG END ===")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Failed to save detailed results to {output_path}: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def cleanup_all_servers(self) -> None:
        """A more robust cleanup that terminates any tracked server process."""
        if not self.all_server_pids:
            self.logger.info("No tracked server PIDs to clean up.")
            return

        self.logger.info(f"Performing robust cleanup of all {len(self.all_server_pids)} tracked server PIDs.")
        # Iterate over a copy as the set will be modified
        for pid in list(self.all_server_pids):
            try:
                # Find the task_id associated with this PID for logging
                task_id = "unknown_task"
                for tid, proc in self.server_processes.items():
                    if proc.pid == pid:
                        task_id = tid
                        break

                self.logger.warning(
                    f"Force-cleaning up potentially orphaned server process for task '{task_id}' (PID: {pid})."
                )
                # Use the same killpg logic
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(pid), 9)  # Use SIGKILL for forceful cleanup
                else:
                    os.kill(pid, 9)
                self.all_server_pids.discard(pid)

            except ProcessLookupError:
                # Process already gone, which is fine
                self.all_server_pids.discard(pid)
            except Exception as e:
                self.logger.error(f"Error during robust cleanup of PID {pid}: {e}")
                self.all_server_pids.discard(pid)

    async def cleanup(self, task_ids: Optional[List[str]] = None) -> None:
        """
        Clean up resources for specified tasks or all tasks.

        Args:
            task_ids: List of task IDs to clean up. If None, clean up all tasks.
        """
        task_ids_to_cleanup = task_ids if task_ids is not None else list(self.orchestrators.keys())

        for task_id in task_ids_to_cleanup:
            # Stop resource server if running
            self._stop_resource_server(task_id)

            # Clean up orchestrator resources
            if task_id in self.orchestrators:
                orchestrator = self.orchestrators[task_id]
                if orchestrator.base_resource:
                    try:
                        await orchestrator.base_resource.close()
                        self.logger.info(f"Cleaned up resources for task '{task_id}'.")
                    except Exception as e:
                        self.logger.error(f"Error cleaning up resources for task '{task_id}': {e}")
                del self.orchestrators[task_id]

        # Perform robust cleanup of any remaining orphaned processes
        self.cleanup_all_servers()
