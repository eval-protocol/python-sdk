import asyncio
import base64
import json
import os
import tempfile
import time
import zipfile
from typing import Any, Callable, Dict, List, Optional

import requests

from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.types.remote_rollout_processor import DataLoaderConfig, RolloutMetadata

from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig
from .tracing_utils import default_fireworks_output_data_loader, build_init_request, update_row_with_remote_trace


class GithubActionRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that dispatches and monitors a GitHub Actions workflow per evaluation row.

    Expected GitHub Actions workflow:
    - Workflow dispatch with inputs: model, messages_b64, tools_b64, rollout_id, etc.
    - Workflow uploads artifact named "rollout-trace-{rollout_id}" containing trace JSON
    - Trace JSON format: {"status": "success"|"error", "messages": [...], "tools": [...], "error": str?}
    """

    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        workflow_id: str,
        ref: str = "main",
        model_base_url: str = "https://tracing.fireworks.ai",
        poll_interval: float = 3.0,
        timeout_seconds: float = 1800.0,
        output_data_loader: Optional[Callable[[DataLoaderConfig], DynamicDataLoader]] = None,
    ):
        self.owner = owner
        self.repo = repo
        self.workflow_id = workflow_id
        self.ref = ref
        self.model_base_url = model_base_url
        _ep_model_base_url = os.getenv("EP_MODEL_BASE_URL")
        if _ep_model_base_url:
            self.model_base_url = _ep_model_base_url
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
        self._output_data_loader = output_data_loader or default_fireworks_output_data_loader

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_run(self, run_id: int) -> Dict[str, Any]:
        """Get status of a specific workflow run."""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs/{run_id}"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def _get_run_artifacts(self, run_id: int) -> Dict[str, Any]:
        """Get artifacts for a specific workflow run."""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs/{run_id}/artifacts"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def _download_and_extract_trace(self, artifact_url: str, rollout_id: str) -> Optional[Dict[str, Any]]:
        """Download artifact and extract trace JSON."""
        # Download artifact
        r = requests.get(artifact_url, headers=self._headers(), timeout=60)
        r.raise_for_status()

        # Extract trace JSON
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(r.content)
            tmp_file.flush()

            with zipfile.ZipFile(tmp_file.name, "r") as zip_file:
                trace_filename = f"rollout_trace_{rollout_id}.json"
                if trace_filename in zip_file.namelist():
                    with zip_file.open(trace_filename) as trace_file:
                        return json.loads(trace_file.read().decode("utf-8"))
        return None

    def _apply_trace_data_to_row(
        self, row: EvaluationRow, trace_data: Optional[Dict[str, Any]], workflow_conclusion: Optional[str]
    ) -> EvaluationRow:
        """Apply trace data from GitHub Actions artifact to the evaluation row."""
        # First check workflow conclusion
        if workflow_conclusion != "success":
            if workflow_conclusion == "failure":
                row.rollout_status = Status.rollout_error("GitHub Actions workflow failed")
            elif workflow_conclusion == "cancelled":
                row.rollout_status = Status(
                    code=Status.Code.CANCELLED, message="GitHub Actions workflow was cancelled"
                )
            elif workflow_conclusion == "skipped":
                row.rollout_status = Status(code=Status.Code.CANCELLED, message="GitHub Actions workflow was skipped")
            else:
                row.rollout_status = Status(
                    code=Status.Code.UNKNOWN, message=f"GitHub Actions workflow concluded with '{workflow_conclusion}'"
                )
            return row

        # Workflow succeeded, now check trace data
        if not trace_data:  # No trace data found
            row.rollout_status = Status(code=Status.Code.NOT_FOUND, message="No trace data found")
            return row
        elif trace_data.get("status") == "error":  # Rollout script failed
            error_msg = trace_data.get("error", "Unknown error")
            row.rollout_status = Status.rollout_error(f"Rollout failed: {error_msg}")
            return row
        elif trace_data.get("status") == "success":  # Successful rollout
            trace_messages = trace_data.get("messages", [])

            # if the trace has the same number of messages as the original row, something went wrong
            if len(trace_messages) == len(row.messages):
                row.rollout_status = Status.rollout_error(
                    "Rollout finished with the same number of messages as the original row"
                )
                return row

            row.messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in trace_messages]
            if trace_data.get("tools"):
                row.tools = trace_data["tools"]
            return row
        else:
            row.rollout_status = Status.rollout_error(f"Unknown trace status: {trace_data.get('status')}")
            return row

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        async def _process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            if row.execution_metadata.invocation_id is None:
                raise ValueError("Invocation ID is required in GithubActionRolloutProcessor")
            if row.execution_metadata.experiment_id is None:
                raise ValueError("Experiment ID is required in GithubActionRolloutProcessor")
            if row.execution_metadata.rollout_id is None:
                raise ValueError("Rollout ID is required in GithubActionRolloutProcessor")
            if row.execution_metadata.run_id is None:
                raise ValueError("Run ID is required in GithubActionRolloutProcessor")
            if row.input_metadata.row_id is None:
                raise ValueError("Row ID is required in GithubActionRolloutProcessor")

            init_request = build_init_request(row, config, self.model_base_url)

            def _dispatch_workflow():
                url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/workflows/{self.workflow_id}/dispatches"
                payload = {
                    "ref": self.ref,
                    "inputs": {
                        "model": init_request.model,
                        "metadata": init_request.metadata.model_dump_json(),
                        "messages": json.dumps(init_request.messages),
                        "tools": json.dumps(init_request.tools),
                        "model_base_url": init_request.model_base_url,
                    },
                }
                r = requests.post(url, json=payload, headers=self._headers(), timeout=30)
                r.raise_for_status()

            await asyncio.to_thread(_dispatch_workflow)

            # Wait for GitHub to create the run, then find it by name. TODO: not sure if this is janky
            await asyncio.sleep(5)

            def _get_workflow_runs() -> Dict[str, Any]:
                """Get recent workflow runs for this workflow."""
                url = (
                    f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/workflows/{self.workflow_id}/runs"
                )
                params = {"event": "workflow_dispatch", "branch": self.ref, "per_page": 20}
                r = requests.get(url, params=params, headers=self._headers(), timeout=30)
                r.raise_for_status()
                return r.json()

            runs_data = await asyncio.to_thread(_get_workflow_runs)

            # Find our specific run by name
            target_name = f"rollout:{row.execution_metadata.rollout_id}"
            run_id = None
            for run in runs_data.get("workflow_runs", []):
                if run.get("name") == target_name:
                    run_id = run.get("id")
                    break

            if not run_id:
                row.rollout_status = Status.rollout_error(
                    f"Failed to find workflow run in GHA with rollout_id {row.execution_metadata.rollout_id}"
                )
                row.execution_metadata.duration_seconds = time.perf_counter() - start_time
                return row

            print(f"DEBUG: Found and polling run {run_id} for rollout {row.execution_metadata.rollout_id}")

            # Poll the specific run until completion
            deadline = time.time() + self.timeout_seconds
            workflow_conclusion = None
            # TODO: no clue what to do with workflow_conclusion

            while time.time() < deadline:
                run_data = await asyncio.to_thread(self._get_run, run_id)

                if run_data.get("status") == "completed":
                    # Store the conclusion for later use in trace application
                    # workflow_conclusion = run_data.get("conclusion")
                    break

                await asyncio.sleep(self.poll_interval)
            else:
                row.rollout_status = Status.rollout_error(
                    f"GitHub Actions run timed out after {self.timeout_seconds} seconds"
                )
                row.execution_metadata.duration_seconds = time.perf_counter() - start_time
                return row

            row.execution_metadata.duration_seconds = time.perf_counter() - start_time

            def _update_with_trace() -> None:
                return update_row_with_remote_trace(row, self._output_data_loader, self.model_base_url)

            await asyncio.to_thread(_update_with_trace)
            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await _process_row(r)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]

    def cleanup(self) -> None:
        return None
