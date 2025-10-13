import asyncio
import json
import os
import tempfile
import time
import zipfile
from typing import Any, Dict, List, Optional

import requests

from eval_protocol.models import EvaluationRow, Message, Status

from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig


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
        github_token: Optional[str] = None,
        poll_interval: float = 3.0,
        timeout_seconds: float = 1800.0,
    ):
        self._owner = owner
        self._repo = repo
        self._workflow_id = workflow_id
        self._ref = ref
        self._poll_interval = poll_interval
        self._timeout_seconds = timeout_seconds
        self._token = github_token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        async def _process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            # Extract model
            model: Optional[str] = None
            if row.input_metadata and row.input_metadata.completion_params:
                model = row.input_metadata.completion_params.get("model")
            if model is None and config.completion_params:
                model = config.completion_params.get("model")
            if model is None:
                raise ValueError("Model must be provided")

            # Extract user prompt (first user message)
            user_prompt = None
            for msg in row.messages:
                if hasattr(msg, "role"):
                    if msg.role == "user":
                        user_prompt = msg.content
                        break
                elif isinstance(msg, dict):
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content")
                        break

            if not user_prompt:
                raise ValueError("At least one user message is required")

            # Prepare workflow inputs
            inputs = {
                "model": model,
                "rollout_id": row.execution_metadata.rollout_id,
                "prompt": user_prompt,
            }

            # Dispatch workflow
            def _dispatch():
                url = f"https://api.github.com/repos/{self._owner}/{self._repo}/actions/workflows/{self._workflow_id}/dispatches"
                payload = {"ref": self._ref, "inputs": inputs}
                r = requests.post(url, json=payload, headers=self._headers(), timeout=30)
                r.raise_for_status()

            await asyncio.to_thread(_dispatch)

            # Poll for completion
            deadline = time.time() + self._timeout_seconds
            run_id = None

            while time.time() < deadline:

                def _list_runs():
                    url = f"https://api.github.com/repos/{self._owner}/{self._repo}/actions/workflows/{self._workflow_id}/runs"
                    params = {"event": "workflow_dispatch", "branch": self._ref, "per_page": 10}
                    r = requests.get(url, params=params, headers=self._headers(), timeout=30)
                    r.raise_for_status()
                    return r.json()

                runs_data = await asyncio.to_thread(_list_runs)
                runs = runs_data.get("workflow_runs", [])

                # Find our run (prefer by name, fallback to newest)
                preferred_name = f"rollout-{row.execution_metadata.rollout_id}"
                candidate_run = None
                for r in runs:
                    if r.get("name") == preferred_name:
                        candidate_run = r
                        break
                if not candidate_run and runs:
                    candidate_run = sorted(runs, key=lambda r: r.get("id", 0), reverse=True)[0]

                if candidate_run and candidate_run.get("status") == "completed":
                    run_id = candidate_run.get("id")
                    row.rollout_status = self._map_conclusion_to_status(candidate_run.get("conclusion"))
                    break

                await asyncio.sleep(self._poll_interval)
            else:
                row.rollout_status = Status.rollout_error(
                    f"GitHub Actions run timed out after {self._timeout_seconds} seconds"
                )
                row.execution_metadata.duration_seconds = time.perf_counter() - start_time
                return row

            # Fetch trace from artifacts
            if run_id:

                def _get_artifacts():
                    url = f"https://api.github.com/repos/{self._owner}/{self._repo}/actions/runs/{run_id}/artifacts"
                    r = requests.get(url, headers=self._headers(), timeout=30)
                    r.raise_for_status()
                    return r.json()

                artifacts_data = await asyncio.to_thread(_get_artifacts)
                artifacts = artifacts_data.get("artifacts", [])

                # Find trace artifact
                trace_artifact = None
                for artifact in artifacts:
                    if artifact.get("name") == f"rollout-trace-{row.execution_metadata.rollout_id}":
                        trace_artifact = artifact
                        break

                if trace_artifact:

                    def _download_and_extract():
                        # Download artifact
                        r = requests.get(trace_artifact["archive_download_url"], headers=self._headers(), timeout=60)
                        r.raise_for_status()

                        # Extract trace JSON
                        with tempfile.NamedTemporaryFile() as tmp_file:
                            tmp_file.write(r.content)
                            tmp_file.flush()

                            with zipfile.ZipFile(tmp_file.name, "r") as zip_file:
                                trace_filename = f"rollout_trace_{row.execution_metadata.rollout_id}.json"
                                if trace_filename in zip_file.namelist():
                                    with zip_file.open(trace_filename) as trace_file:
                                        return json.loads(trace_file.read().decode("utf-8"))
                        return None

                    trace_data = await asyncio.to_thread(_download_and_extract)

                    if trace_data and trace_data.get("status") == "success":
                        trace_messages = trace_data.get("messages", [])
                        if len(trace_messages) > len(row.messages):
                            row.messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in trace_messages]
                            if trace_data.get("tools"):
                                row.tools = trace_data["tools"]
                        else:
                            row.rollout_status = Status.rollout_error("Rollout finished with same number of messages")
                    else:
                        error_msg = trace_data.get("error", "Unknown error") if trace_data else "No trace data found"
                        row.rollout_status = Status.rollout_error(f"Rollout failed: {error_msg}")

            row.execution_metadata.duration_seconds = time.perf_counter() - start_time
            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await _process_row(r)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]

    @staticmethod
    def _map_conclusion_to_status(conclusion: Optional[str]) -> Status:
        if conclusion == "success":
            return Status.finished("GitHub Actions workflow succeeded")
        if conclusion in {"failure", "timed_out", "cancelled", "stale"}:
            return Status.rollout_error(f"GitHub Actions workflow concluded with '{conclusion}'")
        return Status(code=Status.Code.UNKNOWN, message=f"GitHub Actions workflow concluded with '{conclusion}'")

    def cleanup(self) -> None:
        return None
