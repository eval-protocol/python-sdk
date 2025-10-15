import asyncio
import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests
from datetime import datetime, timezone, timedelta
from eval_protocol.models import EvaluationRow, Status
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.types.remote_rollout_processor import DataLoaderConfig

from .rollout_processor import RolloutProcessor
from .types import RolloutProcessorConfig
from .tracing_utils import default_fireworks_output_data_loader, build_init_request, update_row_with_remote_trace


class GithubActionRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that dispatches and monitors a GitHub Actions workflow per evaluation row.

    Expected GitHub Actions workflow:
    - Workflow dispatch with inputs: model, metadata (JSON), model_base_url
    - Workflow makes API calls that get traced (e.g., via Fireworks tracing proxy)
    - Traces are fetched later via output_data_loader using rollout_id tags

    NOTE: GHA has a rate limit of 5000 requests per hour.
    """

    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        workflow_id: str,
        ref: str = "main",
        model_base_url: str = "https://tracing.fireworks.ai",
        poll_interval: float = 10.0,
        timeout_seconds: float = 1800.0,
        max_find_workflow_retries: int = 5,
        github_token: Optional[str] = None,
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
        self.max_find_workflow_retries = max_find_workflow_retries
        self.github_token = github_token
        self._output_data_loader = output_data_loader or default_fireworks_output_data_loader

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        token = self.github_token or os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "GitHub token is required. Provide it via github_token parameter or GITHUB_TOKEN environment variable"
            )
        headers["Authorization"] = f"Bearer {token}"
        return headers

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        # Calculate max_pages based on number of rows we're processing
        num_rows = len(rows)
        max_pages = (num_rows + 99) // 100  # Round up pages

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
                        "model_base_url": init_request.model_base_url,
                    },
                }
                r = requests.post(url, json=payload, headers=self._headers(), timeout=30)
                r.raise_for_status()

            await asyncio.to_thread(_dispatch_workflow)

            run = None
            target_name = f"rollout:{row.execution_metadata.rollout_id}"

            # Look for runs created in the last 15 minutes (we just dispatched it)
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=15)
            cutoff_iso = cutoff_time.isoformat()

            for attempt in range(self.max_find_workflow_retries):
                try:
                    page = 1
                    while page <= max_pages:

                        def _list_runs():
                            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/workflows/{self.workflow_id}/runs"
                            params = {
                                "event": "workflow_dispatch",
                                "branch": self.ref,
                                "per_page": 100,  # Max per_page is 100, minimize total number of pages
                                "page": page,
                                "created": f">={cutoff_iso}",  # Only look at recent runs
                            }

                            r = requests.get(url, params=params, headers=self._headers(), timeout=30)
                            r.raise_for_status()
                            return r.json()

                        runs_data = await asyncio.to_thread(_list_runs)

                        # Search for our target run in this page
                        for candidate_run in runs_data.get("workflow_runs", []):
                            if candidate_run.get("name") == target_name:
                                run = candidate_run

                        # If we got fewer results than 100, we've reached the end, since we paginate in chunks of 100
                        if len(runs_data.get("workflow_runs", [])) < 100:
                            break

                        page += 1

                    # If no run found, GHA might still be populating it, retry
                    if attempt < self.max_find_workflow_retries - 1:
                        delay = 2**attempt  # Exponential backoff
                        await asyncio.sleep(delay)

                except requests.exceptions.HTTPError as e:
                    # Retry on rate limits (HTTP 429)
                    if e.response and e.response.status_code == 429:
                        if attempt < self.max_find_workflow_retries - 1:
                            delay = 2**attempt  # Exponential backoff
                            await asyncio.sleep(delay)
                        else:
                            # Give up after max attempts
                            raise e
                    else:
                        raise e

            if not run:
                row.rollout_status = Status.rollout_error(
                    f"Failed to find workflow run in GHA with rollout_id {row.execution_metadata.rollout_id}"
                )
                row.execution_metadata.duration_seconds = time.perf_counter() - start_time
                return row

            run_id = run.get("id")
            if not run_id:
                row.rollout_status = Status.rollout_error(
                    f"Failed to find workflow run in GHA with rollout_id {row.execution_metadata.rollout_id}"
                )
                row.execution_metadata.duration_seconds = time.perf_counter() - start_time
                return row

            # Poll the specific run until completion
            deadline = time.time() + self.timeout_seconds

            def _get_run() -> Dict[str, Any]:
                """Get status of a specific workflow run."""
                url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs/{run_id}"
                r = requests.get(url, headers=self._headers(), timeout=30)
                r.raise_for_status()
                return r.json()

            while time.time() < deadline:
                run_data = await asyncio.to_thread(_get_run)

                if run_data.get("status") == "completed":
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

            # Add GitHub Actions run URL to session data
            if run_id:
                github_run_url = f"https://github.com/{self.owner}/{self.repo}/actions/runs/{run_id}"
                if not row.input_metadata.session_data:
                    row.input_metadata.session_data = {}
                row.input_metadata.session_data["github_actions_run_url"] = github_run_url

            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await _process_row(r)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]

    def cleanup(self) -> None:
        return None
