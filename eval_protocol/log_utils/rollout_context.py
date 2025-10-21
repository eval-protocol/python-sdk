import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import contextvars


# Context variables used to correlate logs with rollouts under concurrency
current_rollout_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("ep_rollout_id", default=None)
current_rollout_ids: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "ep_rollout_ids", default=None
)
current_experiment_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("ep_experiment_id", default=None)
current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("ep_run_id", default=None)


class ContextRolloutIdFilter(logging.Filter):
    """
    Logging filter that injects correlation fields into a LogRecord from ContextVars.

    The filter is intended to be attached ONLY to external sink handlers (e.g.,
    Fireworks or Elasticsearch). If there is no active rollout context, it drops
    the record for that handler to avoid shipping uncorrelated logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        rollout_id = current_rollout_id.get()
        if not rollout_id:
            # Allow explicit rollout IDs on the record or via environment fallback.
            rollout_id = getattr(record, "rollout_id", None) or os.getenv("EP_ROLLOUT_ID")
        if not rollout_id:
            # No correlation context → do not emit to external sink
            return False

        # Inject primary correlation fields
        setattr(record, "rollout_id", rollout_id)

        rollout_ids = current_rollout_ids.get()
        if rollout_ids:
            setattr(record, "rollout_ids", rollout_ids)

        experiment_id = current_experiment_id.get()
        if experiment_id:
            setattr(record, "experiment_id", experiment_id)

        run_id = current_run_id.get()
        if run_id:
            setattr(record, "run_id", run_id)

        return True


@asynccontextmanager
async def rollout_logging_context(
    rollout_id: str,
    *,
    experiment_id: Optional[str] = None,
    run_id: Optional[str] = None,
    rollout_ids: Optional[List[str]] = None,
):
    """
    Async context manager to set correlation ContextVars for the current task.

    Args:
        rollout_id: Primary rollout identifier for correlation.
        experiment_id: Optional experiment ID for tagging.
        run_id: Optional run ID for tagging.
        rollout_ids: Optional list of related rollout IDs (e.g., groupwise mode).
    """
    t_rollout = current_rollout_id.set(rollout_id)
    t_rollouts = current_rollout_ids.set(rollout_ids) if rollout_ids is not None else None
    t_experiment = current_experiment_id.set(experiment_id) if experiment_id is not None else None
    t_run = current_run_id.set(run_id) if run_id is not None else None
    try:
        yield
    finally:
        current_rollout_id.reset(t_rollout)
        if t_rollouts is not None:
            current_rollout_ids.reset(t_rollouts)
        if t_experiment is not None:
            current_experiment_id.reset(t_experiment)
        if t_run is not None:
            current_run_id.reset(t_run)
