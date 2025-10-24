from typing import List
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from eval_protocol.utils.evaluation_row_utils import create_rows_from_indices


def rows() -> List[EvaluationRow]:
    return create_rows_from_indices(500)  # All instances


# -------------------- Harness result attachment (UI pass/fail) --------------------
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    max_dataset_rows=2,
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://127.0.0.1:3000",
        model_base_url="https://tracing.fireworks.ai",
        timeout_seconds=1800,
        disable_elastic_search_setup=True,
    ),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    max_concurrent_rollouts=3,
)
async def test_swebench_remote(row: EvaluationRow) -> EvaluationRow:
    """Evaluate SWE-bench instance by reading results from Fireworks tracing logs."""
    import logging

    logger = logging.getLogger(__name__)

    rollout_id = row.execution_metadata.rollout_id
    logger.info(f"[DEBUG] Processing rollout_id: {rollout_id}")

    if not rollout_id:
        logger.warning("[DEBUG] No rollout_id")
        return row

    try:
        from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter

        adapter = FireworksTracingAdapter(base_url="https://tracing.fireworks.ai")
        logger.info("[DEBUG] Created adapter for https://tracing.fireworks.ai")

        # Fetch logs for this rollout
        logger.info(f"[DEBUG] Searching for tag: rollout_id:{rollout_id}")
        log_entries = adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=100, hours_back=24)

        logger.info(f"[DEBUG] Received {len(log_entries)} log entries")
        if log_entries:
            logger.info(f"[DEBUG] Sample messages: {[e.get('message', '')[:50] for e in log_entries[:3]]}")

        # Find EVAL_RESULT message
        found = False
        for entry in log_entries:
            message = entry.get("message", "")
            if message.startswith("EVAL_RESULT:"):
                logger.info("[DEBUG] Found EVAL_RESULT message!")
                result_json = message.replace("EVAL_RESULT:", "")
                logger.info(f"[DEBUG] Parsing JSON: {result_json[:100]}...")

                if result_json != "null":
                    row.evaluation_result = EvaluateResult.model_validate_json(result_json)
                    logger.info(
                        f"[DEBUG] Attached result: score={row.evaluation_result.score}, reason={row.evaluation_result.reason}"
                    )
                    found = True
                break

        if not found:
            logger.warning(f"[DEBUG] No EVAL_RESULT message found in {len(log_entries)} logs")

    except Exception as e:
        logger.error(f"[DEBUG] Exception: {e}", exc_info=True)

    logger.info(f"[DEBUG] Returning row, has evaluation_result: {row.evaluation_result is not None}")
    return row
