from typing import List
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor, create_elasticsearch_config_from_env

# from eval_protocol.pytest.tracing_utils import default_fireworks_output_data_loader
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
        elastic_search_config=create_elasticsearch_config_from_env(),
    ),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    max_concurrent_rollouts=3,
)
async def test_swebench_remote(row: EvaluationRow) -> EvaluationRow:
    """Evaluate SWE-bench instance by reading results from Elasticsearch."""
    rollout_id = row.execution_metadata.rollout_id
    if not rollout_id:
        return row

    try:
        from eval_protocol.log_utils.elasticsearch_client import ElasticsearchClient

        es_config = create_elasticsearch_config_from_env()
        es_client = ElasticsearchClient(es_config)

        # Get all logs for this rollout and find EVAL_RESULT message
        query = {"match": {"rollout_id": rollout_id}}
        search_results = es_client.search(query=query, size=50)

        if search_results and search_results["hits"]["total"]["value"] > 0:
            for hit in search_results["hits"]["hits"]:
                message = hit["_source"].get("message", "")

                if message.startswith("EVAL_RESULT:"):
                    result_json = message.replace("EVAL_RESULT:", "")
                    row.evaluation_result = EvaluateResult.model_validate_json(result_json)
                    break

    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Could not read results from Elasticsearch: {e}")

    return row
