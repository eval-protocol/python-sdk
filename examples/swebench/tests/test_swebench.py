from typing import List
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor, create_elasticsearch_config_from_env
from eval_protocol.pytest.tracing_utils import default_fireworks_output_data_loader


def rows_from_indices(count: int) -> List[EvaluationRow]:
    out: List[EvaluationRow] = []
    for idx in range(count):
        out.append(
            EvaluationRow(
                messages=[],
                input_metadata={
                    "row_id": str(idx),
                    "instance_index": str(idx),
                },
            )
        )
    return out


def rows() -> List[EvaluationRow]:
    # Generate 10 rows by index; server maps index -> dataset instance via --slice
    return rows_from_indices(2)


# -------------------- Harness result attachment (UI pass/fail) --------------------
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://127.0.0.1:3000",
        model_base_url="https://tracing.fireworks.ai",
        timeout_seconds=1800,
        output_data_loader=default_fireworks_output_data_loader,
        disable_elastic_search_setup=True,
        elastic_search_config=create_elasticsearch_config_from_env(),
    ),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    max_concurrent_rollouts=3,
)
async def test_swebench_remote(row: EvaluationRow) -> EvaluationRow:
    """Evaluate SWE-bench instance by reading results from Elasticsearch."""
    import logging

    logger = logging.getLogger(__name__)

    rollout_id = row.execution_metadata.rollout_id
    if not rollout_id:
        return row

    # Query Elasticsearch for results logged by server
    try:
        from eval_protocol.log_utils.elasticsearch_client import ElasticsearchClient

        es_config = create_elasticsearch_config_from_env()
        es_client = ElasticsearchClient(es_config)

        # Search for results log from this rollout
        query = {"bool": {"must": [{"term": {"rollout_id.keyword": rollout_id}}, {"exists": {"field": "results"}}]}}

        search_results = es_client.es.search(index=es_config.index_name, query=query, size=1)

        if search_results["hits"]["total"]["value"] > 0:
            hit = search_results["hits"]["hits"][0]["_source"]
            results_data = hit.get("results", {})
            resolved = results_data.get("resolved")
            instance_id = results_data.get("instance_id")

            if resolved is not None:
                row.evaluation_result = EvaluateResult(
                    score=1.0 if resolved else 0.0,
                    reason=f"instance={instance_id}, resolved={resolved}",
                    is_score_valid=True,
                    metrics={
                        "resolved": MetricResult(
                            score=1.0 if resolved else 0.0,
                            is_score_valid=True,
                            reason=f"resolved={resolved}",
                            value=int(resolved),
                        )
                    },
                )
    except Exception as e:
        logger.warning(f"Could not read results from Elasticsearch: {e}")

    return row
