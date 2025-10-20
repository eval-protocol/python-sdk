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
        remote_base_url="http://35.209.134.123:3000",
        model_base_url="https://tracing.fireworks.ai",
        timeout_seconds=1800,
        output_data_loader=default_fireworks_output_data_loader,
        disable_elastic_search_setup=True,
        elastic_search_config=create_elasticsearch_config_from_env(),
    ),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    max_concurrent_rollouts=3,
)
# async def test_swebench_remote(row: EvaluationRow) -> EvaluationRow:
#     """Evaluate SWE-bench instance by reading results from Elasticsearch."""
#     import logging
#     logger = logging.getLogger(__name__)

#     rollout_id = row.execution_metadata.rollout_id
#     logger.info(f"[DEBUG] Processing rollout_id: {rollout_id}")

#     if not rollout_id:
#         logger.warning("[DEBUG] No rollout_id, returning early")
#         return row

#     try:
#         from eval_protocol.log_utils.elasticsearch_client import ElasticsearchClient

#         es_config = create_elasticsearch_config_from_env()
#         es_client = ElasticsearchClient(es_config)
#         logger.info(f"[DEBUG] ES client created for index: {es_config.index_name}")

#         # Search for EVAL_RESULT log by message prefix
#         query = {"match": {"rollout_id": rollout_id}}
#         search_results = es_client.search(query=query, size=50)  # Get more to find EVAL_RESULT
#         logger.info(f"[DEBUG] Total logs: {search_results['hits']['total']['value']}")

#         # Filter for EVAL_RESULT in Python
#         if search_results and search_results["hits"]["total"]["value"] > 0:
#             for hit in search_results["hits"]["hits"]:
#                 message = hit["_source"].get("message", "")

#                 if message.startswith("EVAL_RESULT:"):
#                     logger.info(f"[DEBUG] Found EVAL_RESULT message!")
#                     result_json = message.replace("EVAL_RESULT:", "")
#                     row.evaluation_result = EvaluateResult.model_validate_json(result_json)
#                     logger.info(f"[DEBUG] Attached evaluation_result: score={row.evaluation_result.score}")
#                     break
#             else:
#                 logger.warning("[DEBUG] EVAL_RESULT message not found in logs")
#         else:
#             logger.warning("[DEBUG] No logs found for rollout")

#         logger.info(f"[DEBUG] Searching ES for EVAL_RESULT")
#         import asyncio
#         search_results = None
#         for attempt in range(5):
#             search_results = es_client.search(query=query, size=1)
#             if search_results and search_results["hits"]["total"]["value"] > 0:
#                 logger.info(f"[DEBUG] Found result on attempt {attempt + 1}")
#                 break
#             logger.info(f"[DEBUG] Attempt {attempt + 1}: No hits, retrying in 1s...")
#             await asyncio.sleep(1)

#         logger.info(f"[DEBUG] Final: ES returned {search_results['hits']['total']['value'] if search_results else 0} hits")
#         debug_query = {"match": {"rollout_id": rollout_id}}
#         debug_results = es_client.search(query=debug_query, size=26)
#         logger.info(f"[DEBUG] Total logs for {rollout_id}: {debug_results['hits']['total']['value']}")

#         if debug_results["hits"]["total"]["value"] > 0:
#             for hit in debug_results["hits"]["hits"]:
#                 msg = hit["_source"].get("message", "")[:80]
#                 logger.info(f"[DEBUG] Sample message: {msg}")
#         else:
#             logger.warning("[DEBUG] No logs at all for this rollout_id!")
#         if search_results and search_results["hits"]["total"]["value"] > 0:
#             hit = search_results["hits"]["hits"][0]["_source"]
#             message = hit.get("message", "")
#             logger.info(f"[DEBUG] Found message: {message[:100]}...")

#             if message.startswith("EVAL_RESULT:"):
#                 result_json = message.replace("EVAL_RESULT:", "")
#                 logger.info(f"[DEBUG] Parsing EvaluateResult JSON")

#                 if result_json != "null":
#                     # Deserialize directly to EvaluateResult
#                     row.evaluation_result = EvaluateResult.model_validate_json(result_json)
#                     logger.info(f"[DEBUG] Attached evaluation_result: score={row.evaluation_result.score}, reason={row.evaluation_result.reason}")
#                 else:
#                     logger.warning("[DEBUG] Result was null (no resolved status available)")
#             else:
#                 logger.warning(f"[DEBUG] Message doesn't start with EVAL_RESULT: {message[:50]}")
#         else:
#             logger.warning("[DEBUG] No EVAL_RESULT found in Elasticsearch")

#     except Exception as e:
#         logger.error(f"[DEBUG] Exception in test: {e}", exc_info=True)

#     logger.info(f"[DEBUG] Returning row, has evaluation_result: {row.evaluation_result is not None}")
#     return row


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
