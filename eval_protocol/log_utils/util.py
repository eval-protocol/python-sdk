import os
from eval_protocol.types.remote_rollout_processor import ElasticsearchConfig
from .elasticsearch_direct_http_handler import ElasticsearchDirectHttpHandler


def setup_rollout_logging_for_elasticsearch_handler(
    handler: ElasticsearchDirectHttpHandler, rollout_id: str, elastic_search_config: ElasticsearchConfig
) -> None:
    """
    Whenever a new subprocess is created, we need to setup the rollout context
    for the subprocess. This is useful when implementing your own remote server
    for rollout processing.

    1. Set the EP_ROLLOUT_ID environment variable
    2. Configure the Elasticsearch handler with the Elasticsearch config
    """

    # this should only affect this subprocess so logs from this subprocess can
    # be correlated to the rollout
    os.environ["EP_ROLLOUT_ID"] = rollout_id

    handler.configure(elasticsearch_config=elastic_search_config)
