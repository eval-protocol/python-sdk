import os
import logging
import time
import requests
import pytest
from urllib.parse import urlparse

from eval_protocol.logging.elasticsearch_direct_http_handler import ElasticsearchDirectHttpHandler
from eval_protocol.pytest.elasticsearch_setup import ElasticsearchSetup
from eval_protocol.types.remote_rollout_processor import ElasticSearchConfig


@pytest.fixture
def elasticsearch_config():
    """Set up Elasticsearch and return configuration."""
    import time

    index_name = f"test-logs-{int(time.time())}"
    setup = ElasticsearchSetup()
    config = setup.setup_elasticsearch(index_name)
    return config


@pytest.fixture
def elasticsearch_handler(elasticsearch_config: ElasticSearchConfig):
    """Create and configure ElasticsearchDirectHttpHandler."""
    # Use a unique test-specific index name with timestamp

    handler = ElasticsearchDirectHttpHandler(elasticsearch_config)

    # Set a specific log level
    handler.setLevel(logging.INFO)

    return handler


@pytest.fixture
def test_logger(elasticsearch_handler, elasticsearch_config):
    """Set up a test logger with the Elasticsearch handler."""
    # Create the index for this specific handler
    setup = ElasticsearchSetup()
    setup.create_logging_index(elasticsearch_handler.index_name)

    logger = logging.getLogger("test_elasticsearch_logger")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Add our Elasticsearch handler
    logger.addHandler(elasticsearch_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
def test_elasticsearch_direct_http_handler_sends_logs(
    elasticsearch_config: ElasticSearchConfig, test_logger: logging.Logger
):
    """Test that ElasticsearchDirectHttpHandler successfully sends logs to Elasticsearch."""

    # Generate a unique test message to avoid conflicts
    test_message = f"Test log message at {time.time()}"

    # Send the log message
    test_logger.info(test_message)

    # Give Elasticsearch a moment to process the document
    time.sleep(3)

    # Query Elasticsearch to verify the document was received
    # Parse the URL to construct the search endpoint
    parsed_url = urlparse(elasticsearch_config.url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    search_url = f"{base_url}/{elasticsearch_config.index_name}/_search"

    # Prepare the search query with sorting by @timestamp
    search_query = {
        "query": {"match": {"message": test_message}},
        "sort": [{"@timestamp": {"order": "desc"}}],
        "size": 1,
    }

    # Execute the search
    response = requests.post(
        search_url,
        headers={"Content-Type": "application/json", "Authorization": f"ApiKey {elasticsearch_config.api_key}"},
        json=search_query,
        verify=parsed_url.scheme == "https",
    )

    # Check for errors and provide better debugging
    if response.status_code != 200:
        print(f"Elasticsearch search failed with status {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()

    search_results = response.json()

    # Assert that we found our log message
    assert "hits" in search_results, "Search response should contain 'hits'"
    assert "total" in search_results["hits"], "Search hits should contain 'total'"

    total_hits = search_results["hits"]["total"]
    if isinstance(total_hits, dict):
        # Elasticsearch 7+ format
        total_count = total_hits["value"]
    else:
        # Elasticsearch 6 format
        total_count = total_hits

    assert total_count > 0, f"Expected to find at least 1 log message, but found {total_count}"

    # Verify the content of the found document
    hits = search_results["hits"]["hits"]
    assert len(hits) > 0, "Expected at least one hit"

    found_document = hits[0]["_source"]
    assert found_document["message"] == test_message, (
        f"Expected message '{test_message}', got '{found_document['message']}'"
    )
    assert found_document["level"] == "INFO", f"Expected level 'INFO', got '{found_document['level']}'"
    assert found_document["logger_name"] == "test_elasticsearch_logger", (
        f"Expected logger name 'test_elasticsearch_logger', got '{found_document['logger_name']}'"
    )
    assert "@timestamp" in found_document, "Expected document to contain '@timestamp' field"

    print(f"Successfully verified log message in Elasticsearch: {test_message}")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
def test_elasticsearch_direct_http_handler_sorts_logs_chronologically(
    elasticsearch_config: ElasticSearchConfig, test_logger: logging.Logger
):
    """Test that logs can be sorted chronologically by timestamp."""

    # Send multiple log messages with small delays to ensure different timestamps
    test_messages = []
    for i in range(3):
        message = f"Chronological test message {i} at {time.time()}"
        test_messages.append(message)
        test_logger.info(message)
        time.sleep(0.1)  # Small delay to ensure different timestamps

    # Give Elasticsearch time to process all documents
    time.sleep(2)

    # Query Elasticsearch to get all our test messages sorted by timestamp
    parsed_url = urlparse(elasticsearch_config.url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    search_url = f"{base_url}/{elasticsearch_config.index_name}/_search"

    # Search for all messages containing our test prefix
    search_query = {
        "query": {"match_phrase_prefix": {"message": "Chronological test message"}},
        "sort": [{"@timestamp": {"order": "asc"}}],  # Ascending order (oldest first)
        "size": 10,
    }

    response = requests.post(
        search_url,
        headers={"Content-Type": "application/json", "Authorization": f"ApiKey {elasticsearch_config.api_key}"},
        json=search_query,
        verify=parsed_url.scheme == "https",
    )

    if response.status_code != 200:
        print(f"Elasticsearch search failed with status {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()

    search_results = response.json()

    # Verify we found our messages
    hits = search_results["hits"]["hits"]
    assert len(hits) >= 3, f"Expected at least 3 messages, found {len(hits)}"

    # Extract messages and verify they are in chronological order
    found_messages = [hit["_source"]["message"] for hit in hits]
    found_timestamps = [hit["_source"]["@timestamp"] for hit in hits]

    # Verify all our test messages are present
    for test_message in test_messages:
        assert test_message in found_messages, f"Expected message '{test_message}' not found in results"

    # Verify timestamps are in ascending order (chronological)
    for i in range(1, len(found_timestamps)):
        assert found_timestamps[i - 1] <= found_timestamps[i], (
            f"Timestamps not in chronological order: {found_timestamps[i - 1]} > {found_timestamps[i]}"
        )

    print(f"Successfully verified chronological sorting of {len(hits)} log messages")
    print(f"Timestamps in order: {found_timestamps}")
