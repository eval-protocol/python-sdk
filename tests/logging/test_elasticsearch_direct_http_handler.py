import os
import logging
import time
import pytest

from eval_protocol.log_utils.elasticsearch_direct_http_handler import ElasticsearchDirectHttpHandler
from eval_protocol.log_utils.elasticsearch_client import ElasticsearchClient
from eval_protocol.pytest.elasticsearch_setup import ElasticsearchSetup
from eval_protocol.types.remote_rollout_processor import ElasticsearchConfig


@pytest.fixture
def rollout_id():
    """Set up EP_ROLLOUT_ID environment variable for tests."""
    import uuid

    # Generate a unique rollout ID for this test session
    test_rollout_id = f"test-rollout-{uuid.uuid4().hex[:8]}"

    # Set the environment variable
    os.environ["EP_ROLLOUT_ID"] = test_rollout_id

    yield test_rollout_id

    # Clean up after the test
    if "EP_ROLLOUT_ID" in os.environ:
        del os.environ["EP_ROLLOUT_ID"]


@pytest.fixture
def elasticsearch_config():
    """Set up Elasticsearch and return configuration."""
    import time

    index_name = f"test-logs-{int(time.time())}"
    setup = ElasticsearchSetup()
    config = setup.setup_elasticsearch(index_name)
    return config


@pytest.fixture
def elasticsearch_handler(elasticsearch_config: ElasticsearchConfig, rollout_id: str):
    """Create and configure ElasticsearchDirectHttpHandler."""
    # Use a unique test-specific index name with timestamp

    handler = ElasticsearchDirectHttpHandler(elasticsearch_config)

    # Set a specific log level
    handler.setLevel(logging.INFO)

    return handler


@pytest.fixture
def elasticsearch_client(elasticsearch_config: ElasticsearchConfig):
    """Create an Elasticsearch client for testing."""
    # Create a new config instance for the client
    return ElasticsearchClient(elasticsearch_config)


@pytest.fixture
def test_logger(elasticsearch_handler, elasticsearch_config, rollout_id: str):
    """Set up a test logger with the Elasticsearch handler."""
    # Create the index for this specific handler
    setup = ElasticsearchSetup()
    setup.create_logging_index(elasticsearch_handler.config.index_name)

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
    elasticsearch_client: ElasticsearchClient, test_logger: logging.Logger, rollout_id: str
):
    """Test that ElasticsearchDirectHttpHandler successfully sends logs to Elasticsearch."""

    # Generate a unique test message to avoid conflicts
    test_message = f"Test log message at {time.time()}"

    # Send the log message
    test_logger.info(test_message)

    # Give Elasticsearch a moment to process the document
    time.sleep(3)

    # Search for the document using the client
    search_results = elasticsearch_client.search_by_match("message", test_message, size=1)

    # Assert that we found our log message
    assert search_results is not None, "Search should return results"
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
    elasticsearch_client: ElasticsearchClient, test_logger: logging.Logger, rollout_id: str
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

    # Search for all messages containing our test prefix
    search_results = elasticsearch_client.search_by_match_phrase_prefix(
        "message", "Chronological test message", size=10
    )

    # Add sorting to the search
    if search_results is None:
        search_results = elasticsearch_client.search(
            {"match_phrase_prefix": {"message": "Chronological test message"}},
            size=10,
            sort=[{"@timestamp": {"order": "asc"}}],
        )

    assert search_results is not None, "Search should return results"

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


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
def test_elasticsearch_direct_http_handler_includes_rollout_id(
    elasticsearch_client: ElasticsearchClient, test_logger: logging.Logger, rollout_id: str
):
    """Test that ElasticsearchDirectHttpHandler includes rollout_id field in indexed logs."""

    # Generate a unique test message to avoid conflicts
    test_message = f"Rollout ID test message at {time.time()}"

    # Send the log message
    test_logger.info(test_message)

    # Give Elasticsearch a moment to process the document
    time.sleep(3)

    # Search for the document using the client
    search_results = elasticsearch_client.search_by_match("message", test_message, size=1)

    # Assert that we found our log message
    assert search_results is not None, "Search should return results"
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

    # Verify the rollout_id field is present and correct
    assert "rollout_id" in found_document, "Expected document to contain 'rollout_id' field"
    assert found_document["rollout_id"] == rollout_id, (
        f"Expected rollout_id '{rollout_id}', got '{found_document['rollout_id']}'"
    )

    # Verify other expected fields are still present
    assert found_document["message"] == test_message, (
        f"Expected message '{test_message}', got '{found_document['message']}'"
    )
    assert found_document["level"] == "INFO", f"Expected level 'INFO', got '{found_document['level']}'"
    assert found_document["logger_name"] == "test_elasticsearch_logger", (
        f"Expected logger name 'test_elasticsearch_logger', got '{found_document['logger_name']}'"
    )
    assert "@timestamp" in found_document, "Expected document to contain '@timestamp' field"

    print(f"Successfully verified log message with rollout_id '{rollout_id}' in Elasticsearch: {test_message}")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
def test_elasticsearch_direct_http_handler_search_by_rollout_id(
    elasticsearch_client: ElasticsearchClient, test_logger: logging.Logger, rollout_id: str
):
    """Test that logs can be searched by rollout_id field in Elasticsearch."""

    # Generate unique test messages to avoid conflicts
    test_messages = []
    for i in range(3):
        message = f"Rollout search test message {i} at {time.time()}"
        test_messages.append(message)
        test_logger.info(message)
        time.sleep(0.1)  # Small delay to ensure different timestamps

    # Give Elasticsearch time to process all documents
    time.sleep(3)

    # Search for logs with our specific rollout_id using term query
    search_results = elasticsearch_client.search_by_term("rollout_id", rollout_id, size=10)

    # Assert that we found our log messages
    assert search_results is not None, "Search should return results"
    assert "hits" in search_results, "Search response should contain 'hits'"
    assert "total" in search_results["hits"], "Search hits should contain 'total'"

    total_hits = search_results["hits"]["total"]
    if isinstance(total_hits, dict):
        # Elasticsearch 7+ format
        total_count = total_hits["value"]
    else:
        # Elasticsearch 6 format
        total_count = total_hits

    assert total_count >= 3, f"Expected to find at least 3 log messages, but found {total_count}"

    # Verify the content of the found documents
    hits = search_results["hits"]["hits"]
    assert len(hits) >= 3, f"Expected at least 3 hits, found {len(hits)}"

    # Verify all found documents have the correct rollout_id
    found_messages = []
    for hit in hits:
        document = hit["_source"]
        assert document["rollout_id"] == rollout_id, (
            f"Expected rollout_id '{rollout_id}', got '{document['rollout_id']}'"
        )
        found_messages.append(document["message"])

    # Verify all our test messages are present in the search results
    for test_message in test_messages:
        assert test_message in found_messages, f"Expected message '{test_message}' not found in search results"

    # Test searching for a different rollout_id (should return no results)
    different_rollout_id = f"different-rollout-{time.time()}"
    different_results = elasticsearch_client.search_by_term("rollout_id", different_rollout_id, size=10)

    assert different_results is not None, "Search should return results"
    different_total_hits = different_results["hits"]["total"]
    if isinstance(different_total_hits, dict):
        different_count = different_total_hits["value"]
    else:
        different_count = different_total_hits

    assert different_count == 0, f"Expected 0 results for different rollout_id, but found {different_count}"

    print(f"Successfully verified search by rollout_id '{rollout_id}' found {len(hits)} log messages")
    print("Verified that search for different rollout_id returns 0 results")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
def test_elasticsearch_direct_http_handler_logs_status_info(
    elasticsearch_client: ElasticsearchClient, test_logger: logging.Logger, rollout_id: str
):
    """Test that ElasticsearchDirectHttpHandler logs Status class instances and can search by status code."""
    from eval_protocol import Status

    # Create a Status instance
    test_status = Status.rollout_running()

    # Generate a unique test message
    test_message = f"Status logging test message at {time.time()}"

    # Log with Status instance in extra data
    test_logger.info(test_message, extra={"status": test_status})

    # Give Elasticsearch time to process the document
    time.sleep(3)

    # Search for logs with our specific status code
    search_results = elasticsearch_client.search_by_term("status_code", test_status.code.value, size=1)

    # Assert that we found our log message
    assert search_results is not None, "Search should return results"
    assert "hits" in search_results, "Search response should contain 'hits'"
    assert "total" in search_results["hits"], "Search hits should contain 'total'"

    total_hits = search_results["hits"]["total"]
    if isinstance(total_hits, dict):
        total_count = total_hits["value"]
    else:
        total_count = total_hits

    assert total_count > 0, f"Expected to find at least 1 log message, but found {total_count}"

    # Verify the content of the found document
    hits = search_results["hits"]["hits"]
    assert len(hits) > 0, "Expected at least one hit"

    found_document = hits[0]["_source"]

    # Verify the status fields are present and correct
    assert "status_code" in found_document, "Expected document to contain 'status_code' field"
    assert found_document["status_code"] == test_status.code.value, (
        f"Expected status_code {test_status.code.value}, got {found_document['status_code']}"
    )
    assert "status_message" in found_document, "Expected document to contain 'status_message' field"
    assert found_document["status_message"] == test_status.message, (
        f"Expected status_message '{test_status.message}', got '{found_document['status_message']}'"
    )
    assert "status_details" in found_document, "Expected document to contain 'status_details' field"
    assert found_document["status_details"] == test_status.details, (
        f"Expected status_details {test_status.details}, got {found_document['status_details']}"
    )

    # Verify other expected fields are still present
    assert found_document["message"] == test_message, (
        f"Expected message '{test_message}', got '{found_document['message']}'"
    )
    assert found_document["rollout_id"] == rollout_id, (
        f"Expected rollout_id '{rollout_id}', got '{found_document['rollout_id']}'"
    )

    print(f"Successfully verified Status logging with code {test_status.code.value} in Elasticsearch: {test_message}")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
def test_elasticsearch_direct_http_handler_search_by_status_code(
    elasticsearch_client: ElasticsearchClient, test_logger: logging.Logger, rollout_id: str
):
    """Test that logs can be searched by status code in Elasticsearch."""
    from eval_protocol.models import Status

    # Create different Status instances for testing
    statuses = [
        Status.rollout_running(),
        Status.eval_finished(),
        Status.error("Test error message"),
    ]

    # Generate unique test messages
    test_messages = []
    for i, status in enumerate(statuses):
        message = f"Status search test message {i} at {time.time()}"
        test_messages.append((message, status))
        test_logger.info(message, extra={"status": status})
        time.sleep(0.1)  # Small delay to ensure different timestamps

    # Give Elasticsearch time to process all documents
    time.sleep(3)

    # Search for logs with RUNNING status code
    running_status = Status.Code.RUNNING
    search_results = elasticsearch_client.search_by_term("status_code", running_status.value, size=10)

    # Assert that we found our log messages
    assert search_results is not None, "Search should return results"
    assert "hits" in search_results, "Search response should contain 'hits'"
    assert "total" in search_results["hits"], "Search hits should contain 'total'"

    total_hits = search_results["hits"]["total"]
    if isinstance(total_hits, dict):
        total_count = total_hits["value"]
    else:
        total_count = total_hits

    assert total_count >= 1, f"Expected to find at least 1 log message with RUNNING status, but found {total_count}"

    # Verify the content of the found documents
    hits = search_results["hits"]["hits"]
    assert len(hits) >= 1, f"Expected at least 1 hit, found {len(hits)}"

    # Verify all found documents have the correct status code
    for hit in hits:
        document = hit["_source"]
        assert document["status_code"] == running_status.value, (
            f"Expected status_code {running_status.value}, got {document['status_code']}"
        )
        assert document["rollout_id"] == rollout_id, (
            f"Expected rollout_id '{rollout_id}', got '{document['rollout_id']}'"
        )

    print(f"Successfully verified search by status code {running_status.value} found {len(hits)} log messages")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
def test_elasticsearch_direct_http_handler_rollout_id_from_extra_overrides_env(
    elasticsearch_client: ElasticsearchClient, test_logger: logging.Logger, rollout_id: str
):
    """Test that rollout_id in extra parameter overrides environment variable."""

    # Create a different rollout_id to pass in extra
    extra_rollout_id = f"extra-rollout-{time.time()}"

    # Generate a unique test message
    test_message = f"Rollout ID override test message at {time.time()}"

    # Log with rollout_id in extra data (should override environment variable)
    test_logger.info(test_message, extra={"rollout_id": extra_rollout_id})

    # Give Elasticsearch time to process the document
    time.sleep(3)

    # Search for logs with the extra rollout_id (not the environment one)
    search_results = elasticsearch_client.search_by_term("rollout_id", extra_rollout_id, size=1)

    # Assert that we found our log message with the extra rollout_id
    assert search_results is not None, "Search should return results"
    assert "hits" in search_results, "Search response should contain 'hits'"
    assert "total" in search_results["hits"], "Search hits should contain 'total'"

    total_hits = search_results["hits"]["total"]
    if isinstance(total_hits, dict):
        total_count = total_hits["value"]
    else:
        total_count = total_hits

    assert total_count > 0, f"Expected to find at least 1 log message with extra rollout_id, but found {total_count}"

    # Verify the content of the found document
    hits = search_results["hits"]["hits"]
    assert len(hits) > 0, "Expected at least one hit"

    found_document = hits[0]["_source"]

    # Verify the rollout_id field matches the extra parameter (not environment variable)
    assert "rollout_id" in found_document, "Expected document to contain 'rollout_id' field"
    assert found_document["rollout_id"] == extra_rollout_id, (
        f"Expected rollout_id '{extra_rollout_id}', got '{found_document['rollout_id']}'"
    )

    # Verify it's NOT the environment variable rollout_id
    assert found_document["rollout_id"] != rollout_id, (
        f"Expected rollout_id to be overridden, but got environment rollout_id '{rollout_id}'"
    )

    # Verify other expected fields are still present
    assert found_document["message"] == test_message, (
        f"Expected message '{test_message}', got '{found_document['message']}'"
    )
    assert found_document["level"] == "INFO", f"Expected level 'INFO', got '{found_document['level']}'"
    assert found_document["logger_name"] == "test_elasticsearch_logger", (
        f"Expected logger name 'test_elasticsearch_logger', got '{found_document['logger_name']}'"
    )
    assert "@timestamp" in found_document, "Expected document to contain '@timestamp' field"

    # Verify that searching for the original environment rollout_id doesn't find this message
    env_search_results = elasticsearch_client.search(
        {"bool": {"must": [{"term": {"rollout_id": rollout_id}}, {"match": {"message": test_message}}]}}, size=1
    )

    assert env_search_results is not None, "Environment rollout_id search should return results"
    env_total_hits = env_search_results["hits"]["total"]
    if isinstance(env_total_hits, dict):
        env_count = env_total_hits["value"]
    else:
        env_count = env_total_hits

    assert env_count == 0, (
        f"Expected 0 results when searching for message with environment rollout_id, but found {env_count}"
    )

    print(f"Successfully verified rollout_id override: extra '{extra_rollout_id}' overrode environment '{rollout_id}'")
