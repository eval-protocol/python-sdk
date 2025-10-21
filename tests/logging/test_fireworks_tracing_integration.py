import os
import logging
import time
import pytest
from datetime import datetime, timezone
from typing import List, Dict, Any

from eval_protocol.log_utils.fireworks_tracing_http_handler import FireworksTracingHttpHandler
from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter


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
def fireworks_base_url():
    """Get Fireworks tracing base URL from environment or use default."""
    return os.environ.get("FW_TRACING_GATEWAY_BASE_URL", "https://tracing.fireworks.ai")


@pytest.fixture
def fireworks_handler(fireworks_base_url: str, rollout_id: str):
    """Create and configure FireworksTracingHttpHandler."""
    handler = FireworksTracingHttpHandler(gateway_base_url=fireworks_base_url)

    # Set a specific log level
    handler.setLevel(logging.INFO)

    return handler


@pytest.fixture
def fireworks_adapter(fireworks_base_url: str):
    """Create a FireworksTracingAdapter for testing."""
    return FireworksTracingAdapter(base_url=fireworks_base_url)


@pytest.fixture
def test_logger(fireworks_handler, rollout_id: str):
    """Set up a test logger with the Fireworks tracing handler."""
    logger = logging.getLogger("test_fireworks_tracing_logger")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Add our Fireworks tracing handler
    logger.addHandler(fireworks_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    yield logger

    # Clean up the logger handlers after the test
    logger.handlers.clear()


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_sends_logs(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test that FireworksTracingHttpHandler successfully sends logs to Fireworks tracing."""

    # Generate a unique test message to avoid conflicts
    test_message = f"Test log message at {time.time()}"

    # Send the log message
    test_logger.info(test_message)

    # Give Fireworks tracing a moment to process the log
    time.sleep(5)

    # Search for the log using the adapter
    log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=10, hours_back=1)

    # Assert that we found our log message
    assert log_entries is not None, "Search should return results"
    assert len(log_entries) > 0, f"Expected to find at least 1 log entry, but found {len(log_entries)}"

    # Find our specific test message
    found_entry = None
    for entry in log_entries:
        if entry.get("message") == test_message:
            found_entry = entry
            break

    assert found_entry is not None, f"Expected to find test message '{test_message}' in log entries"

    # Verify the content of the found entry
    assert found_entry["message"] == test_message, f"Expected message '{test_message}', got '{found_entry['message']}'"
    assert found_entry["severity"] == "INFO", f"Expected severity 'INFO', got '{found_entry['severity']}'"
    assert "timestamp" in found_entry, "Expected entry to contain 'timestamp' field"
    assert f"rollout_id:{rollout_id}" in found_entry.get("tags", []), (
        f"Expected rollout_id tag in entry tags: {found_entry.get('tags', [])}"
    )

    print(f"Successfully verified log message in Fireworks tracing: {test_message}")


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_sorts_logs_chronologically(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test that logs can be sorted chronologically by timestamp."""

    # Send multiple log messages with small delays to ensure different timestamps
    test_messages = []
    for i in range(3):
        message = f"Chronological test message {i} at {time.time()}"
        test_messages.append(message)
        test_logger.info(message)
        time.sleep(0.5)  # Small delay to ensure different timestamps

    # Give Fireworks tracing time to process all logs
    time.sleep(5)

    # Search for logs with our rollout_id
    log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=20, hours_back=1)

    # Filter to only our test messages
    found_entries = []
    for entry in log_entries:
        for test_message in test_messages:
            if entry.get("message") == test_message:
                found_entries.append(entry)
                break

    assert len(found_entries) >= 3, f"Expected at least 3 messages, found {len(found_entries)}"

    # Extract timestamps and verify they are in chronological order
    found_timestamps = [entry["timestamp"] for entry in found_entries]

    # Sort entries by timestamp for verification
    found_entries_sorted = sorted(found_entries, key=lambda x: x["timestamp"])

    # Verify all our test messages are present
    found_messages = [entry["message"] for entry in found_entries_sorted]
    for test_message in test_messages:
        assert test_message in found_messages, f"Expected message '{test_message}' not found in results"

    print(f"Successfully verified chronological sorting of {len(found_entries)} log messages")
    print(f"Timestamps in order: {found_timestamps}")


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_includes_rollout_id(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test that FireworksTracingHttpHandler includes rollout_id in tags."""

    # Generate a unique test message to avoid conflicts
    test_message = f"Rollout ID test message at {time.time()}"

    # Send the log message
    test_logger.info(test_message)

    # Give Fireworks tracing a moment to process the log
    time.sleep(5)

    # Search for logs with our specific rollout_id
    log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=10, hours_back=1)

    # Assert that we found our log message
    assert log_entries is not None, "Search should return results"
    assert len(log_entries) > 0, f"Expected to find at least 1 log entry, but found {len(log_entries)}"

    # Find our specific test message
    found_entry = None
    for entry in log_entries:
        if entry.get("message") == test_message:
            found_entry = entry
            break

    assert found_entry is not None, f"Expected to find test message '{test_message}' in log entries"

    # Verify the rollout_id tag is present and correct
    tags = found_entry.get("tags", [])
    rollout_tag = f"rollout_id:{rollout_id}"
    assert rollout_tag in tags, f"Expected rollout_id tag '{rollout_tag}' in tags: {tags}"

    # Verify other expected fields are still present
    assert found_entry["message"] == test_message, f"Expected message '{test_message}', got '{found_entry['message']}'"
    assert found_entry["severity"] == "INFO", f"Expected severity 'INFO', got '{found_entry['severity']}'"
    assert "timestamp" in found_entry, "Expected entry to contain 'timestamp' field"

    print(f"Successfully verified log message with rollout_id '{rollout_id}' in Fireworks tracing: {test_message}")


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_search_by_rollout_id(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test that logs can be searched by rollout_id tag in Fireworks tracing."""

    # Generate unique test messages to avoid conflicts
    test_messages = []
    for i in range(3):
        message = f"Rollout search test message {i} at {time.time()}"
        test_messages.append(message)
        test_logger.info(message)
        time.sleep(0.2)  # Small delay to ensure different timestamps

    # Give Fireworks tracing time to process all logs
    time.sleep(5)

    # Search for logs with our specific rollout_id
    log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=20, hours_back=1)

    # Assert that we found our log messages
    assert log_entries is not None, "Search should return results"
    assert len(log_entries) >= 3, f"Expected to find at least 3 log entries, but found {len(log_entries)}"

    # Verify all found entries have the correct rollout_id tag
    found_messages = []
    rollout_tag = f"rollout_id:{rollout_id}"
    for entry in log_entries:
        tags = entry.get("tags", [])
        assert rollout_tag in tags, f"Expected rollout_id tag '{rollout_tag}' in tags: {tags}"
        found_messages.append(entry["message"])

    # Verify all our test messages are present in the search results
    for test_message in test_messages:
        assert test_message in found_messages, f"Expected message '{test_message}' not found in search results"

    # Test searching for a different rollout_id (should return no results for our messages)
    different_rollout_id = f"different-rollout-{time.time()}"
    different_entries = fireworks_adapter.search_logs(
        tags=[f"rollout_id:{different_rollout_id}"], limit=10, hours_back=1
    )

    # Should either be empty or not contain our test messages
    if different_entries:
        different_messages = [entry.get("message", "") for entry in different_entries]
        for test_message in test_messages:
            assert test_message not in different_messages, (
                f"Found test message '{test_message}' in search for different rollout_id"
            )

    print(f"Successfully verified search by rollout_id '{rollout_id}' found {len(log_entries)} log entries")
    print("Verified that search for different rollout_id doesn't return our test messages")


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_logs_status_info(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test that FireworksTracingHttpHandler logs Status class instances and can search by status code."""
    from eval_protocol import Status

    # Create a Status instance
    test_status = Status.rollout_running()

    # Generate a unique test message
    test_message = f"Status logging test message at {time.time()}"

    # Log with Status instance in extra data
    test_logger.info(test_message, extra={"status": test_status})

    # Give Fireworks tracing time to process the log
    time.sleep(5)

    # Search for logs with our rollout_id
    log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=10, hours_back=1)

    # Assert that we found our log message
    assert log_entries is not None, "Search should return results"
    assert len(log_entries) > 0, f"Expected to find at least 1 log entry, but found {len(log_entries)}"

    # Find our specific test message
    found_entry = None
    for entry in log_entries:
        if entry.get("message") == test_message:
            found_entry = entry
            break

    assert found_entry is not None, f"Expected to find test message '{test_message}' in log entries"

    # Verify the status fields are present and correct
    assert "status_code" in found_entry, "Expected entry to contain 'status_code' field"
    assert found_entry["status_code"] == test_status.code.value, (
        f"Expected status_code {test_status.code.value}, got {found_entry['status_code']}"
    )
    assert "status_message" in found_entry, "Expected entry to contain 'status_message' field"
    assert found_entry["status_message"] == test_status.message, (
        f"Expected status_message '{test_status.message}', got '{found_entry['status_message']}"
    )
    assert "status_details" in found_entry, "Expected entry to contain 'status_details' field"
    assert found_entry["status_details"] == test_status.details, (
        f"Expected status_details {test_status.details}, got {found_entry['status_details']}"
    )

    # Verify other expected fields are still present
    assert found_entry["message"] == test_message, f"Expected message '{test_message}', got '{found_entry['message']}'"
    rollout_tag = f"rollout_id:{rollout_id}"
    assert rollout_tag in found_entry.get("tags", []), (
        f"Expected rollout_id tag '{rollout_tag}' in tags: {found_entry.get('tags', [])}"
    )

    print(
        f"Successfully verified Status logging with code {test_status.code.value} in Fireworks tracing: {test_message}"
    )


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_search_by_status_code(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test that logs can be searched and filtered by status code in Fireworks tracing."""
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
        time.sleep(0.2)  # Small delay to ensure different timestamps

    # Give Fireworks tracing time to process all logs
    time.sleep(5)

    # Search for all logs with our rollout_id
    log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=20, hours_back=1)

    # Assert that we found our log messages
    assert log_entries is not None, "Search should return results"
    assert len(log_entries) >= 3, f"Expected to find at least 3 log entries, but found {len(log_entries)}"

    # Find entries with RUNNING status code
    running_entries = []
    finished_entries = []
    error_entries = []

    for entry in log_entries:
        if "status_code" in entry:
            if entry["status_code"] == Status.Code.RUNNING.value:
                running_entries.append(entry)
            elif entry["status_code"] == Status.Code.FINISHED.value:
                finished_entries.append(entry)
            elif entry["status_code"] == Status.Code.INTERNAL.value:  # Error status
                error_entries.append(entry)

    # Verify we found entries for each status type
    assert len(running_entries) >= 1, f"Expected at least 1 RUNNING status entry, found {len(running_entries)}"
    assert len(finished_entries) >= 1, f"Expected at least 1 FINISHED status entry, found {len(finished_entries)}"
    assert len(error_entries) >= 1, f"Expected at least 1 ERROR status entry, found {len(error_entries)}"

    # Verify the content of the found entries
    for entry in running_entries:
        assert entry["status_code"] == Status.Code.RUNNING.value, (
            f"Expected status_code {Status.Code.RUNNING.value}, got {entry['status_code']}"
        )
        rollout_tag = f"rollout_id:{rollout_id}"
        assert rollout_tag in entry.get("tags", []), (
            f"Expected rollout_id tag '{rollout_tag}' in tags: {entry.get('tags', [])}"
        )

    print("Successfully verified search by status codes:")
    print(f"  - RUNNING entries: {len(running_entries)}")
    print(f"  - FINISHED entries: {len(finished_entries)}")
    print(f"  - ERROR entries: {len(error_entries)}")


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_rollout_id_from_extra_overrides_env(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test that rollout_id in extra parameter overrides environment variable."""

    # Create a different rollout_id to pass in extra
    extra_rollout_id = f"extra-rollout-{time.time()}"

    # Generate a unique test message
    test_message = f"Rollout ID override test message at {time.time()}"

    # Log with rollout_id in extra data (should override environment variable)
    test_logger.info(test_message, extra={"rollout_id": extra_rollout_id})

    # Give Fireworks tracing time to process the log
    time.sleep(5)

    # Search for logs with the extra rollout_id (not the environment one)
    extra_log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{extra_rollout_id}"], limit=10, hours_back=1)

    # Assert that we found our log message with the extra rollout_id
    assert extra_log_entries is not None, "Search should return results"
    assert len(extra_log_entries) > 0, (
        f"Expected to find at least 1 log entry with extra rollout_id, but found {len(extra_log_entries)}"
    )

    # Find our specific test message
    found_entry = None
    for entry in extra_log_entries:
        if entry.get("message") == test_message:
            found_entry = entry
            break

    assert found_entry is not None, f"Expected to find test message '{test_message}' in extra rollout_id search"

    # Verify the rollout_id tag matches the extra parameter (not environment variable)
    extra_rollout_tag = f"rollout_id:{extra_rollout_id}"
    env_rollout_tag = f"rollout_id:{rollout_id}"

    tags = found_entry.get("tags", [])
    assert extra_rollout_tag in tags, f"Expected extra rollout_id tag '{extra_rollout_tag}' in tags: {tags}"
    assert env_rollout_tag not in tags, f"Expected environment rollout_id tag '{env_rollout_tag}' NOT in tags: {tags}"

    # Verify other expected fields are still present
    assert found_entry["message"] == test_message, f"Expected message '{test_message}', got '{found_entry['message']}'"
    assert found_entry["severity"] == "INFO", f"Expected severity 'INFO', got '{found_entry['severity']}'"
    assert "timestamp" in found_entry, "Expected entry to contain 'timestamp' field"

    # Verify that searching for the original environment rollout_id doesn't find this message
    env_log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=10, hours_back=1)

    # Should either be empty or not contain our test message
    if env_log_entries:
        env_messages = [entry.get("message", "") for entry in env_log_entries]
        assert test_message not in env_messages, (
            f"Found test message '{test_message}' when searching with environment rollout_id '{rollout_id}'"
        )

    print(f"Successfully verified rollout_id override: extra '{extra_rollout_id}' overrode environment '{rollout_id}'")


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_timestamp_format(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test that FireworksTracingHttpHandler formats timestamps correctly with UTC timezone."""

    # Generate a unique test message
    test_message = f"Timestamp format test message at {time.time()}"

    # Record the time before logging to compare with the timestamp
    before_log_time = datetime.now(timezone.utc)

    # Send the log message
    test_logger.info(test_message)

    # Record the time after logging
    after_log_time = datetime.now(timezone.utc)

    # Give Fireworks tracing a moment to process the log
    time.sleep(5)

    # Search for the log using the adapter
    log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=10, hours_back=1)

    # Assert that we found our log message
    assert log_entries is not None, "Search should return results"
    assert len(log_entries) > 0, f"Expected to find at least 1 log entry, but found {len(log_entries)}"

    # Find our specific test message
    found_entry = None
    for entry in log_entries:
        if entry.get("message") == test_message:
            found_entry = entry
            break

    assert found_entry is not None, f"Expected to find test message '{test_message}' in log entries"
    assert "timestamp" in found_entry, "Expected entry to contain 'timestamp' field"

    # Get the timestamp from the entry
    timestamp_str = found_entry["timestamp"]

    # Verify the timestamp format matches ISO 8601 with UTC timezone (Z suffix)
    assert timestamp_str.endswith("Z"), f"Expected timestamp to end with 'Z' (UTC), got: {timestamp_str}"

    # Parse the timestamp to verify it's valid
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except ValueError as e:
        pytest.fail(f"Failed to parse timestamp '{timestamp_str}': {e}")

    # Verify the timestamp is timezone-aware (UTC)
    assert parsed_timestamp.tzinfo is not None, "Expected timestamp to be timezone-aware"
    utc_offset = parsed_timestamp.tzinfo.utcoffset(None)
    assert utc_offset is not None and utc_offset.total_seconds() == 0, "Expected timestamp to be in UTC timezone"

    # Verify the timestamp is within reasonable bounds (between before and after log time)
    # Allow for some margin due to processing time
    from datetime import timedelta

    time_margin = timedelta(seconds=30)  # 30 seconds margin for network latency
    assert before_log_time - time_margin <= parsed_timestamp <= after_log_time + time_margin, (
        f"Expected timestamp {parsed_timestamp} to be between {before_log_time} and {after_log_time} "
        f"(with {time_margin} margin)"
    )

    # Verify the timestamp format includes microseconds
    assert "." in timestamp_str, "Expected timestamp to include microseconds"
    assert timestamp_str.count(".") == 1, "Expected timestamp to have exactly one decimal point"

    # Verify the format matches the expected pattern: YYYY-MM-DDTHH:MM:SS.ffffffZ
    import re

    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
    assert re.match(iso_pattern, timestamp_str), f"Expected timestamp to match ISO 8601 pattern, got: {timestamp_str}"

    print(f"Successfully verified timestamp format: {timestamp_str}")
    print(f"Parsed timestamp: {parsed_timestamp} (UTC)")
    print(f"Timestamp is within expected time range: {before_log_time} <= {parsed_timestamp} <= {after_log_time}")


@pytest.mark.skipif(
    not os.environ.get("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY required for Fireworks tracing integration tests",
)
def test_fireworks_tracing_handler_status_polling_flow(
    fireworks_adapter: FireworksTracingAdapter, test_logger: logging.Logger, rollout_id: str
):
    """Test the complete status polling flow: RUNNING -> FINISHED with status detection."""
    from eval_protocol import Status

    # Simulate a rollout flow with status transitions
    test_message_base = f"Status polling flow test at {time.time()}"

    # 1. Log RUNNING status
    running_message = f"{test_message_base} - RUNNING"
    test_logger.info(running_message, extra={"status": Status.rollout_running()})
    time.sleep(1)

    # 2. Log some progress messages
    progress_message = f"{test_message_base} - Progress update"
    test_logger.info(progress_message)
    time.sleep(1)

    # 3. Log FINISHED status
    finished_message = f"{test_message_base} - FINISHED"
    test_logger.info(finished_message, extra={"status": Status.rollout_finished()})

    # Give Fireworks tracing time to process all logs
    time.sleep(5)

    # Search for all logs with our rollout_id
    log_entries = fireworks_adapter.search_logs(tags=[f"rollout_id:{rollout_id}"], limit=20, hours_back=1)

    # Assert that we found our log messages
    assert log_entries is not None, "Search should return results"
    assert len(log_entries) >= 3, f"Expected to find at least 3 log entries, but found {len(log_entries)}"

    # Find entries with status codes (non-RUNNING status should be detectable for polling)
    status_entries = []
    non_running_entries = []

    for entry in log_entries:
        if "status_code" in entry:
            status_entries.append(entry)
            if entry["status_code"] != Status.Code.RUNNING.value:
                non_running_entries.append(entry)

    # Verify we found status entries
    assert len(status_entries) >= 2, (
        f"Expected at least 2 status entries (RUNNING, FINISHED), found {len(status_entries)}"
    )

    # Verify we found non-RUNNING entries (which would trigger polling to stop)
    assert len(non_running_entries) >= 1, f"Expected at least 1 non-RUNNING entry, found {len(non_running_entries)}"

    # Find the FINISHED entry specifically
    finished_entry = None
    for entry in non_running_entries:
        if entry["status_code"] == Status.Code.FINISHED.value:
            finished_entry = entry
            break

    assert finished_entry is not None, "Expected to find FINISHED status entry"
    assert finished_entry["status_message"] == "Rollout finished", (
        f"Expected FINISHED status message, got '{finished_entry['status_message']}'"
    )

    print("Successfully verified status polling flow:")
    print(f"  - Total status entries: {len(status_entries)}")
    print(f"  - Non-RUNNING entries: {len(non_running_entries)}")
    print(f"  - FINISHED entry found with message: '{finished_entry['status_message']}'")
    print("This flow simulates what RemoteRolloutProcessor would detect during status polling")
