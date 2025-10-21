import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict
from datetime import datetime, timezone

from eval_protocol.types.remote_rollout_processor import ElasticsearchConfig
from .elasticsearch_client import ElasticsearchClient

import logging

logger = logging.getLogger(__name__)

# do not inherit root logger since we are a handler ourselves
logger.propagate = False

logger.addHandler(logging.StreamHandler())

if os.environ.get("EP_DEBUG") == "true":
    logger.setLevel(logging.DEBUG)
    logger.debug("EP_DEBUG=true detected, set log level to DEBUG")


class ElasticsearchDirectHttpHandler(logging.Handler):
    def __init__(self, elasticsearch_config: ElasticsearchConfig | None = None) -> None:
        super().__init__()
        self.config = elasticsearch_config
        self.client = ElasticsearchClient(self.config) if self.config else None
        self.formatter: logging.Formatter = logging.Formatter()
        self._executor = None

    def configure(self, elasticsearch_config: ElasticsearchConfig) -> None:
        self.config = elasticsearch_config
        self.client = ElasticsearchClient(self.config)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by scheduling it for async transmission."""
        try:
            # Create proper ISO 8601 timestamp in UTC
            timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            rollout_id = self._get_rollout_id(record)
            logger.debug(f"Emitting log record: {record.getMessage()} with rollout_id: {rollout_id}")
            if not rollout_id:
                logger.debug(
                    "No rollout_id provided in extra data for ElasticsearchDirectHttpHandler through EP_ROLLOUT_ID environment variable or rollout_id extra data. Skipping log record."
                )
                return
            status_info = self._get_status_info(record)

            data: Dict[str, Any] = {
                "@timestamp": timestamp,
                "level": record.levelname,
                "message": record.getMessage(),
                "logger_name": record.name,
                "rollout_id": rollout_id,
            }

            # Add status information if present
            if status_info:
                data.update(status_info)

            # Optional correlation enrichment
            experiment_id = getattr(record, "experiment_id", None)
            if experiment_id is not None:
                data["experiment_id"] = experiment_id
            run_id = getattr(record, "run_id", None)
            if run_id is not None:
                data["run_id"] = run_id
            rollout_ids = getattr(record, "rollout_ids", None)
            if rollout_ids is not None:
                data["rollout_ids"] = rollout_ids

            # Schedule the HTTP request to run asynchronously
            self._schedule_async_send(data, record)
        except Exception as e:
            self.handleError(record)
            print(f"Error preparing log for Elasticsearch: {e}")

    def _get_rollout_id(self, record: logging.LogRecord) -> str | None:
        """Get the rollout ID from record extra data or environment variables."""
        # Check if rollout_id is provided in the extra data first
        if hasattr(record, "rollout_id") and record.rollout_id is not None:  # type: ignore
            return str(record.rollout_id)  # type: ignore

        # Fall back to environment variable
        rollout_id = os.getenv("EP_ROLLOUT_ID")
        return rollout_id

    def _get_status_info(self, record: logging.LogRecord) -> Optional[Dict[str, Any]]:
        """Extract status information from the log record's extra data."""
        # Check if 'status' is in the extra data (passed via extra parameter)
        if hasattr(record, "status") and record.status is not None:  # type: ignore
            status = record.status  # type: ignore

            # Handle Status class instances (Pydantic BaseModel)
            if hasattr(status, "code") and hasattr(status, "message"):
                # Status object - extract code and message
                status_code = status.code
                # Handle both enum values and direct integer values
                if hasattr(status_code, "value"):
                    status_code = status_code.value

                return {
                    "status_code": status_code,
                    "status_message": status.message,
                    "status_details": getattr(status, "details", []),
                }
            elif isinstance(status, dict):
                # Dictionary representation of status
                return {
                    "status_code": status.get("code"),
                    "status_message": status.get("message"),
                    "status_details": status.get("details", []),
                }
        return None

    def _schedule_async_send(self, data: Dict[str, Any], record: logging.LogRecord) -> None:
        """Schedule an async task to send the log data to Elasticsearch."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="elasticsearch-logger")

        # Submit the HTTP request to the thread pool
        future = self._executor.submit(self._send_to_elasticsearch, data, record)

        # Add error handling callback
        future.add_done_callback(lambda f: self._handle_async_result(f, record))

    def _send_to_elasticsearch(self, data: Dict[str, Any], record: logging.LogRecord) -> None:
        """Send data to Elasticsearch (runs in thread pool)."""
        if not self.client:
            logger.warning("No Elasticsearch client configured, skipping log record")
            return
        try:
            success = self.client.index_document(data)
            if not success:
                raise Exception("Failed to index document to Elasticsearch")
        except Exception as e:
            # Re-raise to be handled by the callback
            raise e

    def _handle_async_result(self, future, record: logging.LogRecord) -> None:
        """Handle the result of the async send operation."""
        try:
            future.result()  # This will raise any exception that occurred
        except Exception as e:
            self.handleError(record)
            # You might want to log this error to a file or console
            # to prevent a logging loop.
            if hasattr(e, "response") and getattr(e, "response", None) is not None:
                print(f"Error sending log to Elasticsearch: {e}")
                print(f"Response content: {getattr(e, 'response').text}")
            else:
                print(f"Error sending log to Elasticsearch: {e}")

    def close(self) -> None:
        """Clean up resources when the handler is closed."""
        super().close()
        if self._executor:
            self._executor.shutdown(wait=True)
