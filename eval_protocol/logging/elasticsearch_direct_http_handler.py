import json
import logging
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Any, Dict
from datetime import datetime

from eval_protocol.types.remote_rollout_processor import ElasticsearchConfig
from .elasticsearch_client import ElasticsearchClient


class ElasticsearchDirectHttpHandler(logging.Handler):
    def __init__(self, elasticsearch_config: ElasticsearchConfig) -> None:
        super().__init__()
        self.config = ElasticsearchConfig(
            url=elasticsearch_config.url,
            api_key=elasticsearch_config.api_key,
            index_name=elasticsearch_config.index_name,
        )
        self.client = ElasticsearchClient(self.config)
        self.formatter: logging.Formatter = logging.Formatter()
        self._executor = None

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by scheduling it for async transmission."""
        try:
            # Create proper ISO 8601 timestamp
            timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            rollout_id = self._get_rollout_id(record)
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

            # Schedule the HTTP request to run asynchronously
            self._schedule_async_send(data, record)
        except Exception as e:
            self.handleError(record)
            print(f"Error preparing log for Elasticsearch: {e}")

    def _get_rollout_id(self, record: logging.LogRecord) -> str:
        """Get the rollout ID from environment variables."""
        rollout_id = os.getenv("EP_ROLLOUT_ID")
        if rollout_id is None:
            raise ValueError(
                "EP_ROLLOUT_ID environment variable is not set but needed for ElasticsearchDirectHttpHandler"
            )
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
