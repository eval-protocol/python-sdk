import json
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Any, Dict
from datetime import datetime
from urllib.parse import urlparse
import requests

from eval_protocol.types.remote_rollout_processor import ElasticSearchConfig


class ElasticsearchDirectHttpHandler(logging.Handler):
    def __init__(self, elasticsearch_config: ElasticSearchConfig) -> None:
        super().__init__()
        self.base_url: str = elasticsearch_config.url.rstrip("/")
        self.index_name: str = elasticsearch_config.index_name
        self.api_key: str = elasticsearch_config.api_key
        self.url: str = f"{self.base_url}/{self.index_name}/_doc"
        self.formatter: logging.Formatter = logging.Formatter()
        self._executor = None

        # Parse URL to determine if we should verify SSL
        parsed_url = urlparse(elasticsearch_config.url)
        self.verify_ssl = parsed_url.scheme == "https"

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by scheduling it for async transmission."""
        try:
            # Create proper ISO 8601 timestamp
            timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            data: Dict[str, Any] = {
                "@timestamp": timestamp,
                "level": record.levelname,
                "message": record.getMessage(),
                "logger_name": record.name,
                # Add other relevant record attributes if needed
            }

            # Schedule the HTTP request to run asynchronously
            self._schedule_async_send(data, record)
        except Exception as e:
            self.handleError(record)
            print(f"Error preparing log for Elasticsearch: {e}")

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
            response: requests.Response = requests.post(
                self.url,
                headers={"Content-Type": "application/json", "Authorization": f"ApiKey {self.api_key}"},
                data=json.dumps(data),
                verify=self.verify_ssl,  # If using HTTPS, verify SSL certificate
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
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
