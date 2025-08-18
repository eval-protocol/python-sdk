"""Google BigQuery adapter for Eval Protocol.

This adapter allows querying data from Google BigQuery tables and converting it
to EvaluationRow format for use in evaluation pipelines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Union

from eval_protocol.models import CompletionParams, EvaluationRow, InputMetadata, Message

logger = logging.getLogger(__name__)

try:
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import bigquery
    from google.cloud.exceptions import Forbidden, NotFound
    from google.oauth2 import service_account

    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    # Optional dependency: avoid noisy warnings during import
    logger.debug("Google Cloud BigQuery not installed. Optional feature disabled.")

# Avoid importing BigQuery types at runtime for annotations when not installed
if TYPE_CHECKING:
    from google.cloud import bigquery as _bigquery_type

    QueryParameterType = Union[
        _bigquery_type.ScalarQueryParameter,
        _bigquery_type.ArrayQueryParameter,
    ]
else:
    QueryParameterType = Any

# Type alias for transformation function
TransformFunction = Callable[[Dict[str, Any]], Dict[str, Any]]


class BigQueryAdapter:
    """Adapter to query data from Google BigQuery and convert to EvaluationRow format.

    This adapter connects to Google BigQuery, executes SQL queries, and applies
    a user-provided transformation function to convert each row to the format
    expected by EvaluationRow.

    The transformation function should take a BigQuery row dictionary and return:
    {
        'messages': List[Dict] - list of message dictionaries with 'role' and 'content'
        'ground_truth': Optional[str] - expected answer/output
        'metadata': Optional[Dict] - any additional metadata to preserve
        'tools': Optional[List[Dict]] - tool definitions for tool calling scenarios
    }
    """

    def __init__(
        self,
        transform_fn: TransformFunction,
        dataset_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: Optional[str] = None,
        **client_kwargs,
    ):
        """Initialize the BigQuery adapter.

        Args:
            transform_fn: Function to transform BigQuery rows to evaluation format
            dataset_id: Google Cloud project ID (if None, uses default from environment)
            credentials_path: Path to service account JSON file (if None, uses default auth)
            location: Default location for BigQuery jobs
            **client_kwargs: Additional arguments to pass to BigQuery client

        Raises:
            ImportError: If google-cloud-bigquery is not installed
            DefaultCredentialsError: If authentication fails
        """
        if not BIGQUERY_AVAILABLE:
            raise ImportError(
                "Google Cloud BigQuery not installed. Install with: pip install 'eval-protocol[bigquery]'"
            )

        self.transform_fn = transform_fn
        self.dataset_id = dataset_id
        self.location = location

        # Initialize BigQuery client
        try:
            client_args = {}
            if dataset_id:
                client_args["project"] = dataset_id
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                client_args["credentials"] = credentials
            if location:
                client_args["location"] = location

            client_args.update(client_kwargs)
            self.client = bigquery.Client(**client_args)

        except DefaultCredentialsError as e:
            logger.error("Failed to authenticate with BigQuery: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to initialize BigQuery client: %s", e)
            raise

    def get_evaluation_rows(
        self,
        query: str,
        query_params: Optional[List[QueryParameterType]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **completion_params_kwargs,
    ) -> Iterator[EvaluationRow]:
        """Execute BigQuery query and convert results to EvaluationRow format.

        Args:
            query: SQL query to execute
            query_params: Optional list of query parameters for parameterized queries
            limit: Maximum number of rows to return (applied after BigQuery query)
            offset: Number of rows to skip (applied after BigQuery query)
            model_name: Model name for completion parameters
            temperature: Temperature for completion parameters
            max_tokens: Max tokens for completion parameters
            **completion_params_kwargs: Additional completion parameters

        Yields:
            EvaluationRow: Converted evaluation rows

        Raises:
            NotFound: If the query references non-existent tables/datasets
            Forbidden: If insufficient permissions
        """
        try:
            # Configure query job
            job_config = bigquery.QueryJobConfig()
            if query_params:
                job_config.query_parameters = query_params
            if self.location:
                job_config.location = self.location

            query_job = self.client.query(query, job_config=job_config)

            results = query_job.result()

            completion_params: CompletionParams = {
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **completion_params_kwargs,
            }

            # Convert rows with offset/limit
            row_count = 0
            processed_count = 0

            for raw_row in results:
                # Apply offset
                if row_count < offset:
                    row_count += 1
                    continue

                # Apply limit
                if limit is not None and processed_count >= limit:
                    break

                try:
                    eval_row = self._convert_row_to_evaluation_row(raw_row, processed_count, completion_params)
                    if eval_row:
                        yield eval_row
                        processed_count += 1

                except (AttributeError, ValueError, KeyError) as e:
                    logger.warning("Failed to convert row %d: %s", row_count, e)

                row_count += 1

        except (NotFound, Forbidden) as e:
            logger.error("BigQuery access error: %s", e)
            raise
        except Exception as e:
            logger.error("Error executing BigQuery query: %s", e)
            raise

    def _convert_row_to_evaluation_row(
        self,
        raw_row: Dict[str, Any],
        row_index: int,
        completion_params: CompletionParams,
    ) -> EvaluationRow:
        """Convert a single BigQuery row to EvaluationRow format.

        Args:
            raw_row: BigQuery row dictionary
            row_index: Index of the row in the result set
            completion_params: Completion parameters to use

        Returns:
            EvaluationRow object or None if conversion fails
        """
        # Apply user transformation
        transformed = self.transform_fn(raw_row)

        # Validate required fields
        if "messages" not in transformed:
            raise ValueError("Transform function must return 'messages' field")

        # Convert message dictionaries to Message objects
        messages = []
        for msg_dict in transformed["messages"]:
            if not isinstance(msg_dict, dict):
                raise ValueError("Each message must be a dictionary")
            if "role" not in msg_dict:
                raise ValueError("Each message must have a 'role' field")

            messages.append(
                Message(
                    role=msg_dict["role"],
                    content=msg_dict.get("content"),
                    name=msg_dict.get("name"),
                    tool_call_id=msg_dict.get("tool_call_id"),
                    tool_calls=msg_dict.get("tool_calls"),
                    function_call=msg_dict.get("function_call"),
                )
            )

        # Extract other fields
        ground_truth = transformed.get("ground_truth")
        tools = transformed.get("tools")
        user_metadata = transformed.get("metadata", {})

        # Create dataset info
        dataset_info = {
            "source": "bigquery",
            "dataset_id": self.dataset_id or self.client.project,
            "row_index": row_index,
            "transform_function": (
                self.transform_fn.__name__ if hasattr(self.transform_fn, "__name__") else "anonymous"
            ),
        }

        # Add user metadata
        dataset_info.update(user_metadata)

        # Add original row data (with prefix to avoid conflicts)
        for key, value in raw_row.items():
            # Convert BigQuery types to JSON-serializable types
            dataset_info[f"original_{key}"] = value

        # Create input metadata (following HuggingFace pattern)
        input_metadata = InputMetadata(
            row_id=f"{self.dataset_id}_{row_index}",
            completion_params=completion_params,
            dataset_info=dataset_info,
            session_data={
                "dataset_source": "bigquery",
            },
        )

        return EvaluationRow(
            messages=messages,
            tools=tools,
            input_metadata=input_metadata,
            ground_truth=str(ground_truth) if ground_truth is not None else None,
        )


def create_bigquery_adapter(
    transform_fn: TransformFunction,
    dataset_id: Optional[str] = None,
    credentials_path: Optional[str] = None,
    location: Optional[str] = None,
    **client_kwargs,
) -> BigQueryAdapter:
    """Factory function to create a BigQuery adapter.

    Args:
        transform_fn: Function to transform BigQuery rows to evaluation format
        dataset_id: Google Cloud project ID
        credentials_path: Path to service account JSON file
        location: Default location for BigQuery jobs
        **client_kwargs: Additional arguments for BigQuery client

    Returns:
        BigQueryAdapter instance
    """
    return BigQueryAdapter(
        transform_fn=transform_fn,
        dataset_id=dataset_id,
        credentials_path=credentials_path,
        location=location,
        **client_kwargs,
    )
