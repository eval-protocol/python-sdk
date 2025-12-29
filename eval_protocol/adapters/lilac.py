"""
Lilac ML integration for Eval Protocol.

This adapter provides utilities for converting between EvaluationRow format
and Lilac dataset format, enabling powerful data curation features like:
- Clustering and deduplication
- Semantic search and filtering
- Quality scoring with embeddings
- Interactive data exploration

Prerequisites:
    pip install 'lilac[all]'

Example usage:
    >>> from eval_protocol.adapters.lilac import (
    ...     evaluation_rows_to_lilac_dataset,
    ...     lilac_dataset_to_evaluation_rows,
    ... )
    >>>
    >>> # Convert EvaluationRows to Lilac dataset
    >>> dataset = evaluation_rows_to_lilac_dataset(rows, name='my-traces')
    >>>
    >>> # Do Lilac operations (cluster, filter, etc.)
    >>> dataset.cluster('messages_json')  # or create your own text column
    >>>
    >>> # Convert back to EvaluationRows
    >>> processed_rows = lilac_dataset_to_evaluation_rows(dataset)
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

import pandas as pd

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    ExecutionMetadata,
    InputMetadata,
    Message,
)

if TYPE_CHECKING:
    import lilac as ll

logger = logging.getLogger(__name__)

# Check if lilac is available
try:
    import lilac as ll

    LILAC_AVAILABLE = True
except ImportError:
    LILAC_AVAILABLE = False
    ll = None  # type: ignore


def _ensure_lilac_available() -> None:
    """Raise ImportError if lilac is not installed."""
    if not LILAC_AVAILABLE:
        raise ImportError("Lilac is not installed. Install it with: pip install 'lilac[all]'")


# =============================================================================
# Core Conversion Functions
# =============================================================================


def _serialize_message(msg: Message) -> dict[str, Any]:
    """Serialize a Message to a dictionary."""
    return msg.model_dump(exclude_none=True)


def _deserialize_messages(messages_json: str | None) -> list[Message]:
    """Deserialize messages JSON back to Message objects."""
    if not messages_json:
        return []
    try:
        messages_data = json.loads(messages_json)
        return [Message.model_validate(m) for m in messages_data]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to deserialize messages: {e}")
        return []


def evaluation_row_to_dict(row: EvaluationRow) -> dict[str, Any]:
    """Convert a single EvaluationRow to a dictionary for Lilac.

    The output contains JSON-serialized fields that can be reconstructed back
    to EvaluationRow. Users can add their own text columns for clustering.
    """
    result: dict[str, Any] = {
        # Identifiers
        "row_id": row.input_metadata.row_id if row.input_metadata else None,
        # Full data as JSON (for reconstruction)
        "messages_json": json.dumps([_serialize_message(m) for m in row.messages]),
        "tools_json": json.dumps(row.tools) if row.tools else None,
        "ground_truth_json": json.dumps(row.ground_truth) if row.ground_truth else None,
        "input_metadata_json": row.input_metadata.model_dump_json() if row.input_metadata else None,
        "execution_metadata_json": row.execution_metadata.model_dump_json() if row.execution_metadata else None,
        "evaluation_result_json": row.evaluation_result.model_dump_json() if row.evaluation_result else None,
        # Scalar fields for filtering
        "score": row.evaluation_result.score if row.evaluation_result else None,
        "message_count": len(row.messages),
        "has_tools": bool(row.tools),
    }

    return result


def dict_to_evaluation_row(data: dict[str, Any]) -> EvaluationRow:
    """Convert a Lilac row dictionary back to an EvaluationRow."""
    # Parse messages
    messages = _deserialize_messages(data.get("messages_json"))

    # Parse tools
    tools = None
    if data.get("tools_json"):
        try:
            tools = json.loads(data["tools_json"])
        except json.JSONDecodeError:
            pass

    # Parse ground truth
    ground_truth = None
    if data.get("ground_truth_json"):
        try:
            ground_truth = json.loads(data["ground_truth_json"])
        except json.JSONDecodeError:
            pass

    # Parse input metadata
    input_metadata = InputMetadata()
    if data.get("input_metadata_json"):
        try:
            input_metadata = InputMetadata.model_validate_json(data["input_metadata_json"])
        except (json.JSONDecodeError, ValueError):
            input_metadata = InputMetadata(row_id=data.get("row_id"))

    # Parse execution metadata
    execution_metadata = ExecutionMetadata()
    if data.get("execution_metadata_json"):
        try:
            execution_metadata = ExecutionMetadata.model_validate_json(data["execution_metadata_json"])
        except (json.JSONDecodeError, ValueError):
            pass

    # Parse evaluation result
    evaluation_result = None
    if data.get("evaluation_result_json"):
        try:
            evaluation_result = EvaluateResult.model_validate_json(data["evaluation_result_json"])
        except (json.JSONDecodeError, ValueError):
            pass

    return EvaluationRow(
        messages=messages,
        tools=tools,
        ground_truth=ground_truth,
        input_metadata=input_metadata,
        execution_metadata=execution_metadata,
        evaluation_result=evaluation_result,
    )


# =============================================================================
# Main Conversion Functions
# =============================================================================


def evaluation_rows_to_lilac_dataset(
    rows: list[EvaluationRow],
    namespace: str = "local",
    name: str = "eval-data",
    project_dir: str | None = None,
) -> Any:
    """Convert EvaluationRows to a Lilac dataset.

    Args:
        rows: List of EvaluationRow objects
        namespace: Lilac namespace (default: 'local')
        name: Dataset name
        project_dir: Lilac project directory (uses default if None)

    Returns:
        Lilac Dataset object ready for clustering, filtering, etc.

    Example:
        >>> dataset = evaluation_rows_to_lilac_dataset(rows, name='my-traces')
        >>>
        >>> # Add your own text column for clustering
        >>> df = dataset.to_pandas()
        >>> df['user_query'] = df['messages_json'].apply(extract_user_query)
        >>> # Re-create dataset with new column, then cluster
    """
    _ensure_lilac_available()
    import lilac as ll_module  # Re-import after ensuring available

    if project_dir:
        ll_module.set_project_dir(project_dir)

    # Convert to DataFrame
    records = [evaluation_row_to_dict(row) for row in rows]
    df = pd.DataFrame(records)

    config = ll_module.DatasetConfig(
        namespace=namespace,
        name=name,
        source=ll_module.PandasSource(df),
    )

    return ll_module.create_dataset(config)


def lilac_dataset_to_evaluation_rows(
    dataset: Any,
    filters: list[tuple[str, str, Any]] | None = None,
    limit: int | None = None,
) -> list[EvaluationRow]:
    """Convert a Lilac dataset back to EvaluationRows.

    Args:
        dataset: Lilac Dataset object
        filters: Optional Lilac filter tuples, e.g. [('score', 'greater', 0.5)]
        limit: Maximum number of rows to return

    Returns:
        List of EvaluationRow objects
    """
    _ensure_lilac_available()

    # Build query
    kwargs: dict[str, Any] = {}
    if filters:
        kwargs["filters"] = filters
    if limit:
        kwargs["limit"] = limit

    df = dataset.select_rows(**kwargs).df()
    return dataframe_to_evaluation_rows(df)


def evaluation_rows_to_dataframe(rows: list[EvaluationRow]) -> pd.DataFrame:
    """Convert EvaluationRows to a pandas DataFrame.

    Useful if you want to work with the DataFrame directly.
    """
    records = [evaluation_row_to_dict(row) for row in rows]
    return pd.DataFrame(records)


def dataframe_to_evaluation_rows(df: pd.DataFrame) -> list[EvaluationRow]:
    """Convert a pandas DataFrame back to EvaluationRows."""
    rows = []
    for _, row_data in df.iterrows():
        try:
            row = dict_to_evaluation_row(row_data.to_dict())
            rows.append(row)
        except Exception as e:
            logger.warning(f"Failed to convert row: {e}")
            continue
    return rows
