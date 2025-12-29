"""
Lilac ML integration for Eval Protocol.

This adapter provides utilities for converting between EvaluationRow format
and pandas DataFrame format, enabling integration with Lilac for data curation:
- Clustering and deduplication
- Semantic search and filtering
- Quality scoring with embeddings
- Interactive data exploration

Example usage:
    >>> from eval_protocol.adapters.lilac import (
    ...     evaluation_rows_to_dataframe,
    ...     dataframe_to_evaluation_rows,
    ... )
    >>>
    >>> # Convert EvaluationRows to DataFrame for Lilac
    >>> df = evaluation_rows_to_dataframe(rows)
    >>> df['user_query'] = df['messages_json'].apply(extract_user_message)
    >>>
    >>> # Use with Lilac for clustering
    >>> import lilac as ll
    >>> dataset = ll.create_dataset(ll.DatasetConfig(
    ...     namespace='local', name='my-data', source=ll.PandasSource(df)
    ... ))
    >>> dataset.cluster('user_query')
    >>>
    >>> # Convert back to EvaluationRows
    >>> processed_df = dataset.to_pandas(include_signals=True)
    >>> processed_rows = dataframe_to_evaluation_rows(processed_df)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    ExecutionMetadata,
    InputMetadata,
    Message,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Internal Helpers
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


def _evaluation_row_to_dict(row: EvaluationRow) -> dict[str, Any]:
    """Convert a single EvaluationRow to a dictionary.

    The output contains JSON-serialized fields that can be reconstructed back
    to EvaluationRow. Users can add their own text columns for clustering.
    """
    return {
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


def _dict_to_evaluation_row(data: dict[str, Any]) -> EvaluationRow:
    """Convert a dictionary back to an EvaluationRow."""
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
# Public API
# =============================================================================


def evaluation_rows_to_dataframe(rows: list[EvaluationRow]) -> pd.DataFrame:
    """Convert EvaluationRows to a pandas DataFrame.

    The DataFrame can be used directly with Lilac for clustering and curation.

    Args:
        rows: List of EvaluationRow objects

    Returns:
        DataFrame with JSON-serialized fields for reconstruction
    """
    records = [_evaluation_row_to_dict(row) for row in rows]
    return pd.DataFrame(records)


def dataframe_to_evaluation_rows(df: pd.DataFrame) -> list[EvaluationRow]:
    """Convert a pandas DataFrame back to EvaluationRows.

    Args:
        df: DataFrame with messages_json and other serialized fields

    Returns:
        List of EvaluationRow objects
    """
    rows = []
    for _, row_data in df.iterrows():
        try:
            row = _dict_to_evaluation_row(row_data.to_dict())
            rows.append(row)
        except Exception as e:
            logger.warning(f"Failed to convert row: {e}")
            continue
    return rows
