"""
Pandas DataFrame adapter for Eval Protocol.

This module provides utilities for converting between EvaluationRow format
and pandas DataFrame format, enabling integration with data curation tools
such as Lilac, Great Expectations, or any pandas-based workflow.

Example usage:
    >>> from eval_protocol.adapters.dataframe import (
    ...     evaluation_rows_to_dataframe,
    ...     dataframe_to_evaluation_rows,
    ... )
    >>>
    >>> # Convert EvaluationRows to DataFrame
    >>> df = evaluation_rows_to_dataframe(rows)
    >>>
    >>> # Convert back to EvaluationRows
    >>> rows = dataframe_to_evaluation_rows(df)
"""

from __future__ import annotations

import logging

import pandas as pd

from ..models import EvaluationRow

logger = logging.getLogger(__name__)


def evaluation_rows_to_dataframe(rows: list[EvaluationRow]) -> pd.DataFrame:
    """Convert EvaluationRows to a pandas DataFrame.

    Uses EvaluationRow.to_dict() for serialization.

    Args:
        rows: List of EvaluationRow objects

    Returns:
        DataFrame with 'data_json' containing serialized rows plus convenience fields
    """
    records = [row.to_dict() for row in rows]
    return pd.DataFrame(records)


def dataframe_to_evaluation_rows(df: pd.DataFrame) -> list[EvaluationRow]:
    """Convert a pandas DataFrame back to EvaluationRows.

    Uses EvaluationRow.from_dict() for deserialization.

    Args:
        df: DataFrame with 'data_json' column containing serialized EvaluationRows

    Returns:
        List of EvaluationRow objects
    """
    rows = []
    for _, row_data in df.iterrows():
        try:
            row = EvaluationRow.from_dict(row_data.to_dict())
            rows.append(row)
        except Exception as e:
            logger.warning(f"Failed to convert row: {e}")
            continue
    return rows
