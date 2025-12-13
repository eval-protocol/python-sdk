"""
Backwards-compatible alias for DatasetLoggerAdapter.

This module is kept for backwards compatibility. New code should use
DatasetLoggerAdapter from dataset_logger_adapter.py instead.
"""

from eval_protocol.dataset_logger.dataset_logger_adapter import DatasetLoggerAdapter

# Backwards-compatible alias
SqliteDatasetLoggerAdapter = DatasetLoggerAdapter

__all__ = ["SqliteDatasetLoggerAdapter"]
