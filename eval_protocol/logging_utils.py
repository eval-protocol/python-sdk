#!/usr/bin/env python3
"""
Logging utilities for the eval_protocol package.

This module provides centralized logging configuration and utilities
for consistent logging across the eval_protocol package.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from eval_protocol.directory_utils import find_eval_protocol_dir


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name: Logger name
        log_file: Optional log file name (will be created in logs directory)
        level: Overall logger level
        console_level: Console handler level
        file_level: File handler level

    Returns:
        Configured logger instance
    """
    # Create logs directory under eval_protocol
    eval_protocol_dir = Path(find_eval_protocol_dir())
    logs_dir = eval_protocol_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)

    # Only configure if not already configured (has handlers and proper level)
    if logger.handlers and logger.level != logging.NOTSET:
        return logger

    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # Console handler - explicitly write to sys.stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file specified) - explicitly write to file only
    if log_file:
        log_file_path = logs_dir / log_file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logging
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. If it doesn't exist, create it with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up with defaults
    if not logger.handlers:
        # For eval_watcher, check if running in daemon mode
        if name == "eval_watcher":
            import sys

            # Check if running in daemon mode (subprocess)
            if "--daemon" in sys.argv:
                # Subprocess: log to file only
                logger = setup_logger(name, f"{name}.log", console_level=logging.CRITICAL)
            else:
                # Top-level: log to console only
                logger = setup_logger(name, None)
        else:
            logger = setup_logger(name, f"{name}.log")

    return logger


def log_evaluation_event(
    event_type: str, evaluation_id: str, message: str, level: int = logging.INFO, **kwargs
) -> None:
    """
    Log evaluation-specific events to a dedicated evaluation log file.

    Args:
        event_type: Type of event (e.g., 'start', 'complete', 'error')
        evaluation_id: Evaluation identifier
        message: Log message
        level: Log level
        **kwargs: Additional context to include in log
    """
    logger = get_logger("evaluation_events")

    # Create structured log entry
    log_entry = {"event_type": event_type, "evaluation_id": evaluation_id, "message": message, **kwargs}

    if level == logging.DEBUG:
        logger.debug(f"EVENT: {log_entry}")
    elif level == logging.INFO:
        logger.info(f"EVENT: {log_entry}")
    elif level == logging.WARNING:
        logger.warning(f"EVENT: {log_entry}")
    elif level == logging.ERROR:
        logger.error(f"EVENT: {log_entry}")
    elif level == logging.CRITICAL:
        logger.critical(f"EVENT: {log_entry}")


def log_performance_metric(metric_name: str, value: float, unit: str = "", context: Optional[dict] = None) -> None:
    """
    Log performance metrics to a dedicated metrics log file.

    Args:
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        context: Additional context information
    """
    logger = get_logger("performance_metrics")

    metric_entry = {"metric": metric_name, "value": value, "unit": unit, "context": context or {}}

    logger.info(f"METRIC: {metric_entry}")


def log_error_with_context(error: Exception, context: str, additional_info: Optional[dict] = None) -> None:
    """
    Log errors with additional context information.

    Args:
        error: The exception that occurred
        context: Context where the error occurred
        additional_info: Additional information about the error
    """
    logger = get_logger("errors")

    error_entry = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "additional_info": additional_info or {},
    }

    logger.error(f"ERROR: {error_entry}", exc_info=True)
