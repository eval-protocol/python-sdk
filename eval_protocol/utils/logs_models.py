"""
Pydantic models for the logs server API.

This module contains data models that match the TypeScript schemas in eval-protocol.ts
to ensure consistent data structure between Python backend and TypeScript frontend.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class LogEntry(BaseModel):
    """
    Represents a single log entry from Elasticsearch.

    This model matches the LogEntrySchema in eval-protocol.ts to ensure
    consistent data structure between Python backend and TypeScript frontend.
    """

    timestamp: str = Field(..., alias="@timestamp", description="ISO 8601 timestamp of the log entry")
    level: str = Field(..., description="Log level (DEBUG, INFO, WARNING, ERROR)")
    message: str = Field(..., description="The log message")
    logger_name: str = Field(..., description="Name of the logger that created this entry")
    rollout_id: str = Field(..., description="ID of the rollout this log belongs to")
    status_code: Optional[int] = Field(None, description="Optional status code")
    status_message: Optional[str] = Field(None, description="Optional status message")
    status_details: Optional[List[Any]] = Field(None, description="Optional status details")

    model_config = ConfigDict(populate_by_name=True)


class LogsResponse(BaseModel):
    """
    Response model for the get_logs endpoint.

    This model matches the LogsResponseSchema in eval-protocol.ts to ensure
    consistent data structure between Python backend and TypeScript frontend.
    """

    logs: List[LogEntry] = Field(..., description="Array of log entries")
    total: int = Field(..., description="Total number of logs available")
    rollout_id: str = Field(..., description="The rollout ID these logs belong to")
    filtered_by_level: Optional[str] = Field(None, description="Log level filter applied")

    model_config = ConfigDict()
