from abc import ABC, abstractmethod
from typing import Any, List


class EventBusDatabase(ABC):
    """
    Abstract base class for cross-process event communication storage.

    Implementations can use different storage backends (SQLite, TinyDB, etc.)
    """

    @abstractmethod
    def publish_event(self, event_type: str, data: Any, process_id: str) -> None:
        """
        Publish an event to the database.

        Args:
            event_type: Type of event (e.g., "log")
            data: Event data (will be serialized to JSON)
            process_id: ID of the publishing process
        """
        pass

    @abstractmethod
    def get_unprocessed_events(self, process_id: str) -> List[dict]:
        """
        Get unprocessed events from other processes.

        Args:
            process_id: Current process ID (events from this process are excluded)

        Returns:
            List of event dictionaries with keys: event_id, event_type, data, timestamp, process_id
        """
        pass

    @abstractmethod
    def mark_event_processed(self, event_id: str) -> None:
        """
        Mark an event as processed.

        Args:
            event_id: The event ID to mark as processed
        """
        pass

    @abstractmethod
    def cleanup_old_events(self, max_age_hours: int = 24) -> None:
        """
        Clean up old processed events.

        Args:
            max_age_hours: Maximum age in hours for processed events
        """
        pass
