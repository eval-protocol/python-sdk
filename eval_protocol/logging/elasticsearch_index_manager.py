from typing import Dict, Any, Optional
from .elasticsearch_client import ElasticsearchClient
from eval_protocol.types.remote_rollout_processor import ElasticsearchConfig


class ElasticsearchIndexManager:
    """Manages Elasticsearch index creation and mapping configuration."""

    def __init__(self, base_url: str, index_name: str, api_key: str) -> None:
        """Initialize the Elasticsearch index manager.

        Args:
            base_url: Elasticsearch base URL (e.g., "https://localhost:9200")
            index_name: Name of the index to manage
            api_key: API key for authentication
        """
        self.config = ElasticsearchConfig(url=base_url, api_key=api_key, index_name=index_name)
        self.client = ElasticsearchClient(self.config)
        self._mapping_created: bool = False

    def create_logging_index_mapping(self) -> bool:
        """Create index with proper mapping for logging data.

        Returns:
            bool: True if mapping was created successfully, False otherwise.
        """
        if self._mapping_created:
            return True

        try:
            # Check if index exists and has correct mapping
            if self._index_exists_with_correct_mapping():
                self._mapping_created = True
                return True

            # If index exists but has wrong mapping, delete and recreate it
            if self.index_exists():
                print(
                    f"Warning: Index {self.config.index_name} exists with incorrect mapping. Deleting and recreating..."
                )
                if not self.delete_index():
                    print(f"Warning: Failed to delete existing index {self.config.index_name}")
                    return False

            # Create index with proper mapping
            mapping = self._get_logging_mapping()
            success = self.client.create_index(mapping)

            if success:
                self._mapping_created = True
                return True
            else:
                print("Warning: Failed to create index mapping")
                return False

        except Exception as e:
            print(f"Warning: Failed to create index mapping: {e}")
            return False

    def _index_exists_with_correct_mapping(self) -> bool:
        """Check if index exists and has the correct @timestamp mapping.

        Returns:
            bool: True if index exists with correct mapping, False otherwise.
        """
        try:
            # Check if index exists
            if not self.client.index_exists():
                return False

            # Check if mapping is correct
            mapping_data = self.client.get_mapping()
            if mapping_data is None:
                return False

            return self._has_correct_timestamp_mapping(mapping_data)

        except Exception:
            return False

    def _has_correct_timestamp_mapping(self, mapping_data: Dict[str, Any]) -> bool:
        """Check if the mapping has @timestamp as a date field, rollout_id as a keyword field, and status fields.

        Args:
            mapping_data: Elasticsearch mapping response data

        Returns:
            bool: True if all required fields are correctly mapped
        """
        try:
            if not (
                self.config.index_name in mapping_data
                and "mappings" in mapping_data[self.config.index_name]
                and "properties" in mapping_data[self.config.index_name]["mappings"]
            ):
                return False

            properties = mapping_data[self.config.index_name]["mappings"]["properties"]

            # Check @timestamp is mapped as date
            timestamp_ok = "@timestamp" in properties and properties["@timestamp"].get("type") == "date"

            # Check rollout_id is mapped as keyword
            rollout_id_ok = "rollout_id" in properties and properties["rollout_id"].get("type") == "keyword"

            # Check status fields are mapped correctly
            status_code_ok = "status_code" in properties and properties["status_code"].get("type") == "integer"
            status_message_ok = "status_message" in properties and properties["status_message"].get("type") == "text"
            status_details_ok = "status_details" in properties and properties["status_details"].get("type") == "object"

            return timestamp_ok and rollout_id_ok and status_code_ok and status_message_ok and status_details_ok
        except (KeyError, TypeError):
            return False

    def _get_logging_mapping(self) -> Dict[str, Any]:
        """Get the standard mapping for logging data.

        Returns:
            Dict containing the index mapping configuration
        """
        return {
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "level": {"type": "keyword"},
                    "message": {"type": "text"},
                    "logger_name": {"type": "keyword"},
                    "rollout_id": {"type": "keyword"},
                    "status_code": {"type": "integer"},
                    "status_message": {"type": "text"},
                    "status_details": {"type": "object"},
                }
            }
        }

    def delete_index(self) -> bool:
        """Delete the managed index.

        Returns:
            bool: True if index was deleted successfully, False otherwise.
        """
        try:
            success = self.client.delete_index()
            if success:
                self._mapping_created = False
                return True
            else:
                print("Warning: Failed to delete index")
                return False
        except Exception as e:
            print(f"Warning: Failed to delete index: {e}")
            return False

    def index_exists(self) -> bool:
        """Check if the index exists.

        Returns:
            bool: True if index exists, False otherwise.
        """
        return self.client.index_exists()

    def get_index_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the index.

        Returns:
            Dict containing index statistics, or None if failed
        """
        return self.client.get_index_stats()
