import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse


class ElasticsearchIndexManager:
    """Manages Elasticsearch index creation and mapping configuration."""

    def __init__(self, base_url: str, index_name: str, api_key: str) -> None:
        """Initialize the Elasticsearch index manager.

        Args:
            base_url: Elasticsearch base URL (e.g., "https://localhost:9200")
            index_name: Name of the index to manage
            api_key: API key for authentication
        """
        self.base_url: str = base_url.rstrip("/")
        self.index_name: str = index_name
        self.api_key: str = api_key
        self.index_url: str = f"{self.base_url}/{self.index_name}"
        self._mapping_created: bool = False

        # Parse URL to determine if we should verify SSL
        parsed_url = urlparse(base_url)
        self.verify_ssl = parsed_url.scheme == "https"

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
                print(f"Warning: Index {self.index_name} exists with incorrect mapping. Deleting and recreating...")
                if not self.delete_index():
                    print(f"Warning: Failed to delete existing index {self.index_name}")
                    return False

            # Create index with proper mapping
            mapping = self._get_logging_mapping()
            response = requests.put(
                self.index_url,
                headers={"Content-Type": "application/json", "Authorization": f"ApiKey {self.api_key}"},
                json=mapping,
                verify=self.verify_ssl,
            )

            if response.status_code in [200, 201]:
                self._mapping_created = True
                return True
            else:
                print(f"Warning: Failed to create index mapping: {response.status_code} - {response.text}")
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
            response = requests.head(
                self.index_url, headers={"Authorization": f"ApiKey {self.api_key}"}, verify=self.verify_ssl
            )

            if response.status_code != 200:
                return False

            # Check if mapping is correct
            mapping_response = requests.get(
                f"{self.index_url}/_mapping",
                headers={"Authorization": f"ApiKey {self.api_key}"},
                verify=self.verify_ssl,
            )

            if mapping_response.status_code != 200:
                return False

            mapping_data = mapping_response.json()
            return self._has_correct_timestamp_mapping(mapping_data)

        except Exception:
            return False

    def _has_correct_timestamp_mapping(self, mapping_data: Dict[str, Any]) -> bool:
        """Check if the mapping has @timestamp as a date field.

        Args:
            mapping_data: Elasticsearch mapping response data

        Returns:
            bool: True if @timestamp is correctly mapped as date field
        """
        try:
            return (
                self.index_name in mapping_data
                and "mappings" in mapping_data[self.index_name]
                and "properties" in mapping_data[self.index_name]["mappings"]
                and "@timestamp" in mapping_data[self.index_name]["mappings"]["properties"]
                and mapping_data[self.index_name]["mappings"]["properties"]["@timestamp"].get("type") == "date"
            )
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
                }
            }
        }

    def delete_index(self) -> bool:
        """Delete the managed index.

        Returns:
            bool: True if index was deleted successfully, False otherwise.
        """
        try:
            response = requests.delete(
                self.index_url, headers={"Authorization": f"ApiKey {self.api_key}"}, verify=self.verify_ssl
            )
            if response.status_code in [200, 404]:  # 404 means index doesn't exist, which is fine
                self._mapping_created = False
                return True
            else:
                print(f"Warning: Failed to delete index: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Warning: Failed to delete index: {e}")
            return False

    def index_exists(self) -> bool:
        """Check if the index exists.

        Returns:
            bool: True if index exists, False otherwise.
        """
        try:
            response = requests.head(
                self.index_url, headers={"Authorization": f"ApiKey {self.api_key}"}, verify=self.verify_ssl
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_index_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the index.

        Returns:
            Dict containing index statistics, or None if failed
        """
        try:
            response = requests.get(
                f"{self.index_url}/_stats",
                headers={"Authorization": f"ApiKey {self.api_key}"},
                verify=self.verify_ssl,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
