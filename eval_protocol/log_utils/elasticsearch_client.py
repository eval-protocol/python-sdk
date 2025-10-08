"""
Centralized Elasticsearch client for all Elasticsearch API operations.

This module provides a unified interface for all Elasticsearch operations
used throughout the codebase, including index management, document operations,
and search functionality.
"""

import requests
from typing import Any, Dict, List, Optional, Union
from eval_protocol.models import Status
from eval_protocol.types.remote_rollout_processor import ElasticsearchConfig


class ElasticsearchClient:
    """Centralized client for all Elasticsearch operations."""

    def __init__(self, config: ElasticsearchConfig):
        """Initialize the Elasticsearch client.

        Args:
            config: Elasticsearch configuration
        """
        self.config = config
        self.base_url = config.url.rstrip("/")
        self.index_url = f"{self.base_url}/{config.index_name}"
        self._headers = {"Content-Type": "application/json", "Authorization": f"ApiKey {config.api_key}"}

    def _make_request(
        self,
        method: str,
        url: str,
        json_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> requests.Response:
        """Make an HTTP request to Elasticsearch.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, HEAD)
            url: Full URL for the request
            json_data: JSON data to send in request body
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            requests.Response object

        Raises:
            requests.RequestException: If the request fails
        """
        return requests.request(
            method=method,
            url=url,
            headers=self._headers,
            json=json_data,
            params=params,
            verify=self.config.verify_ssl,
            timeout=timeout,
        )

    # Index Management Operations

    def create_index(self, mapping: Dict[str, Any]) -> bool:
        """Create an index with the specified mapping.

        Args:
            mapping: Index mapping configuration

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self._make_request("PUT", self.index_url, json_data=mapping)
            return response.status_code in [200, 201]
        except Exception:
            return False

    def index_exists(self) -> bool:
        """Check if the index exists.

        Returns:
            bool: True if index exists, False otherwise
        """
        try:
            response = self._make_request("HEAD", self.index_url)
            return response.status_code == 200
        except Exception:
            return False

    def delete_index(self) -> bool:
        """Delete the index.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self._make_request("DELETE", self.index_url)
            return response.status_code in [200, 404]  # 404 means index doesn't exist
        except Exception:
            return False

    def clear_index(self) -> bool:
        """Clear all documents from the index.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete all documents by query
            response = self._make_request(
                "POST", f"{self.index_url}/_delete_by_query", json_data={"query": {"match_all": {}}}
            )
            if response.status_code == 200:
                # Refresh the index to ensure changes are visible
                refresh_response = self._make_request("POST", f"{self.index_url}/_refresh")
                return refresh_response.status_code == 200
            return False
        except Exception:
            return False

    def get_mapping(self) -> Optional[Dict[str, Any]]:
        """Get the index mapping.

        Returns:
            Dict containing mapping data, or None if failed
        """
        try:
            response = self._make_request("GET", f"{self.index_url}/_mapping")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def get_index_stats(self) -> Optional[Dict[str, Any]]:
        """Get index statistics.

        Returns:
            Dict containing index statistics, or None if failed
        """
        try:
            response = self._make_request("GET", f"{self.index_url}/_stats")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    # Document Operations

    def index_document(self, document: Dict[str, Any], doc_id: Optional[str] = None) -> bool:
        """Index a document.

        Args:
            document: Document to index
            doc_id: Optional document ID

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if doc_id:
                url = f"{self.index_url}/_doc/{doc_id}"
            else:
                url = f"{self.index_url}/_doc"

            response = self._make_request("POST", url, json_data=document)
            return response.status_code in [200, 201]
        except Exception:
            return False

    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Bulk index multiple documents.

        Args:
            documents: List of documents to index

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare bulk request body
            bulk_body = []
            for doc in documents:
                bulk_body.append({"index": {}})
                bulk_body.append(doc)

            response = self._make_request("POST", f"{self.index_url}/_bulk", json_data=bulk_body)
            return response.status_code == 200
        except Exception:
            return False

    # Search Operations

    def search(
        self, query: Dict[str, Any], size: int = 10, from_: int = 0, sort: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Search documents in the index.

        Args:
            query: Elasticsearch query
            size: Number of results to return
            from_: Starting offset
            sort: Sort specification

        Returns:
            Dict containing search results, or None if failed
        """
        try:
            search_body = {"query": query, "size": size, "from": from_}

            if sort:
                search_body["sort"] = sort

            response = self._make_request("POST", f"{self.index_url}/_search", json_data=search_body)

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def search_by_term(
        self, field: str, value: Any, size: int = 10, sort: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Search documents by exact term match.

        Args:
            field: Field name to search
            value: Value to match
            size: Number of results to return
            sort: Sort specification

        Returns:
            Dict containing search results, or None if failed
        """
        query = {"term": {field: value}}
        return self.search(query, size=size, sort=sort)

    def search_by_match(
        self, field: str, value: str, size: int = 10, sort: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Search documents by text match.

        Args:
            field: Field name to search
            value: Text to match
            size: Number of results to return
            sort: Sort specification (e.g., [{"@timestamp": {"order": "desc"}}])

        Returns:
            Dict containing search results, or None if failed
        """
        query = {"match": {field: value}}
        return self.search(query, size=size, sort=sort)

    def search_by_match_phrase_prefix(self, field: str, value: str, size: int = 10) -> Optional[Dict[str, Any]]:
        """Search documents by phrase prefix match.

        Args:
            field: Field name to search
            value: Phrase prefix to match
            size: Number of results to return

        Returns:
            Dict containing search results, or None if failed
        """
        query = {"match_phrase_prefix": {field: value}}
        return self.search(query, size=size)

    def search_all(self, size: int = 10) -> Optional[Dict[str, Any]]:
        """Search all documents in the index.

        Args:
            size: Number of results to return

        Returns:
            Dict containing search results, or None if failed
        """
        query = {"match_all": {}}
        return self.search(query, size=size)

    def search_by_status_code_not_in(
        self,
        rollout_id: str,
        excluded_codes: list[Status.Code],
        size: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        Search documents where status_code does NOT match any of the provided status codes.

        Args:
            excluded_codes: List of status codes to exclude (i.e., find logs NOT having these codes)
            size: Number of results to return
            rollout_id: Optional rollout ID to filter by

        Returns:
            Dict containing search results, or None if failed
        """
        # Build the query with must_not for status code exclusion
        bool_query: dict[str, list[dict[str, Any]]] = {
            "must_not": [{"terms": {"status_code": [code.value for code in excluded_codes]}}]
        }

        # Add rollout_id filter and ensure status_code exists
        bool_query["must"] = [{"term": {"rollout_id": rollout_id}}, {"exists": {"field": "status_code"}}]

        query = {"bool": bool_query}
        return self.search(query, size=size)

    # Health and Status Operations

    def health_check(self) -> bool:
        """Check if Elasticsearch is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            response = self._make_request("GET", f"{self.base_url}/_cluster/health")
            return response.status_code == 200
        except Exception:
            return False

    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """Get cluster information.

        Returns:
            Dict containing cluster info, or None if failed
        """
        try:
            response = self._make_request("GET", f"{self.base_url}/_cluster/health")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
