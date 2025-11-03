"""Centralized client for making requests to Fireworks API with consistent headers."""

import os
from typing import Any, Dict, Optional

import requests

from .common_utils import get_user_agent


class FireworksAPIClient:
    """Client for making authenticated requests to Fireworks API with proper headers.

    This client automatically includes:
    - Authorization header (Bearer token)
    - User-Agent header for tracking eval-protocol CLI usage
    """

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """Initialize the Fireworks API client.

        Args:
            api_key: Fireworks API key. If None, will be read from environment.
            api_base: API base URL. If None, defaults to https://api.fireworks.ai
        """
        self.api_key = api_key
        self.api_base = api_base or os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
        self._session = requests.Session()

    def _validate_path_is_relative(self, path: str) -> None:
        """Validate that the path is relative, not an absolute URL.

        Args:
            path: The path to validate

        Raises:
            ValueError: If path appears to be an absolute URL (starts with http:// or https://)
        """
        if path.startswith(("http://", "https://")):
            raise ValueError(
                f"Absolute URL detected: '{path}'. FireworksAPIClient methods expect relative paths only. "
                f"Use a relative path like 'v1/path' instead of '{path}'. "
                f"The client will automatically prepend the api_base: '{self.api_base}'"
            )

    def _get_headers(
        self, content_type: Optional[str] = "application/json", additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Build headers for API requests.

        Args:
            content_type: Content-Type header value. If None, Content-Type won't be set.
            additional_headers: Additional headers to merge in.

        Returns:
            Dictionary of headers including authorization and user-agent.
        """
        headers = {
            "User-Agent": get_user_agent(),
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if content_type:
            headers["Content-Type"] = content_type

        if additional_headers:
            headers.update(additional_headers)

        return headers

    def get(
        self, path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30, **kwargs
    ) -> requests.Response:
        """Make a GET request to the Fireworks API.

        Args:
            path: API path (relative to api_base)
            params: Query parameters
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to requests.get

        Returns:
            Response object
        """
        self._validate_path_is_relative(path)
        url = f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"
        headers = self._get_headers(content_type=None)
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        return self._session.get(url, params=params, headers=headers, timeout=timeout, **kwargs)

    def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        **kwargs,
    ) -> requests.Response:
        """Make a POST request to the Fireworks API.

        Args:
            path: API path (relative to api_base)
            json: JSON payload
            data: Form data payload
            files: Files to upload
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to requests.post

        Returns:
            Response object
        """
        self._validate_path_is_relative(path)
        url = f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"

        # For file uploads, don't set Content-Type (let requests handle multipart/form-data)
        content_type = None if files else "application/json"
        headers = self._get_headers(content_type=content_type)

        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        return self._session.post(url, json=json, data=data, files=files, headers=headers, timeout=timeout, **kwargs)

    def put(self, path: str, json: Optional[Dict[str, Any]] = None, timeout: int = 60, **kwargs) -> requests.Response:
        """Make a PUT request to the Fireworks API."""
        self._validate_path_is_relative(path)
        url = f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"
        headers = self._get_headers()
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        return self._session.put(url, json=json, headers=headers, timeout=timeout, **kwargs)

    def patch(
        self, path: str, json: Optional[Dict[str, Any]] = None, timeout: int = 60, **kwargs
    ) -> requests.Response:
        """Make a PATCH request to the Fireworks API."""
        self._validate_path_is_relative(path)
        url = f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"
        headers = self._get_headers()
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        return self._session.patch(url, json=json, headers=headers, timeout=timeout, **kwargs)

    def delete(self, path: str, timeout: int = 30, **kwargs) -> requests.Response:
        """Make a DELETE request to the Fireworks API."""
        self._validate_path_is_relative(path)
        url = f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"
        headers = self._get_headers(content_type=None)
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        return self._session.delete(url, headers=headers, timeout=timeout, **kwargs)
