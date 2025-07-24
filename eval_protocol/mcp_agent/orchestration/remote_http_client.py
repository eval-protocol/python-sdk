import logging
from typing import Any, Dict, List, Optional

import httpx
from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from eval_protocol.mcp_agent.config import (
    AppConfig,
    BackendServerConfig,
    RemoteApiConfig,
)
from eval_protocol.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)

logger = logging.getLogger(__name__)


class RemoteHttpOrchestrationClient(AbstractOrchestrationClient):
    """
    Orchestrates backend MCP server instances by communicating with a remote HTTP API.
    This client translates provisioning, deprovisioning, and tool call requests
    into HTTP requests to a customer-defined remote orchestration service.
    """

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.http_client: Optional[httpx.AsyncClient] = None

    async def startup(self) -> None:
        """Initializes the httpx client."""
        # Default timeout can be overridden by specific remote_api_config later
        timeout_config = httpx.Timeout(
            self.app_config.global_remote_api_defaults.get("timeout", 30.0),
            connect=self.app_config.global_remote_api_defaults.get("connect_timeout", 5.0),
        )
        self.http_client = httpx.AsyncClient(timeout=timeout_config)
        logger.info("RemoteHttpOrchestrationClient started.")

    async def shutdown(self) -> None:
        """Closes the httpx client."""
        if self.http_client:
            await self.http_client.aclose()
            logger.info("HTTPX client for RemoteHttpOrchestrationClient closed.")
        logger.info("RemoteHttpOrchestrationClient shut down.")

    def _get_auth_headers(self, remote_api_config: RemoteApiConfig) -> Dict[str, str]:
        """Constructs authentication headers based on the remote API config."""
        headers = {}
        if remote_api_config.auth_type == "bearer_token":
            token = remote_api_config.auth_details.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            else:
                logger.warning("Bearer token auth selected but no token provided.")
        elif remote_api_config.auth_type == "custom_header":
            header_name = remote_api_config.auth_details.get("header_name")
            header_value = remote_api_config.auth_details.get("header_value")
            if header_name and header_value:
                headers[header_name] = header_value
            else:
                logger.warning("Custom header auth selected but header_name or header_value missing.")
        return headers

    async def _make_request(
        self,
        method: str,
        url: str,
        remote_api_config: RemoteApiConfig,
        json_payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Helper method to make HTTP requests with authentication and error handling."""
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized. Call startup() first.")

        headers = self._get_auth_headers(remote_api_config)
        headers["Content-Type"] = "application/json"  # Assume JSON requests

        try:
            logger.debug(f"Making {method} request to {url} with payload: {json_payload} and params: {params}")
            response = await self.http_client.request(method, url, headers=headers, json=json_payload, params=params)
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses
            return response
        except httpx.RequestError as e:
            logger.error(f"Request error during {method} to {url}: {e}")
            raise RuntimeError(f"Remote API request failed: Network error calling {url}") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error during {method} to {url}: {e.response.status_code} - {e.response.text}")
            try:
                error_details = e.response.json()
            except Exception:
                error_details = e.response.text
            raise RuntimeError(
                f"Remote API request failed: Server returned error {e.response.status_code}. Details: {error_details}"
            ) from e

    async def provision_instances(
        self,
        backend_config: BackendServerConfig,
        num_instances: int,
        session_id: str,
        template_details: Optional[Any] = None,
    ) -> List[ManagedInstanceInfo]:
        if backend_config.orchestration_mode != "remote_http_api":
            raise ValueError("RemoteHttpOrchestrationClient can only handle 'remote_http_api' mode.")

        remote_api_config = self.app_config.get_remote_api_config(backend_config)
        if not remote_api_config:
            raise ValueError(f"RemoteApiConfig not found for backend {backend_config.backend_name_ref}.")

        create_url = (
            f"{remote_api_config.base_url.rstrip('/')}/{remote_api_config.create_instance_endpoint.lstrip('/')}"
        )

        provisioned_instances_info: List[ManagedInstanceInfo] = []

        # The remote API might support batch creation or require individual calls.
        # This example assumes the remote API can take num_instances and returns a list.
        # Adjust if the API requires one call per instance.
        payload = {
            "resource_type_identifier": backend_config.remote_resource_type_identifier,
            "num_instances": num_instances,
            "session_id": session_id,
            "instance_scoping": backend_config.instance_scoping,
            "template_details": template_details,  # Pass along any template info
            # Add any other necessary parameters the remote API expects
        }

        logger.info(
            f"Requesting {num_instances} instances of type '{backend_config.remote_resource_type_identifier}' from {create_url}"
        )

        response = await self._make_request("POST", create_url, remote_api_config, json_payload=payload)
        response_data = response.json()  # Expecting a list of instance details

        if not isinstance(response_data, list):
            raise ValueError(
                f"Remote API at {create_url} did not return a list of instances. Response: {response_data}"
            )

        for i, inst_data in enumerate(response_data):
            # The remote API response should provide necessary details for ManagedInstanceInfo
            # Required: instance_id (client-facing), mcp_endpoint_url, internal_instance_details (like remote_instance_id)
            remote_instance_id = inst_data.get("remote_instance_id")
            mcp_endpoint_url = inst_data.get("mcp_endpoint_url")
            client_facing_instance_id = inst_data.get(
                "instance_id", f"{session_id}-{backend_config.backend_name_ref}-{i}"
            )

            if not remote_instance_id or not mcp_endpoint_url:
                logger.error(
                    f"Remote API response for instance missing 'remote_instance_id' or 'mcp_endpoint_url'. Data: {inst_data}"
                )
                # Decide on error handling: skip this instance, or fail all?
                # For now, let's raise an error if critical info is missing.
                raise ValueError(f"Remote API response for instance creation is incomplete: {inst_data}")

            provisioned_instances_info.append(
                ManagedInstanceInfo(
                    instance_id=client_facing_instance_id,
                    backend_name_ref=backend_config.backend_name_ref,
                    orchestration_mode="remote_http_api",
                    mcp_endpoint_url=mcp_endpoint_url,
                    internal_instance_details={
                        "remote_instance_id": remote_instance_id,
                        **inst_data.get("additional_details", {}),  # Any other info from remote
                    },
                )
            )
            logger.info(
                f"Instance {client_facing_instance_id} (Remote ID: {remote_instance_id}) provisioned. MCP Endpoint: {mcp_endpoint_url}"
            )

        if (
            len(provisioned_instances_info) != num_instances and num_instances > 0 and len(response_data) > 0
        ):  # if API supports batch and returns partial
            logger.warning(
                f"Requested {num_instances} but remote API returned details for {len(provisioned_instances_info)} instances."
            )

        return provisioned_instances_info

    async def deprovision_instances(self, instances: List[ManagedInstanceInfo]) -> None:
        for instance in instances:
            if instance.orchestration_mode != "remote_http_api":
                logger.warning(
                    f"Skipping deprovision for instance {instance.instance_id} as it's not remote_http_api."
                )
                continue

            # Need to find the BackendServerConfig that led to this instance to get its RemoteApiConfig
            backend_cfg = next(
                (b for b in self.app_config.backends if b.backend_name_ref == instance.backend_name_ref),
                None,
            )
            if not backend_cfg:
                logger.error(
                    f"Could not find BackendServerConfig for {instance.backend_name_ref} during deprovision of {instance.instance_id}"
                )
                continue

            remote_api_config = self.app_config.get_remote_api_config(backend_cfg)
            if not remote_api_config:
                logger.error(
                    f"RemoteApiConfig not found for backend {instance.backend_name_ref} during deprovision of {instance.instance_id}."
                )
                continue

            remote_instance_id = instance.internal_instance_details.get("remote_instance_id")
            if not remote_instance_id:
                logger.warning(f"No remote_instance_id found for instance {instance.instance_id}. Cannot deprovision.")
                continue

            delete_url_template = remote_api_config.delete_instance_endpoint_template
            delete_url = f"{remote_api_config.base_url.rstrip('/')}/{delete_url_template.lstrip('/').format(remote_instance_id=remote_instance_id)}"

            logger.info(f"Requesting deprovision of remote instance {remote_instance_id} via {delete_url}")
            try:
                await self._make_request("DELETE", delete_url, remote_api_config)
                logger.info(f"Successfully requested deprovision for remote instance {remote_instance_id}.")
            except Exception as e:
                # Log error but continue trying to deprovision other instances
                logger.error(f"Failed to deprovision remote instance {remote_instance_id}: {e}")

    async def call_tool_on_instance(
        self, instance: ManagedInstanceInfo, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if instance.orchestration_mode != "remote_http_api":
            raise ValueError("This client only handles remote_http_api instances.")

        backend_cfg = next(
            (b for b in self.app_config.backends if b.backend_name_ref == instance.backend_name_ref),
            None,
        )
        if not backend_cfg:
            raise RuntimeError(f"Could not find BackendServerConfig for {instance.backend_name_ref}")

        remote_api_config = self.app_config.get_remote_api_config(backend_cfg)
        if not remote_api_config:
            raise RuntimeError(f"RemoteApiConfig not found for backend {instance.backend_name_ref}.")

        mcp_payload = {"tool_name": tool_name, "arguments": tool_args}

        target_url: str
        # Check if tool calls are proxied through the orchestrator or made directly to the instance
        if remote_api_config.call_tool_endpoint_template:
            remote_instance_id = instance.internal_instance_details.get("remote_instance_id")
            if not remote_instance_id:
                raise ValueError(
                    f"Missing remote_instance_id for instance {instance.instance_id} when proxying tool call."
                )

            call_template = remote_api_config.call_tool_endpoint_template
            # The template might need remote_instance_id and potentially tool_name if it's part of the path
            # Assuming a generic proxy endpoint for now that takes tool_name in payload
            target_url = f"{remote_api_config.base_url.rstrip('/')}/{call_template.lstrip('/').format(remote_instance_id=remote_instance_id)}"
            # The payload to the proxy might need to be wrapped, e.g. including the actual MCP payload
            # For now, assume the proxy forwards the mcp_payload directly.
            logger.debug(f"Proxying tool {tool_name} to {target_url} for instance {instance.instance_id}")
        else:
            # Call tool directly on the instance's MCP endpoint
            target_url = instance.mcp_endpoint_url
            logger.debug(f"Calling tool {tool_name} directly on {target_url} for instance {instance.instance_id}")

        response = await self._make_request("POST", target_url, remote_api_config, json_payload=mcp_payload)
        return response.json()

    async def list_tools_on_instance(self, instance: ManagedInstanceInfo) -> mcp_types.ListToolsResult:
        if instance.orchestration_mode != "remote_http_api":
            raise ValueError("RemoteHttpOrchestrationClient can only list tools for 'remote_http_api' instances.")
        if instance.mcp_transport != "http" or not instance.mcp_endpoint_url:
            raise ValueError(
                f"Instance {instance.instance_id} ({instance.backend_name_ref}) is not configured for HTTP MCP transport or mcp_endpoint_url is missing."
            )

        # Assuming instance.mcp_endpoint_url is the base URL of the target MCP server
        # e.g., "http://localhost:12345"
        target_base_url = instance.mcp_endpoint_url.rstrip("/")

        logger.info(
            f"Listing tools for remote HTTP instance {instance.instance_id} ({instance.backend_name_ref}) at base URL {target_base_url}"
        )

        try:
            # streamablehttp_client will manage its own httpx.AsyncClient if one is not provided.
            # The context manager handles session.initialize() and session.close().
            async with streamablehttp_client(base_url=target_base_url) as session:  # type: ClientSession
                list_tools_result = await session.list_tools()
                logger.info(
                    f"Successfully listed {len(list_tools_result.tools)} tools from {target_base_url} for instance {instance.instance_id} ({instance.backend_name_ref})"
                )
                return list_tools_result
        except Exception as e:
            logger.error(
                f"Error listing tools from {target_base_url} for instance {instance.instance_id} ({instance.backend_name_ref}): {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to list tools from backend instance {instance.instance_id} ({instance.backend_name_ref}) at {target_base_url}"
            ) from e
