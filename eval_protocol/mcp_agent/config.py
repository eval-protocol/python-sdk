from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RemoteApiConfig(BaseModel):
    """Configuration for a remote orchestration API."""

    base_url: str = Field(..., description="Base URL of the remote orchestration API.")
    create_instance_endpoint: str = Field("/instances", description="Endpoint to create a new instance.")
    delete_instance_endpoint_template: str = Field(
        "/instances/{remote_instance_id}",
        description="Template for the endpoint to delete an instance. {remote_instance_id} will be replaced.",
    )
    call_tool_endpoint_template: Optional[str] = Field(
        None,
        description="Template for the endpoint to call a tool on an instance. Optional, if not provided, tools are called directly on the instance's mcp_endpoint_url.",
    )
    auth_type: Literal["none", "bearer_token", "custom_header"] = Field(
        "none", description="Authentication type for the remote API."
    )
    auth_details: Optional[Dict[str, str]] = Field(
        None,
        description="Authentication details, e.g., {'token': 'your_token'} or {'header_name': 'X-API-Key', 'header_value': 'your_key'}.",
    )


class BackendServerConfig(BaseModel):
    """Configuration for a backend server that the intermediary can manage or proxy."""

    backend_name_ref: str = Field(
        ...,
        description="Unique reference name for this backend configuration (e.g., 'workspace_fs', 'shared_fetch_service').",
    )
    backend_type: Literal["filesystem", "duckdb", "memory", "everything", "fetch", "time"] = Field(
        ..., description="The type of the backend server."
    )
    orchestration_mode: Literal["local_docker", "remote_http_api"] = Field(
        ..., description="How this backend server is orchestrated."
    )
    instance_scoping: Literal["session", "shared_global"] = Field(
        "session",
        description="Defines if instances are per-session or shared globally. 'session' implies stateful, 'shared_global' implies stateless.",
    )
    mcp_transport: Literal["http", "stdio"] = Field(
        "http",
        description="MCP transport protocol used by the backend server. Defaults to 'http'. If 'stdio', container_port and http-based startup_check_mcp_tool are ignored.",
    )

    # Local Docker Specific Fields
    docker_image: Optional[str] = Field(
        None, description="Docker image to use if orchestration_mode is 'local_docker'."
    )
    container_port: Optional[int] = Field(
        None,
        description="Internal port of the MCP application within the container (for HTTP). Required if orchestration_mode is 'local_docker'.",
    )
    template_data_path_host: Optional[str] = Field(
        None,
        description="Path on the host machine to data used for pre-seeding state in a template container (for 'docker commit').",
    )
    container_template_data_path: Optional[str] = Field(
        None,
        description="Mount path inside the template container where 'template_data_path_host' will be mounted.",
    )
    docker_run_args: Optional[List[str]] = Field(None, description="Additional arguments for 'docker run'.")
    startup_check_mcp_tool: Optional[Dict[str, Any]] = Field(
        None,
        description="An MCP tool call (e.g., {'tool_name': 'ping', 'arguments': {}}) to verify container startup.",
    )
    # Renamed from container_command_args for clarity with docker-py's 'command' kwarg
    container_command: Optional[List[str]] = Field(
        None,
        description="Command to run in the container. Overrides Docker image's CMD or passed as args to ENTRYPOINT.",
    )
    container_volumes: Optional[Dict[str, Dict[str, str]]] = Field(
        None,
        description="Volume mounts for the container, e.g., {'/host/path': {'bind': '/container/path', 'mode': 'rw'}}.",
    )

    # Remote API Specific Fields
    remote_api_config_ref: Optional[str] = Field(
        None,
        description="Reference to a globally defined RemoteApiConfig by its key/name. Used if orchestration_mode is 'remote_http_api'. Can be inline if not referencing a global one.",
    )
    remote_resource_type_identifier: Optional[str] = Field(
        None,
        description="Type identifier for the resource as known by the remote API (e.g., 'duckdb_v1', 'filesystem_large'). Required if orchestration_mode is 'remote_http_api'.",
    )
    # If remote_api_config_ref is not used, RemoteApiConfig can be defined inline
    remote_api_config_inline: Optional[RemoteApiConfig] = Field(
        None, description="Inline RemoteApiConfig if not using remote_api_config_ref."
    )

    class Config:
        extra = "forbid"

    def model_post_init(self, __context: Any) -> None:
        if self.orchestration_mode == "local_docker":
            if not self.docker_image:
                raise ValueError("docker_image must be set for local_docker orchestration mode.")
            # container_port is only required for http transport in local_docker mode
            if self.mcp_transport == "http" and not self.container_port:
                raise ValueError("container_port must be set for local_docker orchestration mode with http transport.")
        elif self.orchestration_mode == "remote_http_api":
            if not self.remote_resource_type_identifier:
                raise ValueError("remote_resource_type_identifier must be set for remote_http_api orchestration mode.")
            if not self.remote_api_config_ref and not self.remote_api_config_inline:
                raise ValueError(
                    "Either remote_api_config_ref or remote_api_config_inline must be set for remote_http_api orchestration mode."
                )
            if self.remote_api_config_ref and self.remote_api_config_inline:
                raise ValueError("Cannot set both remote_api_config_ref and remote_api_config_inline.")


class AppConfig(BaseModel):
    """Root configuration for the Eval Protocol Intermediary MCP Server."""

    backends: List[BackendServerConfig] = Field(
        default_factory=list,
        description="List of configurations for all backend types the intermediary can manage or proxy.",
    )
    global_remote_apis: Optional[Dict[str, RemoteApiConfig]] = Field(
        default_factory=dict,
        description="Globally defined remote API configurations, keyed by a reference name.",
    )
    log_level: str = Field("INFO", description="Logging level for the server.")
    global_docker_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Global Docker options, e.g., default network settings.",
    )
    global_remote_api_defaults: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Global defaults for remote API interactions, e.g., default timeouts.",
    )

    class Config:
        extra = "forbid"

    def get_remote_api_config(self, backend_cfg: BackendServerConfig) -> Optional[RemoteApiConfig]:
        if backend_cfg.orchestration_mode != "remote_http_api":
            return None
        if backend_cfg.remote_api_config_inline:
            return backend_cfg.remote_api_config_inline
        if backend_cfg.remote_api_config_ref and self.global_remote_apis:
            return self.global_remote_apis.get(backend_cfg.remote_api_config_ref)
        return None
