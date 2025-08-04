import abc
from typing import Any, Dict, List, Literal, Optional

from mcp import types as mcp_types  # Added import
from pydantic import BaseModel, Field

from eval_protocol.mcp_agent.config import BackendServerConfig


class ManagedInstanceInfo(BaseModel):
    """
    Stores all necessary details to interact with a provisioned backend instance.
    """

    instance_id: str = Field(..., description="Client-facing ID for this instance within a session.")
    backend_name_ref: str = Field(..., description="Reference name of the backend configuration used.")
    orchestration_mode: Literal["local_docker", "remote_http_api"] = Field(
        ..., description="Orchestration mode used for this instance."
    )
    mcp_transport: Literal["http", "stdio"] = Field(..., description="MCP transport protocol used by this instance.")
    mcp_endpoint_url: Optional[str] = Field(
        None,
        description="The full MCP endpoint URL for this instance if using HTTP transport (e.g., 'http://localhost:12345/mcp'). None for stdio.",
    )
    internal_instance_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Orchestrator-specific details, e.g., {'container_id': '...', 'host_port': ...} for Docker or {'remote_instance_id': '...'}. Not directly used by the intermediary server logic after provisioning, but useful for deprovisioning.",
    )
    committed_image_tag: Optional[str] = Field(
        None,
        description="If local Docker orchestration created a temporary image via 'docker commit', this stores its tag for later cleanup.",
    )

    class Config:
        extra = "forbid"


class AbstractOrchestrationClient(abc.ABC):
    """
    Abstract base class for orchestration clients.
    Orchestration clients are responsible for provisioning, deprovisioning,
    and interacting with backend MCP server instances.
    """

    @abc.abstractmethod
    async def provision_instances(
        self,
        backend_config: BackendServerConfig,
        num_instances: int,
        session_id: str,
        # template_details might be specific to the backend type,
        # e.g., path to a database dump for DuckDB, or a directory for filesystem.
        template_details: Optional[Any] = None,
    ) -> List[ManagedInstanceInfo]:
        """
        Provisions a number of backend instances based on the given configuration.

        For stateful backends requiring a unique state from a template (e.g., local Docker with a template data path),
        this method might involve:
        1. Creating a temporary "template" instance/container.
        2. Seeding it with data from `template_details` or `backend_config.template_data_path_host`.
        3. Committing this template instance to a new, temporary image (for Docker).
        4. Starting `num_instances` from this temporary image.

        For stateless backends or those not requiring template-based forking, this is simpler.

        Args:
            backend_config: Configuration for the backend type to provision.
            num_instances: Number of instances to provision.
            session_id: The ID of the current intermediary session, useful for naming/tagging resources.
            template_details: Optional backend-specific details for initializing stateful instances.
                              This could be a path to a data file, a directory, or other structured data.

        Returns:
            A list of ManagedInstanceInfo objects, one for each provisioned instance.
        """
        pass

    @abc.abstractmethod
    async def deprovision_instances(self, instances: List[ManagedInstanceInfo]) -> None:
        """
        Deprovisions (e.g., stops and removes) the specified backend instances.
        Also handles cleanup of any temporary resources like committed Docker images.

        Args:
            instances: A list of ManagedInstanceInfo objects for the instances to deprovision.
        """
        pass

    @abc.abstractmethod
    async def call_tool_on_instance(
        self, instance: ManagedInstanceInfo, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calls a specific MCP tool on a given backend instance.

        Args:
            instance: The ManagedInstanceInfo for the target backend instance.
            tool_name: The name of the MCP tool to call.
            tool_args: A dictionary of arguments for the tool.

        Returns:
            A dictionary representing the JSON response from the tool call.
        """
        pass

    @abc.abstractmethod
    async def list_tools_on_instance(self, instance: ManagedInstanceInfo) -> mcp_types.ListToolsResult:
        """
        Lists all available tools on a given backend instance.

        Args:
            instance: The ManagedInstanceInfo for the target backend instance.

        Returns:
            A ListToolsResult object containing the tools available on the instance.
        """
        pass

    async def startup(self) -> None:
        """
        Optional: Perform any setup required when the orchestration client is initialized.
        e.g., check Docker connection, authenticate with remote API.
        """
        pass

    async def shutdown(self) -> None:
        """
        Optional: Perform any cleanup required when the orchestration client is shut down.
        e.g., clean up globally shared resources if any were managed by this client.
        """
        pass
