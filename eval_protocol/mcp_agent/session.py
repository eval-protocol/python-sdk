import logging
from typing import Dict, List, Optional, Set

from eval_protocol.mcp_agent.orchestration.base_client import ManagedInstanceInfo

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field

# Attempting to find ReadStream and WriteStream in a different location
# from mcp.server.streamable_transport import ReadStream, WriteStream # Original problematic import
# Option 1: Try mcp.server.transport
# from mcp.server.transport import ReadStream, WriteStream
# Option 2: If not found, use typing.Any as a fallback for type hints
from typing import Any as ReadStream  # Fallback if specific types are not found
from typing import Any as WriteStream

from mcp.server.session import ServerSession  # Correct base class

# Placeholder BaseSession class removed.
# IntermediarySession class is removed as we are using a separate data class.


@dataclass
class IntermediarySessionData:
    """
    Data class to hold custom state for an intermediary session.
    This state is managed by RewardKitIntermediaryServer and keyed by transport session_id.
    """

    session_id: str  # This is the transport-level session_id
    managed_backends: Dict[str, List[ManagedInstanceInfo]] = field(default_factory=dict)
    temporary_docker_images: Set[str] = field(default_factory=set)

    def add_managed_instances(self, backend_name_ref: str, instances: List[ManagedInstanceInfo]):
        """Adds a list of managed instances for a given backend reference."""
        if backend_name_ref not in self.managed_backends:
            self.managed_backends[backend_name_ref] = []
        self.managed_backends[backend_name_ref].extend(instances)
        logger.info(
            f"SessionData {self.session_id}: Added {len(instances)} instances for backend '{backend_name_ref}'."
        )
        for instance in instances:
            if instance.committed_image_tag:
                self.temporary_docker_images.add(instance.committed_image_tag)
                logger.debug(
                    f"SessionData {self.session_id}: Tracking temporary image '{instance.committed_image_tag}'."
                )

    def get_managed_instances(
        self, backend_name_ref: str, instance_id: Optional[str] = None
    ) -> List[ManagedInstanceInfo]:
        """
        Retrieves managed instances for a backend reference.
        If instance_id is provided, returns a list containing that specific instance (if found).
        Otherwise, returns all instances for the backend_name_ref.
        """
        backend_instances = self.managed_backends.get(backend_name_ref, [])
        if not backend_instances:
            return []

        if instance_id:
            for inst in backend_instances:
                if inst.instance_id == instance_id:
                    return [inst]
            return []  # Specific instance_id not found

        return backend_instances

    def get_all_managed_instances(self) -> List[ManagedInstanceInfo]:
        """Returns a flat list of all managed instances in this session data."""
        all_instances = []
        for instances in self.managed_backends.values():
            all_instances.extend(instances)
        return all_instances


# Note: The IntermediarySession class that inherited from ServerSession has been removed.
# The RewardKitIntermediaryServer will now manage IntermediarySessionData instances directly.
