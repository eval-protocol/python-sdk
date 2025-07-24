"""
Resource Pool for the Agent Evaluation Framework V2.
Manages and allocates resources to specific tasks.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Type

from .resource_abc import ForkableResource


class ResourcePool:
    """
    Manages a pool of ForkableResources that can be shared and reused across tasks.
    Provides tracking and lifecycle management for resources.
    """

    def __init__(self):
        """Initialize an empty resource pool."""
        self.resources: Dict[str, ForkableResource] = {}  # resource_id -> resource instance
        self.resource_tasks: Dict[str, Set[str]] = {}  # resource_id -> set of task_ids using it
        self.task_resources: Dict[str, Set[str]] = {}  # task_id -> set of resource_ids used by it
        self.logger = logging.getLogger("ResourcePool")

    async def create_resource(
        self,
        resource_type: Type[ForkableResource],
        resource_id: str,
        config: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Optional[ForkableResource]:
        """
        Create a new resource of the specified type and add it to the pool.

        Args:
            resource_type: The ForkableResource class to instantiate
            resource_id: Unique identifier for the resource
            config: Configuration dictionary for the resource setup
            task_id: Optional task ID to associate with this resource

        Returns:
            The created resource or None if creation fails
        """
        if resource_id in self.resources:
            self.logger.warning(f"Resource '{resource_id}' already exists in the pool. Returning existing instance.")
            if task_id:
                self._associate_task_with_resource(task_id, resource_id)
            return self.resources[resource_id]

        try:
            resource = resource_type()
            await resource.setup(config)

            self.resources[resource_id] = resource
            self.resource_tasks[resource_id] = set()

            if task_id:
                self._associate_task_with_resource(task_id, resource_id)

            self.logger.info(f"Created resource '{resource_id}' of type {resource_type.__name__}")
            return resource
        except Exception as e:
            self.logger.error(f"Failed to create resource '{resource_id}': {e}")
            return None

    def get_resource(self, resource_id: str) -> Optional[ForkableResource]:
        """
        Get a resource from the pool by its ID.

        Args:
            resource_id: The identifier of the resource to retrieve

        Returns:
            The resource instance or None if not found
        """
        return self.resources.get(resource_id)

    def _associate_task_with_resource(self, task_id: str, resource_id: str) -> None:
        """
        Associate a task with a resource for tracking purposes.

        Args:
            task_id: The task identifier
            resource_id: The resource identifier
        """
        if resource_id not in self.resources:
            self.logger.warning(f"Cannot associate task '{task_id}' with non-existent resource '{resource_id}'.")
            return

        # Add task to resource's task set
        if resource_id not in self.resource_tasks:
            self.resource_tasks[resource_id] = set()
        self.resource_tasks[resource_id].add(task_id)

        # Add resource to task's resource set
        if task_id not in self.task_resources:
            self.task_resources[task_id] = set()
        self.task_resources[task_id].add(resource_id)

        self.logger.debug(f"Associated task '{task_id}' with resource '{resource_id}'.")

    async def fork_resource_for_task(self, resource_id: str, task_id: str) -> Optional[ForkableResource]:
        """
        Fork a resource for a specific task.

        Args:
            resource_id: The identifier of the resource to fork
            task_id: The task that will use the forked resource

        Returns:
            The forked resource instance or None if forking fails
        """
        base_resource = self.get_resource(resource_id)
        if not base_resource:
            self.logger.error(f"Cannot fork non-existent resource '{resource_id}'.")
            return None

        try:
            forked_resource = await base_resource.fork()
            # We don't track forked resources in the pool, as they are typically
            # short-lived and managed by the Orchestrator
            self.logger.debug(f"Forked resource '{resource_id}' for task '{task_id}'.")
            return forked_resource
        except Exception as e:
            self.logger.error(f"Failed to fork resource '{resource_id}' for task '{task_id}': {e}")
            return None

    async def cleanup_task_resources(self, task_id: str) -> None:
        """
        Clean up all resources associated with a task.

        Args:
            task_id: The task identifier
        """
        if task_id not in self.task_resources:
            self.logger.debug(f"No resources to clean up for task '{task_id}'.")
            return

        resource_ids = list(self.task_resources[task_id])
        for resource_id in resource_ids:
            # Remove task from resource's task set
            if resource_id in self.resource_tasks:
                self.resource_tasks[resource_id].discard(task_id)

                # If resource has no more tasks, close and remove it
                if not self.resource_tasks[resource_id]:
                    await self.close_resource(resource_id)

        # Clear task's resource tracking
        self.task_resources.pop(task_id, None)
        self.logger.info(f"Cleaned up resources for task '{task_id}'.")

    async def close_resource(self, resource_id: str) -> None:
        """
        Close a resource and remove it from the pool.

        Args:
            resource_id: The identifier of the resource to close
        """
        if resource_id not in self.resources:
            self.logger.debug(f"Cannot close non-existent resource '{resource_id}'.")
            return

        resource = self.resources[resource_id]
        try:
            await resource.close()
            self.resources.pop(resource_id)
            self.resource_tasks.pop(resource_id, None)
            self.logger.info(f"Closed and removed resource '{resource_id}' from pool.")
        except Exception as e:
            self.logger.error(f"Error closing resource '{resource_id}': {e}")

    async def close_all_resources(self) -> None:
        """Close all resources in the pool and clear it."""
        resource_ids = list(self.resources.keys())
        for resource_id in resource_ids:
            await self.close_resource(resource_id)

        # Clear all tracking dictionaries
        self.resources.clear()
        self.resource_tasks.clear()
        self.task_resources.clear()
        self.logger.info("Closed all resources in the pool.")
