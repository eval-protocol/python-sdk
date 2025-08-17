"""
Resource management for reward functions.

This module provides resource wrappers for external services like LLMs,
databases, etc. Resources are automatically setup and cleaned up by the
reward function decorator.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar("T")
ResourceDict = Dict[str, List["ResourceWrapper"]]


class ResourceWrapper(ABC):
    """Abstract base class for all resource wrappers."""

    @abstractmethod
    def setup(self) -> None:
        """Setup the resource (e.g., start deployment, create connection)."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the resource (e.g., stop deployment, close connection)."""
        pass

    @abstractmethod
    def get_client(self) -> Any:
        """Get the client object for using this resource."""
        pass


class LLMResourceWrapper(ResourceWrapper):
    """Resource wrapper for Fireworks LLM deployments."""

    def __init__(self, llm_instance: Any):
        """
        Initialize LLM resource wrapper.

        Args:
            llm_instance: A Fireworks LLM instance from the Build SDK
        """
        self.llm_instance = llm_instance
        self._client = None
        self._is_setup = False

    def setup(self) -> None:
        """Setup the LLM deployment."""
        if self._is_setup:
            logger.debug(f"LLM resource already setup for model: {self.llm_instance.model}")
            return

        try:
            logger.debug(f"Setting up LLM deployment for model: {self.llm_instance.model}")

            # For on-demand deployments, call apply()
            if hasattr(self.llm_instance, "deployment_type") and self.llm_instance.deployment_type == "on-demand":
                logger.info("Applying on-demand LLM deployment...")
                self.llm_instance.apply()
                logger.info("On-demand LLM deployment applied successfully")

            self._client = self.llm_instance
            self._is_setup = True

            logger.info(f"LLM resource setup completed for model: {self.llm_instance.model}")

        except Exception as e:
            logger.error(f"Failed to setup LLM resource: {e}")
            raise

    def cleanup(self) -> None:
        """Cleanup the LLM deployment."""
        if not self._is_setup:
            logger.debug("LLM resource not setup, nothing to cleanup")
            return

        try:
            logger.debug("Cleaning up LLM resource")

            # For Fireworks Build SDK, we typically don't need explicit
            # cleanup as deployments are managed by the platform
            self._client = None
            self._is_setup = False

            logger.debug("LLM resource cleanup completed")

        except Exception as e:
            logger.error(f"Error during LLM resource cleanup: {e}")
            # Don't re-raise cleanup errors to avoid masking original
            # exceptions

    def get_client(self) -> Any:
        """Get the LLM client for making API calls."""
        if not self._is_setup or self._client is None:
            raise RuntimeError("LLM resource not setup. Call setup() first.")
        return self._client


def create_llm_resource(llm_instance: Any) -> LLMResourceWrapper:
    """
    Create an LLM resource wrapper from a Fireworks LLM instance.

    Args:
        llm_instance: A Fireworks LLM instance from the Build SDK

    Returns:
        LLMResourceWrapper instance

    Example:
        ```python
        from fireworks import LLM
        from eval_protocol import create_llm_resource

        llm = LLM(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            deployment_type="on-demand",
        )

        llm_resource = create_llm_resource(llm)
        ```
    """
    return LLMResourceWrapper(llm_instance)
