"""
DockerResource: A ForkableResource for managing Docker container states.
"""

import io
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..resource_abc import ForkableResource

# Attempt to import Docker SDK with error handling
try:
    import docker

    # Import for runtime; annotate as Any to avoid mismatched type aliasing across modules
    from docker.errors import APIError as _APIError, DockerException as _DockerException, NotFound as _NotFound
    from docker.models.containers import Container as _Container

    DOCKER_SDK_AVAILABLE = True
    # Ensure these are available for type checking even if the runtime import fails
    # The `else` block for DOCKER_SDK_AVAILABLE = False will define runtime dummies.
    DockerException = _DockerException  # type: ignore[assignment]
    NotFound = _NotFound  # type: ignore[assignment]
    APIError = _APIError  # type: ignore[assignment]
    Container = _Container  # type: ignore[assignment]
    try:
        _daemon_check_client = docker.from_env()
        _daemon_check_client.ping()
        DOCKER_DAEMON_AVAILABLE = True
    except Exception:
        DOCKER_DAEMON_AVAILABLE = False
    finally:
        try:
            _daemon_check_client.close()
        except Exception:
            pass

except ImportError:
    DOCKER_SDK_AVAILABLE = False
    DOCKER_DAEMON_AVAILABLE = False

    # Define dummy classes/exceptions if docker SDK is not available
    # These are only defined if the import fails.
    class DockerException(Exception):  # type: ignore[no-redef]
        pass

    class NotFound(DockerException):  # type: ignore[no-redef]
        pass

    class APIError(DockerException):  # type: ignore[no-redef]
        pass

    class Container:  # type: ignore[no-redef]
        id: str = ""
        name: str = ""
        image: Any = None
        status: str = ""
        ports: Dict[str, Any] = {}

        def remove(self, force: bool = False, v: bool = False) -> None:
            pass

        def commit(self, **kwargs: Any) -> Any:
            return None

        def reload(self) -> None:
            pass

        def start(self) -> None:
            pass

        def exec_run(self, **kwargs: Any) -> Tuple[int, bytes]:
            return (0, b"")

        def logs(self, **kwargs: Any) -> bytes:
            return b""


class DockerResource(ForkableResource):
    """
    A ForkableResource for managing Docker container states.

    Allows initializing a container from an image, forking (by committing the
    current container and starting a new one from the committed image),
    checkpointing (committing to an image), and restoring from a checkpoint image.
    Commands can be executed within the container.

    Requires the Docker SDK (`docker` pip package) to be installed and Docker daemon running.
    """

    def __init__(self) -> None:
        if not DOCKER_SDK_AVAILABLE:
            raise ImportError("Docker SDK not found. Please install 'docker' package to use DockerResource.")
        if not DOCKER_DAEMON_AVAILABLE:
            raise RuntimeError("Docker daemon not running or not accessible")
        self._client = docker.from_env()
        self._config: Dict[str, Any] = {}
        self._container: Optional[Any] = None
        self._image_id_for_fork_or_checkpoint: Optional[str] = (
            None  # Stores the ID of the image used for the current container
        )
        self._is_closed = False  # To prevent operations on closed resource

    def _generate_name(self, prefix: str) -> str:
        return f"rk_{prefix}_{uuid.uuid4().hex}"

    def _cleanup_container(self, container: Optional[Any]) -> None:
        if container:
            try:
                container.remove(force=True, v=True)  # v=True to remove volumes
            except NotFound:
                pass  # Already removed
            except APIError as e:
                print(f"DockerResource: Error removing container {(getattr(container, 'id', '') or '')[:12]}: {e}")

    def _cleanup_image(self, image_id: Optional[str]) -> None:
        if image_id:
            try:
                self._client.images.remove(image=image_id, force=True)
            except NotFound:
                pass  # Already removed
            except APIError as e:
                # Often "image is being used by stopped container" if cleanup order is tricky
                print(f"DockerResource: Error removing image {image_id[:12]}: {e}")

    async def setup(self, config: Dict[str, Any]) -> None:
        """
        Initializes and starts a Docker container based on the provided configuration.

        Args:
            config: Configuration dictionary. Expected keys:
                - 'image_name' (str): Name of the Docker image to use (e.g., 'ubuntu:latest').
                - 'container_name' (Optional[str]): Name for the container. Defaults to a UUID.
                - 'docker_run_options' (Optional[Dict[str, Any]]): Options for docker.client.containers.run()
                  e.g., {'detach': True, 'ports': {'80/tcp': 8080}, 'environment': ["VAR=value"]}
                  'detach' will always be True.
        """
        if self._is_closed:
            raise RuntimeError("Cannot setup a closed DockerResource.")
        self._config = config.copy()

        image_name = self._config.get("image_name")
        if not image_name:
            raise ValueError("Missing 'image_name' in DockerResource config.")

        # Pull the image if not present locally (optional, could be pre-pulled)
        try:
            self._client.images.get(image_name)
        except NotFound:
            print(f"DockerResource: Image '{image_name}' not found locally. Pulling...")
            try:
                self._client.images.pull(image_name)
            except APIError as e:
                raise DockerException(f"Failed to pull image '{image_name}': {e}") from e

        self._image_id_for_fork_or_checkpoint = image_name  # Base image for the first container

        container_name = self._config.get("container_name", self._generate_name("container"))
        run_options = self._config.get("docker_run_options", {}).copy()
        run_options["detach"] = True  # Must be detached for this model
        run_options["name"] = container_name

        # Clean up any existing container with the same name (e.g. from a failed previous run)
        try:
            existing_container = self._client.containers.get(container_name)
            self._cleanup_container(existing_container)
        except NotFound:
            pass

        try:
            self._container = self._client.containers.run(image_name, **run_options)
            if self._container:
                self._container.reload()  # Ensure state is up-to-date
        except APIError as e:
            raise DockerException(
                f"Failed to start container '{container_name}' from image '{image_name}': {e}"
            ) from e

    async def fork(self) -> "DockerResource":
        """
        Creates a new DockerResource by committing the current container's state
        to a new image and starting a new container from that image.
        """
        if self._is_closed or not self._container:
            raise RuntimeError("Cannot fork: resource is closed or not set up.")

        # 1. Commit current container to a new image
        fork_image_tag = self._generate_name("fork_img")
        try:
            committed_image = self._container.commit(repository=fork_image_tag)
        except APIError as e:
            raise DockerException(f"Failed to commit container {(self._container.id or '')[:12]} for fork: {e}") from e

        # 2. Create new DockerResource instance
        forked_resource = DockerResource()
        forked_resource._config = self._config.copy()  # Inherit original config

        # Modify config for the new container if needed (e.g., new name)
        forked_container_name = self._generate_name("fork_container")
        forked_resource._config["container_name"] = forked_container_name

        # The new container will run from the committed image
        forked_resource._image_id_for_fork_or_checkpoint = committed_image.id

        run_options = self._config.get("docker_run_options", {}).copy()
        run_options["detach"] = True
        run_options["name"] = forked_container_name

        try:
            forked_resource._container = self._client.containers.run(committed_image.id, **run_options)
            if forked_resource._container:
                forked_resource._container.reload()
        except APIError as e:
            self._cleanup_image(committed_image.id)  # Cleanup committed image if run fails
            raise DockerException(f"Failed to start forked container from image {committed_image.id[:12]}: {e}") from e

        return forked_resource

    async def checkpoint(self) -> Dict[str, Any]:
        """
        Checkpoints the container by committing its current state to a new image.
        Returns the ID of the committed image.
        """
        if self._is_closed or not self._container:
            raise RuntimeError("Cannot checkpoint: resource is closed or not set up.")

        checkpoint_image_tag = self._generate_name("checkpoint_img")
        try:
            committed_image = self._container.commit(repository=checkpoint_image_tag)
            return {"type": "docker_image_id", "image_id": committed_image.id}
        except APIError as e:
            raise DockerException(
                f"Failed to commit container {(self._container.id or '')[:12]} for checkpoint: {e}"
            ) from e

    async def restore(self, state_data: Dict[str, Any]) -> None:
        """
        Restores the resource by starting a new container from a checkpointed image ID.
        The existing container (if any) is stopped and removed.
        """
        if self._is_closed:
            raise RuntimeError("Cannot restore a closed DockerResource.")

        image_id = state_data.get("image_id")
        if state_data.get("type") != "docker_image_id" or not image_id:
            raise ValueError(
                "Invalid state_data for DockerResource restore. Expected {'type': 'docker_image_id', 'image_id': '...'}"
            )

        # Ensure the checkpointed image exists
        try:
            self._client.images.get(image_id)
        except NotFound:
            raise DockerException(f"Checkpoint image ID '{image_id}' not found.") from None

        # Cleanup existing container before restoring
        if self._container:
            self._cleanup_container(self._container)
            self._container = None

        # Update current image ID to the one we are restoring from
        self._image_id_for_fork_or_checkpoint = image_id

        restored_container_name = self._config.get("container_name", self._generate_name("restored_container"))
        # If a container_name was in original config, we might want to reuse it or ensure uniqueness
        self._config["container_name"] = restored_container_name  # Update config for consistency

        run_options = self._config.get("docker_run_options", {}).copy()
        run_options["detach"] = True
        run_options["name"] = restored_container_name

        try:
            self._container = self._client.containers.run(image_id, **run_options)
            if self._container:
                self._container.reload()
        except APIError as e:
            raise DockerException(f"Failed to start container from checkpoint image {image_id[:12]}: {e}") from e

    async def step(self, action_name: str, action_params: Dict[str, Any]) -> Any:
        """
        Executes a command inside the Docker container or performs other Docker actions.

        Supported actions:
        - 'exec_command': Executes a command inside the container.
            Params: {'command': str | List[str], 'workdir': Optional[str], 'user': Optional[str]}
            Returns: {'exit_code': int, 'output': bytes (stdout + stderr)}
        - 'get_logs': Retrieves container logs.
            Params: {'stdout': bool, 'stderr': bool, 'tail': int | 'all'}
            Returns: str (logs)
        """
        if self._is_closed or not self._container:
            raise RuntimeError("Cannot execute step: resource is closed or not set up.")

        self._container.reload()
        if self._container.status != "running":
            try:  # Attempt to start if stopped
                self._container.start()
                self._container.reload()
                if self._container.status != "running":
                    raise DockerException(
                        f"Container {(self._container.id or '')[:12]} is not running (status: {self._container.status}). Cannot execute step."
                    )
            except APIError as e:
                raise DockerException(
                    f"Failed to start container {(self._container.id or '')[:12]} for step: {e}"
                ) from e

        if action_name == "exec_command":
            command = action_params.get("command")
            if not command:
                raise ValueError("Missing 'command' in action_params for 'exec_command'.")

            exec_options = {
                "cmd": command,
                "stdout": True,
                "stderr": True,
                "workdir": action_params.get("workdir"),
                "user": action_params.get("user"),
                "demux": False,  # Get stdout and stderr interleaved as a single stream
            }
            # Filter out None values for docker SDK
            exec_options = {k: v for k, v in exec_options.items() if v is not None}

            try:
                exit_code, output_stream = self._container.exec_run(**exec_options)
                output_bytes = output_stream if output_stream else b""
                return {
                    "exit_code": exit_code,
                    "output": output_bytes.decode("utf-8", errors="replace"),
                }
            except APIError as e:
                raise DockerException(
                    f"Failed to execute command in container {(self._container.id or '')[:12]}: {e}"
                ) from e

        elif action_name == "get_logs":
            log_options = {
                "stdout": action_params.get("stdout", True),
                "stderr": action_params.get("stderr", True),
                "timestamps": action_params.get("timestamps", False),
                "tail": action_params.get("tail", "all"),
            }
            try:
                logs_bytes = self._container.logs(**log_options)
                return logs_bytes.decode("utf-8", errors="replace")
            except APIError as e:
                raise DockerException(
                    f"Failed to get logs for container {(self._container.id or '')[:12]}: {e}"
                ) from e
        else:
            raise NotImplementedError(f"Action '{action_name}' not supported by DockerResource.")

    async def get_observation(self) -> Dict[str, Any]:
        """
        Returns information about the current container.
        """
        if self._is_closed or not self._container:
            return {"status": "closed or not_initialized"}

        self._container.reload()
        return {
            "type": "docker",
            "container_id": self._container.id,
            "container_name": self._container.name,
            "image_id": (
                self._container.image.id
                if hasattr(self._container, "image") and self._container.image
                else self._image_id_for_fork_or_checkpoint
            ),
            "status": self._container.status,
            "ports": self._container.ports,
        }

    async def get_tools_spec(self) -> List[Dict[str, Any]]:
        """
        Returns tool specifications for interacting with the Docker container.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "Executes a command inside the Docker container.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                ],
                                "description": "The command to execute (string or list of strings).",
                            },
                            "workdir": {
                                "type": "string",
                                "description": "Working directory inside the container (optional).",
                            },
                            "user": {
                                "type": "string",
                                "description": "User to run command as (optional).",
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_logs",
                    "description": "Retrieves logs from the Docker container.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stdout": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include stdout.",
                            },
                            "stderr": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include stderr.",
                            },
                            "tail": {
                                "oneOf": [
                                    {"type": "integer"},
                                    {"type": "string", "enum": ["all"]},
                                ],
                                "default": "all",
                                "description": "Number of lines from end of logs or 'all'.",
                            },
                        },
                    },
                },
            },
        ]

    async def close(self) -> None:
        """
        Stops and removes the managed Docker container and any images created
        by this specific resource instance during fork/checkpoint if they are not
        the original base image.
        """
        if self._is_closed:
            return

        self._cleanup_container(self._container)
        self._container = None

        # Cleanup the image that this container was based on, IF it was a result of a fork/checkpoint
        # and not the original user-provided image_name from config.
        # This logic is a bit tricky: we only want to remove images we created.
        # self._image_id_for_fork_or_checkpoint stores the ID of the image the *current* container was made from.
        # If this ID is different from self._config.get("image_name") (the very first image),
        # then it's an image we created via commit.
        original_base_image_name = self._config.get("image_name")
        if self._image_id_for_fork_or_checkpoint and self._image_id_for_fork_or_checkpoint != original_base_image_name:
            # Check if the image ID is a full ID or a tag like the original.
            # This check might need refinement if original_base_image_name is an ID itself.
            if original_base_image_name is not None:
                try:
                    img_obj = self._client.images.get(original_base_image_name)
                    if img_obj.id != self._image_id_for_fork_or_checkpoint:
                        self._cleanup_image(self._image_id_for_fork_or_checkpoint)
                except NotFound:  # Original image name might not be an ID, or might have been removed.
                    self._cleanup_image(self._image_id_for_fork_or_checkpoint)
            else:  # original_base_image_name IS None
                # If original_base_image_name is None, but _image_id_for_fork_or_checkpoint is set
                # (and different from None, due to the outer if), then it's an image to clean up.
                if self._image_id_for_fork_or_checkpoint:
                    self._cleanup_image(self._image_id_for_fork_or_checkpoint)

        self._image_id_for_fork_or_checkpoint = None
        self._is_closed = True
