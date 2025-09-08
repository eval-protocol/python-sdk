import asyncio
import logging
import shutil  # Added for directory copying
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

import docker
import docker.errors
import docker.models.containers
import httpx
import mcp.types as types
from anyio.abc import ObjectReceiveStream, ObjectSendStream

# ListToolsResult is not in mcp.client.session, likely in mcp.types or mcp.shared.message
from mcp.client.session import DEFAULT_CLIENT_INFO, ClientSession

# Assuming ListToolsResult is in mcp.types, which is imported as types
# If not, this will need further correction. For now, we'll use types.ListToolsResult
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import (  # Added for HTTP transport tool listing
    streamablehttp_client,
)

from eval_protocol.mcp_agent.config import AppConfig, BackendServerConfig
from eval_protocol.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)

logger = logging.getLogger(__name__)
ENCODING = "utf-8"
DEFAULT_INSTANCE_DATA_BASE_PATH = Path("/tmp/rk_mcp_instance_data")


class LocalDockerOrchestrationClient(AbstractOrchestrationClient):
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.docker_client: Optional[docker.DockerClient] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self._used_host_ports: Set[int] = set()
        self._temporary_images: Set[str] = set()
        self.instance_data_base_path = DEFAULT_INSTANCE_DATA_BASE_PATH

        self._stdio_instance_tasks: Dict[str, asyncio.Task] = {}
        self._stdio_client_sessions: Dict[str, ClientSession] = {}
        self._stdio_shutdown_events: Dict[str, asyncio.Event] = {}

    async def startup(self) -> None:
        self.instance_data_base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Instance data base path for host-copied templates: {self.instance_data_base_path.resolve()}")
        try:
            self.docker_client = docker.from_env()
            if not self.docker_client.ping():  # type: ignore
                raise ConnectionError("Failed to connect to Docker daemon using docker.from_env().")
            logger.info("Successfully connected to Docker daemon.")
        except docker.errors.DockerException as e:
            logger.warning(f"docker.from_env() failed: {e}. Trying explicit base_url.")
            try:
                # docker.from_env is preferred, but as a fallback use DockerClient with url param name 'base_url'
                self.docker_client = docker.DockerClient(base_url="unix://var/run/docker.sock")
                if not self.docker_client.ping():  # type: ignore
                    raise ConnectionError("Failed to connect to Docker daemon with explicit base_url.")
                logger.info("Successfully connected to Docker daemon with explicit base_url.")
            except docker.errors.DockerException as e_explicit:
                raise ConnectionError(f"Docker client initialization failed: {e_explicit}") from e_explicit

        api_defaults = (
            self.app_config.global_remote_api_defaults
            if isinstance(self.app_config.global_remote_api_defaults, dict)
            else {}
        )
        self.http_client = httpx.AsyncClient(timeout=api_defaults.get("timeout", 30.0))
        logger.info("LocalDockerOrchestrationClient started.")

    async def _manage_stdio_instance_lifecycle(
        self,
        instance_uuid: str,
        container_name: str,
        server_params: StdioServerParameters,
        initialization_complete_event: asyncio.Event,
        shutdown_event: asyncio.Event,
    ):
        client_session_stdio: Optional[ClientSession] = None
        try:
            logger.info(f"[{container_name}] Lifecycle task started.")
            async with stdio_client(server_params) as (read_stream, write_stream):
                logger.info(f"[{container_name}] Stdio transport established via stdio_client.")

                client_session_stdio = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    client_info=DEFAULT_CLIENT_INFO,
                )

                async with client_session_stdio:
                    logger.info(f"[{container_name}] Attempting to initialize ClientSession...")
                    await asyncio.wait_for(client_session_stdio.initialize(), timeout=15.0)
                    logger.info(f"[{container_name}] ClientSession initialized successfully.")

                    try:
                        # Corrected type hint assuming ListToolsResult is in mcp.types
                        list_tools_response: types.ListToolsResult = await asyncio.wait_for(
                            client_session_stdio.list_tools(), timeout=5.0
                        )
                        if hasattr(list_tools_response, "tools") and list_tools_response.tools is not None:
                            reported_tools = [
                                tool.name for tool in list_tools_response.tools
                            ]  # Assuming tool object has a .name
                            logger.info(f"[{container_name}] Backend server reported tools: {reported_tools}")
                        else:
                            logger.warning(
                                f"[{container_name}] Backend server list_tools response did not contain 'tools' attribute or it was None. Response: {list_tools_response}"
                            )
                    except AttributeError as e_attr:
                        logger.warning(
                            f"[{container_name}] AttributeError accessing tools from list_tools response: {e_attr}. Response: {getattr(list_tools_response, '__dict__', list_tools_response)}"
                        )
                    except Exception as e_list_tools:
                        logger.warning(
                            f"[{container_name}] Error calling/processing list_tools on backend server: {e_list_tools}"
                        )

                    self._stdio_client_sessions[instance_uuid] = client_session_stdio
                    initialization_complete_event.set()

                    await shutdown_event.wait()
                    logger.info(f"[{container_name}] Shutdown event received.")

            logger.info(f"[{container_name}] stdio_client context exited cleanly.")

        except asyncio.TimeoutError:
            logger.error(f"[{container_name}] Timeout during ClientSession initialization.")
            initialization_complete_event.set()
        except Exception as e:
            logger.error(
                f"[{container_name}] Error in stdio instance lifecycle: {e}",
                exc_info=True,
            )
            initialization_complete_event.set()
        finally:
            logger.debug(f"[{container_name}] In _manage_stdio_instance_lifecycle finally block.")
            if client_session_stdio is None:
                logger.info(f"[{container_name}] ClientSession was not created or assigned in lifecycle task.")

            self._stdio_client_sessions.pop(instance_uuid, None)
            self._stdio_shutdown_events.pop(instance_uuid, None)  # Ensure event is removed
            logger.info(f"[{container_name}] Lifecycle task finished.")

    async def shutdown(self) -> None:
        if self.http_client:
            await self.http_client.aclose()

        logger.info(
            f"Shutting down LocalDockerOrchestrationClient. Cleaning up {len(self._stdio_instance_tasks)} stdio instance tasks."
        )
        for instance_uuid, event in list(self._stdio_shutdown_events.items()):
            logger.info(f"Signaling shutdown for stdio instance task {instance_uuid}.")
            event.set()

        tasks_to_wait_for = list(self._stdio_instance_tasks.values())
        if tasks_to_wait_for:
            results = await asyncio.gather(*tasks_to_wait_for, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_name = (
                        tasks_to_wait_for[i].get_name() if hasattr(tasks_to_wait_for[i], "get_name") else f"Task-{i}"
                    )
                    logger.error(
                        f"Stdio lifecycle task {task_name} raised an exception during shutdown: {result}",
                        exc_info=result,
                    )
        logger.info("All stdio instance tasks awaited.")
        self._stdio_instance_tasks.clear()

        if self.docker_client:
            for image_tag in list(self._temporary_images):
                try:
                    self.docker_client.images.remove(image=image_tag, force=False)  # type: ignore
                    self._temporary_images.discard(image_tag)
                except Exception as e:
                    logger.warning(f"Failed to remove temp image {image_tag}: {e}")
            if hasattr(self.docker_client, "api") and hasattr(self.docker_client.api, "close"):
                self.docker_client.api.close()  # type: ignore
            elif hasattr(self.docker_client, "close"):
                self.docker_client.close()  # type: ignore
        logger.info("LocalDockerOrchestrationClient shut down.")

    async def _perform_startup_check(self, url: str, check: Dict[str, Any]) -> bool:
        # ... (content remains the same) ...
        if not self.http_client:
            return False
        name, args = check.get("tool_name"), check.get("arguments", {})
        if not name:
            return True
        for attempt in range(5):
            try:
                res = await self.http_client.post(url, json={"tool_name": name, "arguments": args})
                res.raise_for_status()
                return True
            except Exception as e:
                logger.warning(f"Startup check fail {attempt + 1}/5: {e}")
                if attempt < 4:
                    await asyncio.sleep(2)
        return False

    async def provision_instances(
        self,
        backend_config: BackendServerConfig,
        num_instances: int,
        session_id: str,
        template_details: Optional[Any] = None,  # template_details is generic, could be path for fs
    ) -> List[ManagedInstanceInfo]:
        if not self.docker_client:
            raise RuntimeError("Docker client not initialized.")

        image_to_run_from = backend_config.docker_image
        committed_img_tag: Optional[str] = None
        managed_instances: List[ManagedInstanceInfo] = []

        # Determine if we are using host copy for filesystem template
        use_host_copy_template = (
            backend_config.backend_type == "filesystem"
            and backend_config.template_data_path_host
            and Path(backend_config.template_data_path_host).is_dir()
        )

        # Image templating via docker commit (original logic)
        # This might be mutually exclusive with host_copy_template for filesystem, or could be combined if needed.
        # For now, assume host_copy_template takes precedence for filesystem if specified.
        if (
            not use_host_copy_template
            and backend_config.instance_scoping == "session"
            and (template_details or backend_config.template_data_path_host)
            and backend_config.container_template_data_path
        ):
            host_path_for_commit = template_details or backend_config.template_data_path_host
            if not host_path_for_commit or not backend_config.container_template_data_path:
                raise ValueError(
                    "template_data_path_host and container_template_data_path required for stateful session with image template."
                )

            temp_cont_name = f"rk-mcp-template-{session_id}-{backend_config.backend_name_ref}-{uuid.uuid4().hex[:4]}"
            try:
                logger.info(
                    f"Creating template container for commit: {temp_cont_name} from {backend_config.docker_image}"
                )
                if not backend_config.docker_image:
                    raise ValueError(
                        f"docker_image is required for template commit for backend {backend_config.backend_name_ref}"
                    )
                temp_c = self.docker_client.containers.run(  # type: ignore
                    image=backend_config.docker_image,
                    name=temp_cont_name,
                    volumes={
                        str(Path(host_path_for_commit).resolve()): {
                            "bind": backend_config.container_template_data_path,
                            "mode": "rw",
                        }
                    },
                    detach=True,
                )
                # Allow time for potential init scripts in container to modify state from template
                # This duration might need to be configurable or based on a health check.
                await asyncio.sleep(
                    self.app_config.global_docker_options.get("template_commit_delay_s", 5)
                    if self.app_config.global_docker_options
                    else 5
                )

                committed_img_tag = (
                    f"rk-mcp-templateimg-{session_id}-{backend_config.backend_name_ref}:{uuid.uuid4().hex[:6]}"
                )
                logger.info(f"Committing {temp_c.id} to {committed_img_tag}")  # type: ignore
                temp_c.commit(repository=committed_img_tag.split(":")[0], tag=committed_img_tag.split(":")[1])  # type: ignore
                image_to_run_from = committed_img_tag
                self._temporary_images.add(committed_img_tag)
            finally:
                if "temp_c" in locals() and temp_c:
                    try:
                        temp_c.stop(timeout=5)
                        temp_c.remove()  # type: ignore
                    except Exception as e:
                        logger.warning(f"Could not cleanup template container for commit: {e}")

        for i in range(num_instances):
            instance_uuid = uuid.uuid4().hex[:8]
            container_name = f"rk-mcp-inst-{session_id}-{backend_config.backend_name_ref}-{instance_uuid}"
            mcp_endpoint_url: Optional[str] = None
            host_port: Optional[int] = None
            instance_internal_details: Dict[str, Any] = {
                "container_name": container_name,
                "instance_uuid": instance_uuid,
            }
            current_container_volumes = dict(backend_config.container_volumes or {})  # Start with configured volumes

            try:
                if use_host_copy_template and backend_config.template_data_path_host:
                    instance_host_data_path = (
                        self.instance_data_base_path / session_id / backend_config.backend_name_ref / instance_uuid
                    )
                    instance_host_data_path.mkdir(parents=True, exist_ok=True)

                    logger.info(
                        f"Copying template from {backend_config.template_data_path_host} to {instance_host_data_path} for instance {container_name}"
                    )
                    shutil.copytree(
                        backend_config.template_data_path_host,
                        instance_host_data_path,
                        dirs_exist_ok=True,
                    )

                    instance_internal_details["instance_host_data_path"] = str(instance_host_data_path.resolve())
                    # Override/set the volume for /data (assuming /data is the target for mcp/filesystem)
                    # The container_command for mcp/filesystem is often ["/data"], so it serves what's at /data.
                    container_data_path_target = (
                        "/data"  # This should ideally come from config or be standard for "filesystem" type
                    )
                    current_container_volumes = {
                        str(instance_host_data_path.resolve()): {
                            "bind": container_data_path_target,
                            "mode": "rw",
                        }
                    }
                    logger.info(f"Using dynamic volume for {container_name}: {current_container_volumes}")

                logger.info(
                    f"Provisioning instance {container_name} (transport: {backend_config.mcp_transport}) from image {image_to_run_from}"
                )
                # Ensure the image reference is present before using it in Docker APIs
                if not image_to_run_from:
                    raise ValueError(
                        f"docker_image is required to provision instance {container_name} for backend {backend_config.backend_name_ref}"
                    )
                if backend_config.mcp_transport == "http":
                    # ... (HTTP provisioning logic, ensure it uses current_container_volumes) ...
                    if not self.docker_client:
                        raise RuntimeError("Docker client not initialized for HTTP provisioning.")
                    if not backend_config.container_port:
                        raise ValueError("container_port required for http.")
                    port_bindings = {f"{backend_config.container_port}/tcp": 0}
                    run_kwargs: Dict[str, Any] = {
                        "image": image_to_run_from,
                        "name": container_name,
                        "detach": True,
                        "command": backend_config.container_command,
                        "volumes": current_container_volumes,  # Use potentially modified volumes
                        "labels": {
                            "rewardkit-mcp-session-id": session_id,
                            "rewardkit-mcp-backend-name": backend_config.backend_name_ref,
                            "rewardkit-mcp-instance-id": instance_uuid,
                            "rewardkit-mcp-managed": "true",
                        },
                        "ports": port_bindings,
                        **(self.app_config.global_docker_options or {}),
                    }
                    container = self.docker_client.containers.run(**run_kwargs)
                    container.reload()
                    bindings = (
                        container.attrs.get("NetworkSettings", {})
                        .get("Ports", {})
                        .get(f"{backend_config.container_port}/tcp")
                    )
                    if not (bindings and bindings[0].get("HostPort")):
                        logs = "N/A"
                        try:
                            logs = container.logs(stdout=True, stderr=True).decode(ENCODING, "replace")
                        except Exception:
                            pass
                        logger.error(f"Failed to get host port for {container_name}. Logs:\n{logs}")
                        try:
                            container.stop(timeout=5)
                            container.remove()
                        except Exception:
                            pass
                        raise RuntimeError(f"Failed to get host port for {container_name}")
                    host_port = int(bindings[0]["HostPort"])
                    self._used_host_ports.add(host_port)
                    mcp_endpoint_url = f"http://localhost:{host_port}/mcp"  # Assuming /mcp path
                    instance_internal_details["container_id"] = container.id  # Store container_id earlier
                    if backend_config.startup_check_mcp_tool and not await self._perform_startup_check(
                        mcp_endpoint_url, backend_config.startup_check_mcp_tool
                    ):
                        logs = "N/A"
                        try:
                            logs = container.logs(stdout=True, stderr=True).decode(ENCODING, "replace")
                        except Exception:
                            pass
                        logger.error(f"HTTP Startup check failed for {container_name}. Logs:\n{logs}")
                        try:
                            container.stop(timeout=5)
                            container.remove()
                        except Exception:
                            pass
                        self._used_host_ports.discard(host_port)
                        raise RuntimeError(f"Startup check failed for {container_name}")
                    logger.info(f"HTTP Instance {container_name} (ID: {container.id}) on port {host_port}")
                    instance_internal_details.update({"host_port": host_port})

                elif backend_config.mcp_transport == "stdio":
                    docker_run_args = ["run", "--rm", "-i", "--name", container_name]
                    # Use current_container_volumes which might have been dynamically set by host-copy template logic
                    if current_container_volumes:
                        for h_path, c_path_dict in current_container_volumes.items():
                            bind_path, mode = c_path_dict.get("bind"), c_path_dict.get("mode", "rw")
                            if bind_path:
                                docker_run_args.extend(
                                    ["-v", f"{h_path}:{bind_path}:{mode}"]
                                )  # h_path is already resolved if from instance_host_data_path

                    docker_run_args.append(image_to_run_from)
                    if backend_config.container_command:  # This is the command for the MCP server inside docker
                        docker_run_args.extend(backend_config.container_command)

                    # The StdioServerParameters command should be "docker" and args are the docker run command
                    # The backend_config.mcp_server_stdio_command is for *inside* the container if we were to exec.
                    # Here, we are running the container itself as the stdio server process.
                    server_params = StdioServerParameters(command="docker", args=docker_run_args, env=dict(os.environ))  # type: ignore
                    logger.info(
                        f"Preparing to launch stdio container {container_name} via dedicated task with command: docker {' '.join(docker_run_args)}"
                    )

                    initialization_complete_event = asyncio.Event()
                    shutdown_event = asyncio.Event()
                    self._stdio_shutdown_events[instance_uuid] = shutdown_event

                    lifecycle_task = asyncio.create_task(
                        self._manage_stdio_instance_lifecycle(
                            instance_uuid,
                            container_name,
                            server_params,
                            initialization_complete_event,
                            shutdown_event,
                        )
                    )
                    self._stdio_instance_tasks[instance_uuid] = lifecycle_task

                    logger.info(f"Waiting for stdio instance {container_name} (task) to complete initialization...")
                    await asyncio.wait_for(initialization_complete_event.wait(), timeout=30.0)

                    client_session_stdio = self._stdio_client_sessions.get(instance_uuid)
                    if not client_session_stdio:
                        if lifecycle_task.done() and lifecycle_task.exception():
                            raise RuntimeError(
                                f"Stdio instance task for {container_name} failed during initialization."
                            ) from lifecycle_task.exception()
                        raise RuntimeError(f"ClientSession not established by lifecycle task for {container_name}.")

                    logger.info(
                        f"Stdio instance {container_name} (task) initialization complete. ClientSession ready."
                    )
                    instance_internal_details["container_name"] = container_name  # Already set, but ensure it's there

                    if backend_config.startup_check_mcp_tool:
                        logger.info(f"Performing startup check for stdio instance {container_name}...")
                        startup_tool_name = backend_config.startup_check_mcp_tool.get("tool_name", "ping")
                        startup_tool_args = backend_config.startup_check_mcp_tool.get("arguments", {})
                        # The session is already active in the lifecycle task, do not re-enter context manager
                        await asyncio.wait_for(
                            client_session_stdio.call_tool(startup_tool_name, startup_tool_args),
                            timeout=10.0,
                        )
                        logger.info(f"Stdio startup check for {container_name} successful.")
                else:
                    raise ValueError(f"Unsupported mcp_transport: {backend_config.mcp_transport}")

                managed_instances.append(
                    ManagedInstanceInfo(
                        instance_id=instance_uuid,
                        backend_name_ref=backend_config.backend_name_ref,
                        orchestration_mode="local_docker",
                        mcp_transport=backend_config.mcp_transport,
                        mcp_endpoint_url=mcp_endpoint_url,
                        internal_instance_details=instance_internal_details,
                        committed_image_tag=committed_img_tag,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to provision instance {container_name}: {e}", exc_info=True)
                if backend_config.mcp_transport == "stdio":
                    if instance_uuid in self._stdio_shutdown_events:
                        self._stdio_shutdown_events[instance_uuid].set()
                    task_to_clean = self._stdio_instance_tasks.pop(instance_uuid, None)
                    if task_to_clean and not task_to_clean.done():
                        try:
                            await asyncio.wait_for(task_to_clean, timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Timeout waiting for stdio task {instance_uuid} to clean up after provisioning error."
                            )
                            task_to_clean.cancel()
                        except Exception as task_e:
                            logger.error(f"Exception during stdio task cleanup for {instance_uuid}: {task_e}")
                # Cleanup copied host directory if provisioning failed mid-way
                if "instance_host_data_path" in instance_internal_details:
                    shutil.rmtree(
                        instance_internal_details["instance_host_data_path"],
                        ignore_errors=True,
                    )
                    logger.info(
                        f"Cleaned up instance data directory {instance_internal_details['instance_host_data_path']} due to provisioning error."
                    )
                raise
        return managed_instances

    async def deprovision_instances(self, instances: List[ManagedInstanceInfo]) -> None:
        if not self.docker_client:
            logger.warning("Docker client not init for deprovision.")

        for instance in instances:
            if instance.orchestration_mode != "local_docker":
                continue

            details = instance.internal_instance_details
            instance_uuid = details.get("instance_uuid", instance.instance_id)

            if instance.mcp_transport == "http":
                container_id = details.get("container_id")
                if not container_id or not self.docker_client:
                    continue
                try:
                    container = self.docker_client.containers.get(container_id)
                    container.stop(timeout=10)
                    container.remove()
                    logger.info(f"HTTP Container {container_id} deprovisioned.")
                    if details.get("host_port"):
                        self._used_host_ports.discard(details["host_port"])
                except Exception as e:
                    logger.error(f"Error deprovisioning HTTP container {container_id}: {e}")

            elif instance.mcp_transport == "stdio":
                logger.info(f"Deprovisioning stdio instance {instance_uuid} ({details.get('container_name')})...")

                shutdown_event = self._stdio_shutdown_events.pop(instance_uuid, None)
                if shutdown_event:
                    logger.info(f"Signaling shutdown for stdio instance task {instance_uuid}.")
                    shutdown_event.set()
                else:
                    logger.warning(f"No shutdown event found for stdio instance {instance_uuid}.")

                task = self._stdio_instance_tasks.pop(instance_uuid, None)
                if task:
                    logger.info(f"Waiting for stdio instance task {instance_uuid} to complete...")
                    try:
                        await asyncio.wait_for(task, timeout=10.0)
                        logger.info(f"Stdio instance task {instance_uuid} completed.")
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Timeout waiting for stdio instance task {instance_uuid} to complete. Cancelling."
                        )
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            logger.info(f"Stdio instance task {instance_uuid} cancelled.")
                        except Exception as e_task_cancel:
                            logger.error(
                                f"Exception during cancellation of stdio task {instance_uuid}: {e_task_cancel}"
                            )
                    except Exception as e_task_wait:
                        logger.error(
                            f"Exception waiting for stdio instance task {instance_uuid}: {e_task_wait}",
                            exc_info=True,
                        )
                else:
                    logger.warning(f"No lifecycle task found for stdio instance {instance_uuid} during deprovision.")

                if instance_uuid in self._stdio_client_sessions:
                    logger.warning(
                        f"ClientSession for {instance_uuid} still in _stdio_client_sessions after task handling. Popping."
                    )
                    self._stdio_client_sessions.pop(instance_uuid, None)
                logger.info(f"Stdio instance {instance_uuid} deprovisioning process complete.")

            # Cleanup copied host directory if it exists
            instance_host_data_path_str = details.get("instance_host_data_path")
            if instance_host_data_path_str:
                logger.info(f"Cleaning up instance data directory: {instance_host_data_path_str}")
                shutil.rmtree(instance_host_data_path_str, ignore_errors=True)

    async def call_tool_on_instance(
        self, instance: ManagedInstanceInfo, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if instance.orchestration_mode != "local_docker":
            raise ValueError("Only handles local_docker instances.")

        if instance.mcp_transport == "http":
            if not self.http_client:
                raise RuntimeError("HTTP client not initialized.")
            if not instance.mcp_endpoint_url:
                raise ValueError(f"mcp_endpoint_url required for HTTP {instance.instance_id}")
            payload = {"tool_name": tool_name, "arguments": tool_args}
            try:
                res = await self.http_client.post(instance.mcp_endpoint_url, json=payload)
                res.raise_for_status()
                return res.json()
            except Exception as e:
                raise RuntimeError(f"MCP HTTP call failed: {e}") from e

        elif instance.mcp_transport == "stdio":
            instance_uuid = instance.internal_instance_details.get("instance_uuid", instance.instance_id)
            cs = self._stdio_client_sessions.get(instance_uuid)

            if not cs or not isinstance(cs, ClientSession):
                raise RuntimeError(f"Valid ClientSession not found for stdio instance {instance_uuid}.")

            try:
                logger.debug(
                    f"Calling tool {tool_name} via stdio ClientSession for {instance_uuid} (session already active in lifecycle task)"
                )
                tool_result = await cs.call_tool(tool_name, tool_args)

                if hasattr(tool_result, "model_dump"):
                    dumped = tool_result.model_dump(exclude_none=True)
                    if isinstance(dumped, dict):
                        return dumped
                    return {
                        "error": "Tool result model_dump was not a dict or not a Pydantic model",
                        "details": str(dumped),
                    }
                elif isinstance(tool_result, dict):
                    return tool_result
                else:
                    return {
                        "error": "Tool result unexpected format",
                        "details": str(tool_result),
                    }

            except Exception as e:
                logger.error(
                    f"MCP stdio tool call for {tool_name} on instance {instance_uuid} failed: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"MCP stdio tool call for {tool_name} failed: {e}") from e
        else:
            raise ValueError(f"Unsupported mcp_transport: {instance.mcp_transport}")

    async def list_tools_on_instance(self, instance: ManagedInstanceInfo) -> types.ListToolsResult:
        if instance.orchestration_mode != "local_docker":
            raise ValueError("LocalDockerOrchestrationClient can only list tools for 'local_docker' instances.")

        logger.info(
            f"Listing tools for local Docker instance {instance.instance_id} ({instance.backend_name_ref}) using {instance.mcp_transport} transport."
        )

        if instance.mcp_transport == "http":
            if not instance.mcp_endpoint_url:
                raise ValueError(
                    f"Instance {instance.instance_id} ({instance.backend_name_ref}) is HTTP but mcp_endpoint_url is missing."
                )
            target_base_url = instance.mcp_endpoint_url.rstrip("/")
            try:
                async with streamablehttp_client(base_url=target_base_url) as (  # type: ignore
                    read_s,
                    write_s,
                    _,  # get_session_id_func usually not needed for a single call
                ):
                    # Create a ClientSession with these streams
                    mcp_session_for_list_tools = ClientSession(
                        read_stream=read_s,
                        write_stream=write_s,
                        client_info=DEFAULT_CLIENT_INFO,  # Added default client info
                    )
                    # Initialize the session (MCP handshake)
                    await mcp_session_for_list_tools.initialize()
                    list_tools_result = await mcp_session_for_list_tools.list_tools()
                    # ClientSession does not need to be explicitly closed here if not used further,
                    # as the underlying streams from streamablehttp_client will be closed by its context manager.
                    logger.info(
                        f"Successfully listed {len(list_tools_result.tools)} tools from {target_base_url} for HTTP instance {instance.instance_id}"
                    )
                    return list_tools_result
            except Exception as e:
                logger.error(
                    f"Error listing tools from HTTP instance {instance.instance_id} at {target_base_url}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Failed to list tools from HTTP Docker instance {instance.instance_id}") from e

        elif instance.mcp_transport == "stdio":
            instance_uuid = instance.internal_instance_details.get("instance_uuid", instance.instance_id)
            cs = self._stdio_client_sessions.get(instance_uuid)

            if not cs or not isinstance(cs, ClientSession):
                # This could happen if the instance is still initializing or failed to initialize.
                # For simplicity, we raise. A more robust solution might wait or check task status.
                logger.error(
                    f"ClientSession not found or invalid for stdio instance {instance_uuid}. It might be initializing or failed."
                )
                raise RuntimeError(
                    f"Valid ClientSession not found for stdio instance {instance_uuid}. Cannot list tools."
                )

            try:
                logger.debug(f"Listing tools via existing stdio ClientSession for {instance_uuid}")
                list_tools_result = await cs.list_tools()
                logger.info(
                    f"Successfully listed {len(list_tools_result.tools)} tools for stdio instance {instance_uuid}"
                )
                return list_tools_result
            except Exception as e:
                logger.error(
                    f"Error listing tools from stdio instance {instance_uuid}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Failed to list tools from stdio Docker instance {instance_uuid}") from e
        else:
            raise ValueError(f"Unsupported mcp_transport for local_docker: {instance.mcp_transport}")


import os
