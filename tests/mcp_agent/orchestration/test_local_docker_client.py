import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock, patch

import docker  # For type hinting if needed, though we mock it
import httpx
import pytest
import pytest_asyncio
from mcp import types as mcp_types  # Tool is defined here

# from mcp.shared import message_types as mcp_message_types # No longer needed for ToolDefinition
from mcp.client.session import ClientSession  # Import ClientSession

from eval_protocol.mcp_agent.config import AppConfig, BackendServerConfig
from eval_protocol.mcp_agent.orchestration.base_client import ManagedInstanceInfo
from eval_protocol.mcp_agent.orchestration.local_docker_client import (
    LocalDockerOrchestrationClient,
)

logger = logging.getLogger(__name__)

# Skip all tests in this module if the Docker CLI is not available
pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="Docker CLI not available",
)

# IMPORTANT: Before running tests in this file that use MOCK_HTTP_BACKEND_CONFIG,
# ensure you have built the mock Docker image locally:
# From the repository root, run:
# docker build -t mock-mcp-server:latest tests/mcp_agent/mock_mcp_server_image/

# Define a mock backend config for an HTTP server (using the mock image)
MOCK_HTTP_BACKEND_CONFIG = BackendServerConfig(
    backend_name_ref="mock_http_server_test",
    backend_type="everything",  # Generic type for a mock server
    orchestration_mode="local_docker",
    instance_scoping="session",
    mcp_transport="http",
    docker_image="mock-mcp-server:latest",  # Assumes image built as 'mock-mcp-server:latest'
    container_port=8080,  # Default port for the mock_server.py
    startup_check_mcp_tool={"tool_name": "ping", "arguments": {}},
)

# Define a mock backend config for an stdio server (e.g., mcp/filesystem)
# This can be used for stdio specific tests, complementing test_rl_filesystem_scenario.py
MOCK_STDIO_FILESYSTEM_CONFIG = BackendServerConfig(
    backend_name_ref="test_stdio_fs_direct",
    backend_type="filesystem",
    orchestration_mode="local_docker",
    instance_scoping="session",
    mcp_transport="stdio",
    docker_image="mcp/filesystem",
    container_command=["/data"],  # Command for the filesystem server
    # For stdio, template_data_path_host would be set up per test if needed
    # startup_check_mcp_tool can also be a ping for stdio if the server supports it via stdio
    startup_check_mcp_tool={"tool_name": "list_tools", "arguments": {}},
)


@pytest_asyncio.fixture(scope="function")
async def local_docker_orchestrator() -> AsyncGenerator[LocalDockerOrchestrationClient, None]:
    """Fixture to provide an initialized LocalDockerOrchestrationClient."""
    # A minimal AppConfig is needed by the orchestrator
    app_cfg = AppConfig(
        backends=[MOCK_HTTP_BACKEND_CONFIG, MOCK_STDIO_FILESYSTEM_CONFIG],
        log_level="DEBUG",  # Use DEBUG for orchestrator tests
    )
    orchestrator = LocalDockerOrchestrationClient(app_config=app_cfg)
    # Patch docker.from_env for the duration of this fixture
    with patch("eval_protocol.mcp_agent.orchestration.local_docker_client.docker.from_env") as mock_docker_from_env:
        mock_docker_client = MagicMock(spec=docker.DockerClient)
        mock_docker_client.ping = MagicMock(return_value=True)

        # Mock containers attribute and its methods
        mock_containers = MagicMock()
        mock_docker_client.containers = mock_containers

        # Mock images attribute and its methods
        mock_images = MagicMock()
        mock_docker_client.images = mock_images
        mock_images.pull = MagicMock()  # To prevent actual pull
        mock_images.remove = MagicMock()

        mock_docker_from_env.return_value = mock_docker_client

        orchestrator = LocalDockerOrchestrationClient(app_config=app_cfg)
        # Startup will now use the mocked docker_client
        await orchestrator.startup()

        # Store the mock_docker_client on the orchestrator for tests to configure further if needed
        orchestrator.mock_docker_client = mock_docker_client  # type: ignore

        yield orchestrator

        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_provision_deprovision_http_instance(
    local_docker_orchestrator: LocalDockerOrchestrationClient,  # This now has a .mock_docker_client
):
    """
    Tests provisioning and deprovisioning of a single HTTP backend instance.
    Mocks Docker interactions.
    """
    session_id = f"test-session-{uuid.uuid4().hex[:6]}"
    num_instances = 1
    provisioned_instances: List[ManagedInstanceInfo] = []

    # Configure mocks on the injected mock_docker_client
    mock_docker_client = local_docker_orchestrator.mock_docker_client  # type: ignore

    mock_container = MagicMock(spec=docker.models.containers.Container)
    mock_container.id = "mock_container_id_http"
    mock_container.name = "mock_container_name_http"
    # Simulate port bindings from container.attrs
    mock_container.attrs = {
        "NetworkSettings": {
            "Ports": {f"{MOCK_HTTP_BACKEND_CONFIG.container_port}/tcp": [{"HostIp": "0.0.0.0", "HostPort": "12345"}]}
        }
    }
    mock_container.reload = MagicMock()
    mock_container.stop = MagicMock()
    mock_container.remove = MagicMock()
    mock_container.logs = MagicMock(return_value=b"Mock logs")

    mock_docker_client.containers.run = MagicMock(return_value=mock_container)
    mock_docker_client.containers.get = MagicMock(return_value=mock_container)

    # Mock httpx.AsyncClient for startup check and tool calls if LocalDockerOrchestrationClient uses it directly
    # For HTTP transport, LocalDockerOrchestrationClient makes HTTP calls to the container.
    # These need to be mocked if we are not running a real container.
    # The startup_check and call_tool_on_instance for HTTP transport use orchestrator.http_client

    mock_http_post_response = MagicMock(spec=httpx.Response)
    mock_http_post_response.status_code = 200
    mock_http_post_response.json = MagicMock(
        return_value={"status": "pong", "server_id": "mock_http_server_id_from_test"}
    )  # For ping

    # Patch the http_client within the orchestrator instance for this test
    with patch.object(
        local_docker_orchestrator, "http_client", new_callable=AsyncMock
    ) as mock_orchestrator_http_client:
        mock_orchestrator_http_client.post = AsyncMock(return_value=mock_http_post_response)
        # If list_tools for HTTP also uses this client (it should use mcp.client.streamable_http), that needs separate mocking or handling.
        # For now, list_tools_on_instance for HTTP uses streamablehttp_client, which is harder to mock here directly.
        # Let's assume list_tools for HTTP will be tested in an integration manner or its http calls mocked differently.
        # We can mock `local_docker_orchestrator.list_tools_on_instance` directly if it becomes too complex.

        # Mock streamablehttp_client used by list_tools_on_instance for HTTP
        # This function is decorated with @asynccontextmanager, so it returns an ACM when called.
        # We patch the function itself.
        with patch(
            "eval_protocol.mcp_agent.orchestration.local_docker_client.streamablehttp_client",
            new_callable=MagicMock,
        ) as mock_streamablehttp_client_func:

            # Configure the mock Async Context Manager (ACM) that mock_streamablehttp_client_func will return
            mock_acm_instance = AsyncMock()  # This object needs __aenter__ and __aexit__

            # What __aenter__ of the ACM should yield: a tuple of (read_stream, write_stream, get_id_func)
            mock_read_stream = AsyncMock()  # spec=ObjectReceiveStream if imported from anyio.abc
            mock_write_stream = AsyncMock()  # spec=ObjectSendStream if imported from anyio.abc
            mock_get_id_func = MagicMock(return_value="mock_transport_session_id_for_list_tools")
            streams_tuple = (mock_read_stream, mock_write_stream, mock_get_id_func)
            mock_acm_instance.__aenter__.return_value = streams_tuple
            mock_acm_instance.__aexit__ = AsyncMock(return_value=None)  # Ensure __aexit__ is an awaitable mock

            mock_streamablehttp_client_func.return_value = mock_acm_instance

            # Now, mock the ClientSession that will be instantiated inside list_tools_on_instance
            # It will be called with the mocked streams.
            mock_mcp_list_tools_result = mcp_types.ListToolsResult(
                tools=[mcp_types.Tool(name="ping", description="pongs", inputSchema={})]
            )

            with patch(
                "eval_protocol.mcp_agent.orchestration.local_docker_client.ClientSession",
                new_callable=MagicMock,
            ) as MockedClientSessionInSUT:
                # This is the mock for the ClientSession *instance*
                mock_cs_instance = AsyncMock(spec=ClientSession)
                mock_cs_instance.initialize = AsyncMock()  # Mock the initialize call
                mock_cs_instance.list_tools = AsyncMock(return_value=mock_mcp_list_tools_result)

                MockedClientSessionInSUT.return_value = mock_cs_instance  # When ClientSession() is called in SUT

                try:
                    logger.info(f"Attempting to provision {num_instances} HTTP instance(s) for session {session_id}")
                    provisioned_instances = await local_docker_orchestrator.provision_instances(
                        backend_config=MOCK_HTTP_BACKEND_CONFIG,
                        num_instances=num_instances,
                        session_id=session_id,
                    )
                    assert len(provisioned_instances) == num_instances
                    instance_info = provisioned_instances[0]
                    assert instance_info.backend_name_ref == MOCK_HTTP_BACKEND_CONFIG.backend_name_ref
                    assert instance_info.mcp_transport == "http"
                    assert (
                        instance_info.mcp_endpoint_url == "http://localhost:12345/mcp"
                    )  # Based on mock_container.attrs
                    assert instance_info.internal_instance_details["container_id"] == "mock_container_id_http"
                    assert instance_info.internal_instance_details["host_port"] == 12345

                    logger.info(
                        f"Successfully provisioned HTTP instance: {instance_info.instance_id} on {instance_info.mcp_endpoint_url}"
                    )
                    mock_docker_client.containers.run.assert_called_once()
                    mock_container.reload.assert_called()  # provision_instances calls reload

                    # Test list_tools on the provisioned HTTP instance
                    logger.info(f"Attempting to list tools on HTTP instance: {instance_info.instance_id}")
                    list_tools_result = await local_docker_orchestrator.list_tools_on_instance(instance_info)
                    assert isinstance(list_tools_result, mcp_types.ListToolsResult)
                    assert len(list_tools_result.tools) > 0
                    assert list_tools_result.tools[0].name == "ping"
                    logger.info(
                        f"Successfully listed {len(list_tools_result.tools)} tools: {[t.name for t in list_tools_result.tools]}"
                    )
                    mock_streamablehttp_client_func.assert_called_with(base_url="http://localhost:12345/mcp")
                    MockedClientSessionInSUT.assert_called_once()  # Check ClientSession was instantiated
                    mock_cs_instance.initialize.assert_called_once()  # Check initialize was called
                    mock_cs_instance.list_tools.assert_called_once()  # Check list_tools was called

                    # Test a tool call (e.g., ping)
                    logger.info(f"Attempting to call 'ping' on HTTP instance: {instance_info.instance_id}")
                    ping_result = await local_docker_orchestrator.call_tool_on_instance(
                        instance=instance_info,
                        tool_name="ping",
                        tool_args={},
                    )
                    assert isinstance(ping_result, dict)
                    assert ping_result.get("status") == "pong"
                    # This server_id comes from the mock_http_post_response.json()
                    assert ping_result.get("server_id") == "mock_http_server_id_from_test"
                    logger.info(f"Successfully called 'ping', response: {ping_result}")
                    # Assert that the orchestrator's http_client.post was called for the tool call
                    mock_orchestrator_http_client.post.assert_any_call(
                        "http://localhost:12345/mcp",
                        json={"tool_name": "ping", "arguments": {}},
                    )

                finally:
                    if provisioned_instances:
                        logger.info(f"Attempting to deprovision {len(provisioned_instances)} HTTP instance(s)")
                        await local_docker_orchestrator.deprovision_instances(provisioned_instances)
                        logger.info("Deprovisioning complete.")
                        mock_container.stop.assert_called_once()
                        mock_container.remove.assert_called_once()


@pytest.mark.asyncio
async def test_provision_deprovision_stdio_instance(
    local_docker_orchestrator: LocalDockerOrchestrationClient,
    tmp_path: Path,  # local_docker_orchestrator now has .mock_docker_client
):
    """
    Tests provisioning and deprovisioning of a single stdio backend instance (mcp/filesystem).
    Includes a simple file operation to test templating and tool calls.
    """
    session_id = f"test-session-stdio-{uuid.uuid4().hex[:6]}"
    num_instances = 1
    provisioned_instances: List[ManagedInstanceInfo] = []

    # Create a temporary template directory for this test
    template_host_dir = tmp_path / "stdio_template_data"
    template_host_dir.mkdir()
    test_file_content = "Hello from stdio test template!"
    with open(template_host_dir / "test_file.txt", "w") as f:
        f.write(test_file_content)

    # Update a copy of the config to use this temp template path
    stdio_fs_config_with_template = MOCK_STDIO_FILESYSTEM_CONFIG.model_copy(deep=True)
    stdio_fs_config_with_template.template_data_path_host = str(template_host_dir)

    try:
        logger.info(
            f"Attempting to provision {num_instances} stdio instance(s) for session {session_id} using template {template_host_dir}"
        )
        provisioned_instances = await local_docker_orchestrator.provision_instances(
            backend_config=stdio_fs_config_with_template,
            num_instances=num_instances,
            session_id=session_id,
        )
        assert len(provisioned_instances) == num_instances
        instance_info = provisioned_instances[0]
        assert instance_info.backend_name_ref == stdio_fs_config_with_template.backend_name_ref
        assert instance_info.mcp_transport == "stdio"
        assert instance_info.mcp_endpoint_url is None  # Stdio instances don't have an HTTP endpoint URL
        assert "container_name" in instance_info.internal_instance_details  # Changed from container_id for stdio
        # For stdio, host_port is not applicable
        assert (
            "instance_host_data_path" in instance_info.internal_instance_details
        )  # Check if template copy path was stored

        logger.info(
            f"Successfully provisioned stdio instance: {instance_info.instance_id} (container: {instance_info.internal_instance_details.get('container_name')})"
        )

        # Test list_tools on the provisioned stdio instance
        logger.info(f"Attempting to list tools on stdio instance: {instance_info.instance_id}")
        list_tools_result = await local_docker_orchestrator.list_tools_on_instance(instance_info)
        assert isinstance(list_tools_result, mcp_types.ListToolsResult)
        assert len(list_tools_result.tools) > 0  # mcp/filesystem should have tools
        logger.info(
            f"Successfully listed {len(list_tools_result.tools)} tools for stdio: {[t.name for t in list_tools_result.tools]}"
        )

        # Test a tool call (e.g., read_file for the templated file)
        logger.info(
            f"Attempting to call 'read_file' for '/data/test_file.txt' on stdio instance: {instance_info.instance_id}"
        )
        read_file_result = await local_docker_orchestrator.call_tool_on_instance(
            instance=instance_info,
            tool_name="read_file",  # mcp/filesystem tool
            tool_args={"path": "/data/test_file.txt"},  # Path inside the container
        )
        assert isinstance(read_file_result, dict)
        # The content from mcp/filesystem read_file is nested
        read_content_list = read_file_result.get("content", [])
        assert len(read_content_list) == 1
        assert read_content_list[0].get("type") == "text"
        assert read_content_list[0].get("text") == test_file_content
        logger.info(f"Successfully called 'read_file' via stdio, content matches template.")

    finally:
        if provisioned_instances:
            logger.info(f"Attempting to deprovision {len(provisioned_instances)} stdio instance(s)")
            await local_docker_orchestrator.deprovision_instances(provisioned_instances)
            logger.info("Stdio deprovisioning complete.")
        # tmp_path fixture handles cleanup of template_host_dir


# TODO: Add more tests:
# - Test provisioning multiple HTTP instances.
# - Test provisioning multiple stdio instances.
# - Test error handling (e.g., Docker image not found, container fails to start, startup check fails for both http/stdio).
# - Test docker commit based templating (if feature is to be kept and tested here).
# - Test cleanup of temporary images from commit-based templating.
# - Test scenarios where template_data_path_host is not provided for filesystem.
