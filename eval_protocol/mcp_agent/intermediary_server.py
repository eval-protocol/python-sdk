import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

import anyio  # Added for debugging cancel scopes and tasks
from mcp import types as mcp_types  # Added for type hinting
from pydantic import BaseModel, Field

from eval_protocol.mcp_agent.config import AppConfig, BackendServerConfig
from eval_protocol.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)
from eval_protocol.mcp_agent.orchestration.local_docker_client import (
    LocalDockerOrchestrationClient,
)
from eval_protocol.mcp_agent.orchestration.remote_http_client import (
    RemoteHttpOrchestrationClient,
)
from eval_protocol.mcp_agent.session import IntermediarySessionData

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)  # Removed: Let level be set by main config

from mcp.server.fastmcp.server import Context as FastMCPContext
from mcp.server.fastmcp.server import FastMCP

# RequestContext is not directly used by handlers anymore, mcp_ctx is.


# Backend initialization models (moved here to avoid separate backends module)
class BackendInitRequest(BaseModel):
    backend_name_ref: str = Field(
        ...,
        description="The unique reference name of the backend configuration to use (must match one in AppConfig.backends).",
    )
    num_instances: int = Field(
        1,
        ge=1,
        description="Number of instances of this backend to provision for the session.",
    )
    template_details: Optional[Any] = Field(
        None,
        description="Backend-specific details for initializing stateful instances from a template.",
    )

    class Config:
        extra = "forbid"


class BackendInitResult(BaseModel):
    backend_name_ref: str
    instances: List[ManagedInstanceInfo]


# Pydantic models for tool arguments
class InitializeSessionArgs(BaseModel):
    backends: List[BackendInitRequest]


class CallBackendToolArgs(BaseModel):
    rk_session_id: str = Field(..., description="The session ID obtained from initialize_session.")
    backend_name_ref: str = Field(..., description="The reference name of the backend to target.")
    instance_id: str = Field(..., description="The ID of the specific backend instance to use.")
    tool_name: str = Field(..., description="The name of the tool to call on the backend instance.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the backend tool.")


class ListBackendToolsArgs(BaseModel):
    rk_session_id: str = Field(..., description="The session ID obtained from initialize_session.")
    backend_name_ref: str = Field(..., description="The reference name of the backend to target.")
    instance_id: str = Field(..., description="The ID of the specific backend instance to query for tools.")


class CleanupSessionArgs(BaseModel):
    rk_session_id: str = Field(..., description="The session ID to clean up.")


# Ping might not need specific args if it uses session from mcp_ctx, or could take rk_session_id
class PingArgs(BaseModel):
    rk_session_id: Optional[str] = Field(default=None, description="Optional session ID for context.")


class RewardKitIntermediaryServer(FastMCP):
    def __init__(self, app_config: AppConfig, **kwargs_for_fastmcp):
        super().__init__(
            name="RewardKitIntermediaryMCP",
            instructions="Intermediary Server for managing backend MCP resources for RewardKit RL rollouts.",
            **kwargs_for_fastmcp,
        )

        self.app_config = app_config
        self._local_docker_orchestrator: Optional[LocalDockerOrchestrationClient] = None
        self._remote_http_orchestrators: Dict[str, RemoteHttpOrchestrationClient] = {}
        self._shared_global_instances: Dict[str, ManagedInstanceInfo] = {}
        self._shared_instance_locks: Dict[str, asyncio.Lock] = {}
        self.intermediary_session_data: Dict[str, IntermediarySessionData] = {}

        logger.info("RewardKitIntermediaryServer (FastMCP based) initialized. AppConfig loaded.")

        # Register tools directly
        self.add_tool(self._initialize_session_actual, name="initialize_session")
        self.add_tool(self._call_backend_tool_actual, name="call_backend_tool")
        self.add_tool(self._list_backend_tools_actual, name="list_backend_tools")  # New tool
        self.add_tool(self._cleanup_session_actual, name="cleanup_session")
        self.add_tool(self._ping_actual, name="ping")

        logger.info("Registered tools directly with FastMCP.")

        # Explicitly set this module's logger level based on app_config
        # This is to ensure it overrides any prior default or hardcoded DEBUG level
        # if external configuration in main.py isn't fully effective.
        try:
            config_log_level_str = app_config.log_level.upper()
            config_log_level_int = getattr(logging, config_log_level_str, logging.INFO)
            if logger.getEffectiveLevel() != config_log_level_int:
                logger.info(
                    f"Overriding intermediary_server logger level from {logging.getLevelName(logger.getEffectiveLevel())} to {config_log_level_str}"
                )
                logger.setLevel(config_log_level_int)
                # Also ensure handlers attached directly to this logger respect it (if any)
                for handler in logger.handlers:
                    handler.setLevel(config_log_level_int)
            logger.info(
                f"IntermediaryServer logger effective level: {logging.getLevelName(logger.getEffectiveLevel())}"
            )

        except Exception as e_log:
            logger.error(f"Error trying to set intermediary_server logger level: {e_log}")

    # Removed _execute_proxied_tool_impl and _internal_tool_handlers

    async def _initialize_orchestrators(self):
        logger.info("Initializing orchestration clients...")
        if any(b.orchestration_mode == "local_docker" for b in self.app_config.backends):
            self._local_docker_orchestrator = LocalDockerOrchestrationClient(self.app_config)
            await self._local_docker_orchestrator.startup()
            logger.info("LocalDockerOrchestrationClient initialized and started.")

        unique_remote_api_refs = set()
        for backend_cfg in self.app_config.backends:
            if backend_cfg.orchestration_mode == "remote_http_api":
                if backend_cfg.remote_api_config_ref:
                    unique_remote_api_refs.add(backend_cfg.remote_api_config_ref)
                elif backend_cfg.remote_api_config_inline:
                    logger.warning(
                        f"Inline remote_api_config for {backend_cfg.backend_name_ref}. Consider using global_remote_apis."
                    )
                    key = backend_cfg.remote_api_config_inline.base_url
                    if key not in self._remote_http_orchestrators:
                        temp_app_config_for_inline = AppConfig(
                            global_remote_apis={key: backend_cfg.remote_api_config_inline}
                        )
                        client = RemoteHttpOrchestrationClient(temp_app_config_for_inline)
                        await client.startup()
                        self._remote_http_orchestrators[key] = client
                        logger.info(f"RemoteHttpOrchestrationClient for inline config {key} initialized.")

        for ref_name in unique_remote_api_refs:
            if ref_name not in self.app_config.global_remote_apis:
                logger.error(f"Remote API ref '{ref_name}' not in global_remote_apis.")
                continue
            if ref_name not in self._remote_http_orchestrators:
                isolated_app_cfg = AppConfig(
                    global_remote_apis={ref_name: self.app_config.global_remote_apis[ref_name]},
                    global_remote_api_defaults=self.app_config.global_remote_api_defaults,
                )
                client = RemoteHttpOrchestrationClient(isolated_app_cfg)
                await client.startup()
                self._remote_http_orchestrators[ref_name] = client
                logger.info(f"RemoteHttpOrchestrationClient for '{ref_name}' initialized.")
        logger.info("Orchestration clients initialization complete.")

    def _get_orchestration_client(self, backend_cfg: BackendServerConfig) -> AbstractOrchestrationClient:
        if backend_cfg.orchestration_mode == "local_docker":
            if not self._local_docker_orchestrator:
                raise RuntimeError("Local Docker orchestrator not initialized.")
            return self._local_docker_orchestrator
        elif backend_cfg.orchestration_mode == "remote_http_api":
            key = backend_cfg.remote_api_config_ref
            if not key:
                if backend_cfg.remote_api_config_inline:
                    key = backend_cfg.remote_api_config_inline.base_url
                else:
                    raise ValueError(f"Remote API config missing for {backend_cfg.backend_name_ref}")
            client = self._remote_http_orchestrators.get(key)
            if not client:
                raise RuntimeError(f"Remote HTTP orchestrator for '{key}' not initialized.")
            return client
        else:
            raise ValueError(f"Unsupported orchestration mode: {backend_cfg.orchestration_mode}")

    async def _get_or_provision_shared_global_instance(self, backend_name_ref: str) -> ManagedInstanceInfo:
        if backend_name_ref not in self._shared_instance_locks:
            self._shared_instance_locks[backend_name_ref] = asyncio.Lock()
        async with self._shared_instance_locks[backend_name_ref]:
            if backend_name_ref in self._shared_global_instances:
                logger.info(f"Returning existing shared global instance for '{backend_name_ref}'.")
                return self._shared_global_instances[backend_name_ref]
            logger.info(f"Provisioning new shared global instance for '{backend_name_ref}'.")
            backend_cfg = next(
                (b for b in self.app_config.backends if b.backend_name_ref == backend_name_ref),
                None,
            )
            if not backend_cfg or backend_cfg.instance_scoping != "shared_global":
                raise ValueError(f"Backend '{backend_name_ref}' not for shared_global scoping.")
            orchestration_client = self._get_orchestration_client(backend_cfg)
            provisioned_list = await orchestration_client.provision_instances(
                backend_config=backend_cfg,
                num_instances=1,
                session_id="global_shared_session",
                template_details=backend_cfg.template_data_path_host,
            )
            if not provisioned_list:
                raise RuntimeError(f"Failed to provision shared global for '{backend_name_ref}'.")
            instance_info = provisioned_list[0]
            self._shared_global_instances[backend_name_ref] = instance_info
            logger.info(f"Provisioned shared global for '{backend_name_ref}': {instance_info.instance_id}")
            return instance_info

    async def _provision_shared_global_instances(self):
        logger.info("Pre-provisioning all shared_global instances...")
        for backend_cfg in self.app_config.backends:
            if backend_cfg.instance_scoping == "shared_global":
                try:
                    await self._get_or_provision_shared_global_instance(backend_cfg.backend_name_ref)
                except Exception as e:
                    logger.error(
                        f"Failed to pre-provision for '{backend_cfg.backend_name_ref}': {e}",
                        exc_info=True,
                    )
        logger.info("Shared_global instances pre-provisioning complete.")

    async def _initialize_session_actual(self, mcp_ctx: FastMCPContext, args: InitializeSessionArgs) -> Dict[str, Any]:
        task_name = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
        logger.debug(
            f"ENTERING _initialize_session_actual: task='{task_name}', mcp_ctx type: {type(mcp_ctx)}, args: {args}"
        )

        transport_session_id: Optional[str] = None
        if (
            hasattr(mcp_ctx, "session")
            and mcp_ctx.session
            and hasattr(mcp_ctx.session, "client_params")
            and mcp_ctx.session.client_params
            and hasattr(mcp_ctx.session.client_params, "session_id")
            and mcp_ctx.session.client_params.session_id
        ):
            transport_session_id = mcp_ctx.session.client_params.session_id
            logger.info(f"Retrieved transport_session_id: {transport_session_id}")

        rk_session_id = transport_session_id if transport_session_id else uuid.uuid4().hex
        if not transport_session_id:
            logger.warning(f"Transport session ID not found. Generated new rk_session_id: {rk_session_id}")
        else:
            logger.info(f"Using transport_session_id as rk_session_id: {rk_session_id}")

        if rk_session_id in self.intermediary_session_data:
            logger.warning(f"rk_session_id '{rk_session_id}' already exists. Overwriting.")
        session_data = IntermediarySessionData(session_id=rk_session_id)
        self.intermediary_session_data[rk_session_id] = session_data

        logger.info(
            f"Initializing IntermediarySessionData for rk_session_id '{rk_session_id}' with {len(args.backends)} backend requests."
        )
        initialized_backends_results: List[BackendInitResult] = []

        for backend_req in args.backends:
            backend_cfg = next(
                (b for b in self.app_config.backends if b.backend_name_ref == backend_req.backend_name_ref),
                None,
            )
            if not backend_cfg:
                logger.error(f"Session {rk_session_id}: Config for '{backend_req.backend_name_ref}' not found.")
                initialized_backends_results.append(
                    BackendInitResult(backend_name_ref=backend_req.backend_name_ref, instances=[])
                )
                continue
            try:
                if backend_cfg.instance_scoping == "shared_global":
                    shared_instance_info = await self._get_or_provision_shared_global_instance(
                        backend_req.backend_name_ref
                    )
                    instances_for_this_backend = [shared_instance_info] * backend_req.num_instances
                else:
                    orchestration_client = self._get_orchestration_client(backend_cfg)
                    instances_for_this_backend = await orchestration_client.provision_instances(
                        backend_config=backend_cfg,
                        num_instances=backend_req.num_instances,
                        session_id=session_data.session_id,
                        template_details=backend_req.template_details,
                    )
                session_data.add_managed_instances(backend_req.backend_name_ref, instances_for_this_backend)
                initialized_backends_results.append(
                    BackendInitResult(
                        backend_name_ref=backend_req.backend_name_ref,
                        instances=instances_for_this_backend,
                    )
                )
            except Exception as e:
                logger.error(
                    f"Session {rk_session_id}: Error initializing '{backend_req.backend_name_ref}': {e}",
                    exc_info=True,
                )
                initialized_backends_results.append(
                    BackendInitResult(
                        backend_name_ref=backend_req.backend_name_ref,
                        instances=[],
                        error_message=str(e),
                    )
                )

        task_name_exit = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
        logger.debug(f"EXITING _initialize_session_actual: task='{task_name_exit}'")
        return {
            "rk_session_id": rk_session_id,
            "initialized_backends": [res.model_dump(exclude_none=True) for res in initialized_backends_results],
        }

    async def _call_backend_tool_actual(self, mcp_ctx: FastMCPContext, args: CallBackendToolArgs) -> Dict[str, Any]:
        task_name_entry = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
        logger.debug(
            f"ENTERING _call_backend_tool_actual: task='{task_name_entry}', mcp_ctx type: {type(mcp_ctx)}, args: {args}"
        )

        session_data = self.intermediary_session_data.get(args.rk_session_id)
        if not session_data:
            task_name_error = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
            logger.error(
                f"ERROR in _call_backend_tool_actual (session not found): task='{task_name_error}', rk_session_id='{args.rk_session_id}'"
            )
            raise ValueError(f"IntermediarySessionData for rk_session_id '{args.rk_session_id}' not found.")

        target_instances = session_data.get_managed_instances(args.backend_name_ref, args.instance_id)
        if not target_instances:
            raise ValueError(
                f"Instance '{args.instance_id}' for backend '{args.backend_name_ref}' not found in session '{args.rk_session_id}'."
            )
        managed_instance_info = target_instances[0]
        backend_cfg = next(
            (b for b in self.app_config.backends if b.backend_name_ref == args.backend_name_ref),
            None,
        )
        if not backend_cfg:
            raise ValueError(f"Backend config '{args.backend_name_ref}' not found.")
        orchestration_client = self._get_orchestration_client(backend_cfg)

        task_name_before_call = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
        logger.debug(
            f"BEFORE orchestrator.call_tool_on_instance in _call_backend_tool_actual: task='{task_name_before_call}'"
        )

        try:
            result = await orchestration_client.call_tool_on_instance(
                instance=managed_instance_info,
                tool_name=args.tool_name,
                tool_args=args.tool_args,
            )
            task_name_after_call = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
            logger.debug(
                f"AFTER orchestrator.call_tool_on_instance in _call_backend_tool_actual: task='{task_name_after_call}'"
            )

            task_name_exit = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
            logger.debug(f"EXITING _call_backend_tool_actual (SUCCESS): task='{task_name_exit}'")
            return result
        except Exception as e:
            task_name_exception = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
            logger.error(
                f"EXCEPTION in _call_backend_tool_actual: task='{task_name_exception}'. Session {args.rk_session_id}: Error calling tool '{args.tool_name}' on instance '{args.instance_id}': {e}",
                exc_info=True,
            )
            raise

    async def _list_backend_tools_actual(
        self, mcp_ctx: FastMCPContext, args: ListBackendToolsArgs
    ) -> Dict[str, Any]:  # Returning dict for FastMCP, will be ListToolsResult internally
        task_name_entry = anyio.get_current_task().name if anyio.get_current_task() else "unknown_task"
        logger.debug(f"ENTERING _list_backend_tools_actual: task='{task_name_entry}', args: {args}")

        session_data = self.intermediary_session_data.get(args.rk_session_id)
        if not session_data:
            logger.error(
                f"ERROR in _list_backend_tools_actual (session not found): rk_session_id='{args.rk_session_id}'"
            )
            raise ValueError(f"IntermediarySessionData for rk_session_id '{args.rk_session_id}' not found.")

        target_instances = session_data.get_managed_instances(args.backend_name_ref, args.instance_id)
        if not target_instances:
            raise ValueError(
                f"Instance '{args.instance_id}' for backend '{args.backend_name_ref}' not found in session '{args.rk_session_id}'."
            )
        managed_instance_info = target_instances[0]

        backend_cfg = next(
            (b for b in self.app_config.backends if b.backend_name_ref == args.backend_name_ref),
            None,
        )
        if not backend_cfg:
            raise ValueError(f"Backend config '{args.backend_name_ref}' not found.")
        orchestration_client = self._get_orchestration_client(backend_cfg)

        logger.debug(
            f"Calling orchestrator.list_tools_on_instance for backend '{args.backend_name_ref}', instance '{args.instance_id}'"
        )
        try:
            list_tools_result: mcp_types.ListToolsResult = await orchestration_client.list_tools_on_instance(
                instance=managed_instance_info
            )
            # FastMCP tools expect to return a dictionary that can be JSON serialized.
            # ListToolsResult is a Pydantic model, so model_dump() is appropriate.
            return list_tools_result.model_dump(exclude_none=True)
        except Exception as e:
            logger.error(
                f"EXCEPTION in _list_backend_tools_actual for session {args.rk_session_id}, backend {args.backend_name_ref}, instance {args.instance_id}: {e}",
                exc_info=True,
            )
            raise  # Re-raise to let FastMCP handle error reporting to client

    async def cleanup_session_internal(self, session_data_to_clean: IntermediarySessionData, rk_session_id: str):
        logger.info(f"Starting internal cleanup for IntermediarySessionData (rk_session_id: '{rk_session_id}').")
        all_session_instances = session_data_to_clean.get_all_managed_instances()
        local_docker_instances = [inst for inst in all_session_instances if inst.orchestration_mode == "local_docker"]
        if local_docker_instances and self._local_docker_orchestrator:
            try:
                await self._local_docker_orchestrator.deprovision_instances(local_docker_instances)
            except Exception as e:
                logger.error(
                    f"Session {rk_session_id}: Error deprovisioning local Docker: {e}",
                    exc_info=True,
                )

        remote_instances_by_key: Dict[str, List[ManagedInstanceInfo]] = {}
        for inst in all_session_instances:
            if inst.orchestration_mode == "remote_http_api":
                key = self._get_orchestration_client_key_for_instance(inst)
                if key:
                    remote_instances_by_key.setdefault(key, []).append(inst)
        for key, remote_list in remote_instances_by_key.items():
            orchestrator = self._remote_http_orchestrators.get(key)
            if orchestrator and remote_list:
                try:
                    await orchestrator.deprovision_instances(remote_list)
                except Exception as e:
                    logger.error(
                        f"Session {rk_session_id}: Error deprovisioning remote for '{key}': {e}",
                        exc_info=True,
                    )
        logger.info(f"Internal cleanup for session data (rk_session_id: '{rk_session_id}') complete.")

    async def _cleanup_session_actual(self, mcp_ctx: FastMCPContext, args: CleanupSessionArgs) -> Dict[str, str]:
        logger.debug(f"_cleanup_session_actual called. mcp_ctx type: {type(mcp_ctx)}, args: {args}")
        session_data_obj = self.intermediary_session_data.pop(args.rk_session_id, None)
        if not session_data_obj:
            logger.warning(
                f"IntermediarySessionData for rk_session_id '{args.rk_session_id}' not found or already cleaned."
            )
            return {
                "status": "custom_session_data_not_found_or_already_cleaned",
                "rk_session_id": args.rk_session_id,
            }
        await self.cleanup_session_internal(session_data_obj, args.rk_session_id)
        logger.info(f"IntermediarySessionData for rk_session_id '{args.rk_session_id}' fully cleaned up.")
        return {"status": "cleaned", "rk_session_id": args.rk_session_id}

    async def startup(self):
        logger.info("RewardKitIntermediaryServer performing custom startup tasks...")
        try:
            await self._initialize_orchestrators()
            await self._provision_shared_global_instances()
            logger.info("RewardKitIntermediaryServer custom startup tasks complete.")
        except Exception as e:
            logger.error(
                f"Error during RewardKitIntermediaryServer custom startup: {e}",
                exc_info=True,
            )
            raise

    async def _ping_actual(self, mcp_ctx: FastMCPContext, args: PingArgs) -> Dict[str, str]:
        logger.debug(f"_ping_actual called. mcp_ctx type: {type(mcp_ctx)}, args: {args}")
        ping_session_id: Optional[str] = None
        if args.rk_session_id:  # If client provides its known rk_session_id
            ping_session_id = args.rk_session_id
            logger.info(f"Ping using rk_session_id from args: {ping_session_id}")
        elif (
            hasattr(mcp_ctx, "session")
            and mcp_ctx.session
            and hasattr(mcp_ctx.session, "client_params")
            and mcp_ctx.session.client_params
            and hasattr(mcp_ctx.session.client_params, "session_id")
            and mcp_ctx.session.client_params.session_id
        ):
            ping_session_id = mcp_ctx.session.client_params.session_id
            logger.info(f"Ping using transport_session_id from mcp_ctx: {ping_session_id}")
        else:
            ping_session_id = "unknown_session_for_ping"
            logger.warning(f"Session ID for ping not found in args or mcp_ctx, using fallback: {ping_session_id}")
        return {"reply": "pong", "session_id": ping_session_id or ""}

    async def shutdown(self):
        logger.info("RewardKitIntermediaryServer (FastMCP based) performing custom shutdown tasks...")
        logger.info(f"Cleaning up {len(self.intermediary_session_data)} IntermediarySessionData entries...")
        for session_id_key in list(self.intermediary_session_data.keys()):
            session_data_obj = self.intermediary_session_data.pop(session_id_key, None)
            if session_data_obj:
                await self.cleanup_session_internal(session_data_obj, session_id_key)

        shared_instances = list(self._shared_global_instances.values())
        if shared_instances:
            logger.info(f"Deprovisioning {len(shared_instances)} shared global instances.")
            local_shared = [i for i in shared_instances if i.orchestration_mode == "local_docker"]
            if local_shared and self._local_docker_orchestrator:
                await self._local_docker_orchestrator.deprovision_instances(local_shared)
            remote_shared_by_key: Dict[str, List[ManagedInstanceInfo]] = {}
            for inst_info in shared_instances:
                if inst_info.orchestration_mode == "remote_http_api":
                    key = self._get_orchestration_client_key_for_instance(inst_info)
                    if key:
                        remote_shared_by_key.setdefault(key, []).append(inst_info)
            for key, instances_list in remote_shared_by_key.items():
                orchestrator = self._remote_http_orchestrators.get(key)
                if orchestrator:
                    await orchestrator.deprovision_instances(instances_list)

        if self._local_docker_orchestrator:
            await self._local_docker_orchestrator.shutdown()
        for orch in self._remote_http_orchestrators.values():
            await orch.shutdown()
        logger.info("RewardKitIntermediaryServer custom shutdown tasks complete.")

    def _get_orchestration_client_key_for_instance(self, instance_info: ManagedInstanceInfo) -> Optional[str]:
        if instance_info.orchestration_mode == "remote_http_api":
            backend_cfg = next(
                (b for b in self.app_config.backends if b.backend_name_ref == instance_info.backend_name_ref),
                None,
            )
            if backend_cfg:
                return backend_cfg.remote_api_config_ref or (
                    backend_cfg.remote_api_config_inline.base_url if backend_cfg.remote_api_config_inline else None
                )
        return None
