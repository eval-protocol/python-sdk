import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from eval_protocol.pytest.default_agent_rollout_processor import Agent
from klavis import Klavis
from klavis.types import CreateSandboxResponse, SandboxMcpServer
from openai.types import CompletionUsage

logger = logging.getLogger(__name__)


class KlavisSandboxRolloutProcessor(RolloutProcessor):
    def __init__(
        self,
        server_name: str,
        initialize_data_factory: Optional[Callable[[EvaluationRow], Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.server_name = server_name
        self.initialize_data_factory = initialize_data_factory
        self.klavis_client = Klavis(api_key=os.environ.get("KLAVIS_API_KEY"))
        
    def _init_sandbox(self) -> CreateSandboxResponse:
        try:
            server_name_enum = SandboxMcpServer(self.server_name)
            return self.klavis_client.sandbox.create_sandbox(server_name=server_name_enum)
        except Exception as e:
            logger.error(f"Error creating sandbox: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def create_mcp_config(server_url: str, server_key: str = "main", auth_token: str | None = None) -> str:
        """Create a temporary MCP config file and return its path."""
        config = {
            "mcpServers": {
                server_key: {
                    "url": server_url,
                    "transport": "streamable_http",
                    **({"authorization": f"Bearer {auth_token}"} if auth_token else {})
                }
            }
        }
        
        # Create a temp file that persists for the session
        fd, path = tempfile.mkstemp(suffix=".json", prefix="mcp_config_")
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f)
        return path

    def __call__(
        self, rows: List[EvaluationRow], config: RolloutProcessorConfig
    ) -> List[asyncio.Task[EvaluationRow]]:
        """Process evaluation rows with Klavis sandbox lifecycle management"""
        semaphore = config.semaphore

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row with complete sandbox lifecycle"""
            
            start_time = time.perf_counter()
            agent: Agent | None = None
            temp_config_path: str | None = None
            sandbox: CreateSandboxResponse | None = None

            try:
                # Step 0: Create a sandbox for this row
                sandbox = self._init_sandbox()
                logger.info(f"Sandbox created: {sandbox}")

                # Step 1: Initialize data in the sandbox
                init_data: Dict[str, Any] | None = None
                if self.initialize_data_factory:
                    init_data = self.initialize_data_factory(row)
                else:
                    # Allow datasets to provide initialization payload directly
                    init_data = (
                        (row.input_metadata.session_data or {}).get("initialize_data")
                        if row.input_metadata is not None
                        else None
                    )
                
                if init_data:
                    logger.info(f"Initializing {self.server_name} sandbox {sandbox.sandbox_id}")
                    initialize_method = getattr(
                        self.klavis_client.sandbox, f"initialize_{sandbox.server_name.value}_sandbox"
                    )
                    init_response = initialize_method(sandbox_id=sandbox.sandbox_id, **init_data)
                    logger.info(f"Initialization response: {init_response}")
                    
                # Step 2: Create temporary MCP config with sandbox URL
                temp_config_path = self.create_mcp_config(
                    server_url=sandbox.server_url, server_key=sandbox.server_name.value
                )
                logger.info(f"MCP config created: {temp_config_path}")

                # Step 3: Run agent with sandbox MCP server
                logger.info(f"Running agent for row {row.execution_metadata.rollout_id} with {self.server_name} sandbox")
                agent = Agent(
                    model=row.input_metadata.completion_params["model"],
                    row=row,
                    config_path=temp_config_path,
                    logger=config.logger,
                )
                await agent.setup()
                await agent.call_agent()

                # Update usage metadata
                row.execution_metadata.usage = CompletionUsage(
                    prompt_tokens=agent.usage.get("prompt_tokens", 0),
                    completion_tokens=agent.usage.get("completion_tokens", 0),
                    total_tokens=agent.usage.get("total_tokens", 0),
                )
                row = agent.evaluation_row
                logger.info(f"Agent execution completed for row {row.execution_metadata.rollout_id}")

                # Step 4: Export sandbox data
                dump_method = getattr(self.klavis_client.sandbox, f"dump_{sandbox.server_name.value}_sandbox")
                dump_response = dump_method(sandbox_id=sandbox.sandbox_id)
                sandbox_data = dump_response.data
                logger.info(f"Sandbox data: {sandbox_data}")

                # Store sandbox data in row metadata for evaluation
                if not row.execution_metadata.extra:
                    row.execution_metadata.extra = {}
                row.execution_metadata.extra["sandbox_data"] = sandbox_data
                row.execution_metadata.extra["sandbox_id"] = sandbox.sandbox_id
                row.execution_metadata.extra["server_name"] = self.server_name

            except Exception as e:
                logger.error(f"Error processing row {row.execution_metadata.rollout_id}: {str(e)}", exc_info=True)
                if not row.execution_metadata.extra:
                    row.execution_metadata.extra = {}
                row.execution_metadata.extra["error"] = str(e)
                raise

            finally:
                # Cleanup agent MCP client and temp config
                if agent and agent.mcp_client:
                    await agent.mcp_client.cleanup()
                if temp_config_path and os.path.exists(temp_config_path):
                    os.unlink(temp_config_path)
                
                # Release sandbox
                if sandbox and sandbox.sandbox_id:
                    try:
                        self.klavis_client.sandbox.delete_sandbox(
                            server_name=sandbox.server_name, sandbox_id=sandbox.sandbox_id
                        )
                        logger.info(f"Sandbox {sandbox.sandbox_id} released successfully")
                    except Exception as e:
                        logger.error(f"Error releasing sandbox {sandbox.sandbox_id}: {str(e)}", exc_info=True)

                row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time

            return row

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await process_row(r)
                return result

        # Create and return tasks
        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks
