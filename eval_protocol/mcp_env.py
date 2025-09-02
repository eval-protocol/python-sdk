"""
MCP Environment API for Eval Protocol - Backward Compatibility Facade

This module has been refactored into modular components for better maintainability.
This file now serves as a backward compatibility facade.

New modular structure:
- mcp.client.connection: MCP client connection management
- mcp.execution.policy: LLMBasePolicy and FireworksPolicy for tool calling
- mcp.execution.rollout: Rollout coordination and lifecycle
- mcp.session.manager: Session and environment management

Usage remains the same:
    import eval_protocol as ep

    # Create general policy (environment-agnostic)
    policy = ep.FireworksPolicy(model_id="accounts/fireworks/models/qwen3-235b-a22b")

    # Create environments with evaluation_rows configuration
    envs = ep.make("http://localhost:8000/mcp", evaluation_rows=evaluation_rows)

    # Execute tool-calling rollouts
    evaluation_rows = await ep.rollout(envs, policy=policy, steps=512)

Key Features:
- General tool-calling interface that works with any MCP environment
- EvaluationRow-driven configuration with system prompts and user prompt templates
- Automatic MCP tool discovery from servers
- **PROPER MCP PATTERN**: Initial state obtained from MCP resources during session establishment
- Tools used only for actions/interactions, not for getting initial state
- Dynamic user prompt formatting based on current observations
- Environment-agnostic policy that receives tool schemas and makes structured calls
- Backward compatibility with servers that don't expose resources
- **NEW**: LLMBasePolicy abstraction enables easy OpenAI integration

MCP Integration:
- Session establishment creates MCP connection and discovers resources and tools
- Initial state comes from MCP resources (list_resources + read_resource calls)
- Tools are used for subsequent actions during rollout steps
- Resources provide static/configuration data, tools provide dynamic actions
"""

import asyncio
import hashlib
import json

# For legacy compatibility - import the facade functions
import logging
import random
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

# Import all functionality from the new modular components
from .mcp.execution.manager import ExecutionManager
from .mcp.execution.policy import AnthropicPolicy, FireworksPolicy, LiteLLMPolicy, LLMBasePolicy, OpenAIPolicy
from .mcp.session.manager import GeneralMCPVectorEnv
from .models import EvaluationRow
from .types import DatasetRow, MCPSession, MCPToolCall

logger = logging.getLogger(__name__)


def gen_session_id(dataset_row: DatasetRow, model_id: str) -> str:
    """
    Generate a session ID for a dataset row
    """
    seed_value = dataset_row.seed
    config_value = dataset_row.environment_context
    dataset_row_id_value = dataset_row.id
    model_id_value = model_id

    stable_data = {
        "seed": seed_value,
        "config": config_value,
        "dataset_row_id": dataset_row_id_value,
        "model_id": model_id_value,
    }

    stable_str = json.dumps(stable_data, sort_keys=True)

    return hashlib.md5(stable_str.encode()).hexdigest()


async def reset_mcp_sessions(envs: GeneralMCPVectorEnv):
    """
    Reset mcp server sessions
    """
    tasks = [envs.connection_manager.reset_session(session) for session in envs.sessions]
    await asyncio.gather(*tasks, return_exceptions=True)


def make(
    env_spec: str,
    evaluation_rows: Optional[List[EvaluationRow]] = None,
    dataset: Optional[List[Dict]] = None,
    n: Optional[int] = None,
    seeds: Optional[List[int]] = None,
    model_id: str = "unknown",
    user_prompt_formatter: Optional[Callable] = None,
) -> GeneralMCPVectorEnv:
    """
    Create general MCP environments driven by evaluation_rows configuration.

    Args:
        env_spec: MCP server URL
        evaluation_rows: List of EvaluationRow objects containing messages and metadata (preferred)
        dataset: List of dataset entries (for backward compatibility)
        n: Number of environments (for backward compatibility)
        seeds: List of seeds (for backward compatibility)
        model_id: Model identifier
        user_prompt_formatter: Optional callback for formatting user prompts

    Returns:
        General MCP environment that works with any MCP server

    Example:
        # EvaluationRow approach (preferred)
        envs = ep.make("http://localhost:8000/mcp", evaluation_rows=evaluation_rows)

        # Dataset approach (backward compatibility)
        envs = ep.make("http://localhost:8000/mcp", dataset=dataset)

        # Legacy approach (backward compatibility)
        envs = ep.make("http://localhost:8000/mcp", n=10, seeds=seeds)
    """
    # Parse environment specification - make sure URL format is correct
    base_url = env_spec
    if not base_url.startswith("http"):
        raise ValueError("Environment spec must be a valid HTTP URL")

    # Ensure we HAVE a trailing slash to avoid 307 redirects that break POST requests
    if not base_url.endswith("/"):
        base_url += "/"

    # Convert evaluation_rows to dataset format if provided
    internal_dataset = []

    if evaluation_rows:
        for i, row in enumerate(evaluation_rows):
            dataset_info = (
                row.input_metadata.dataset_info
                if (row.input_metadata and row.input_metadata.dataset_info is not None)
                else {}
            )

            system_message = row.get_system_message()
            system_prompt = system_message.content or ""

            dataset_entry = {
                "id": row.input_metadata.row_id if row.input_metadata and row.input_metadata.row_id else f"task_{i}",
                "system_prompt": system_prompt,
                "user_prompt_template": dataset_info.get("user_prompt_template", ""),
                "environment_context": dataset_info.get("environment_context", {}),
                "user_simulation": dataset_info.get("user_simulation", {}),
                "evaluation_criteria": dataset_info.get("evaluation_criteria", {}),
            }
            internal_dataset.append(dataset_entry)
    elif dataset:
        # Use provided dataset directly for backward compatibility
        internal_dataset = dataset

    dataset_rows = []
    sessions = []

    # Handle evaluation_rows vs legacy approaches
    if internal_dataset:
        # New evaluation_rows approach
        dataset_rows = []
        sessions = []

        for row in internal_dataset:
            # Parse dataset row
            if isinstance(row, dict):
                # Handle seed from both old location (backward compatibility) and new location
                environment_context = row.get("environment_context", {})
                seed = environment_context.get("seed")

                dataset_row = DatasetRow(
                    id=row["id"],
                    seed=seed,
                    system_prompt=row["system_prompt"],
                    user_prompt_template=row["user_prompt_template"],
                    environment_context=environment_context,
                    user_simulation=(row["user_simulation"] if "user_simulation" in row else None),
                )
            else:
                dataset_row = row  # Assume it's already a DatasetRow

            dataset_rows.append(dataset_row)

            session_id = gen_session_id(dataset_row, model_id)
            # Create MCP session
            session = MCPSession(
                session_id=session_id,
                base_url=base_url,
                seed=dataset_row.seed,
                model_id=model_id,
                dataset_row=dataset_row,
            )
            sessions.append(session)

    else:
        # Legacy approach for backward compatibility
        if n is None:
            raise ValueError("Either 'evaluation_rows' or 'n' must be provided")

        # Generate seeds if not provided
        if seeds is None:
            seeds = [random.randint(0, 2**31 - 1) for _ in range(n)]
        elif len(seeds) != n:
            raise ValueError(f"Expected {n} seeds, got {len(seeds)}")

        # Create default dataset rows for legacy mode
        dataset_rows = []
        sessions = []

        for i in range(n):
            # Create a default dataset row (environment-agnostic)
            dataset_row = DatasetRow(
                id=f"session_{i}",
                seed=seeds[i],
                system_prompt="You are an AI agent interacting with an environment via available tools.",
                user_prompt_template="Current observation: {observation}. Use available tools to interact with the environment.",
                environment_context={},
            )
            dataset_rows.append(dataset_row)

            session_id = gen_session_id(dataset_row, model_id)

            # Create MCP session
            session = MCPSession(
                session_id=session_id,
                base_url=base_url,
                seed=seeds[i],
                model_id=model_id,
                dataset_row=dataset_row,
            )
            sessions.append(session)

    mcp_envs = GeneralMCPVectorEnv(sessions, dataset_rows, user_prompt_formatter)
    return mcp_envs


async def rollout(
    envs: GeneralMCPVectorEnv,
    policy: Union[FireworksPolicy, LLMBasePolicy, Callable],
    *,
    evaluation_rows: Optional[List[EvaluationRow]] = None,
    dataset: Optional[List[Dict]] = None,
    model_id: Optional[str] = None,
    steps: int = 512,
    openai_format_log_file: Optional[str] = None,
    max_concurrent_rollouts: int = 8,
) -> List[EvaluationRow]:
    """
    Execute general rollouts using tool calling interface with automatic record/playback.

    Uses concurrent execution with semaphore-based concurrency control for efficiency.

    This works with ANY MCP environment because:
    1. Policy receives tool schemas and makes tool calls
    2. Environment prompts come from evaluation_rows
    3. No hardcoded environment logic

    Args:
        envs: Either a GeneralMCPVectorEnv instance or the MCP server URL
        policy: Policy that takes tool schemas, observations, prompts and returns tool calls
        evaluation_rows: EvaluationRow list used when envs is a URL (for automatic env creation)
        dataset: Dataset list used for backward compatibility when envs is a URL
        model_id: Model identifier used when creating environments. Defaults to ``policy.model_id`` when available.
        steps: Maximum steps per rollout
        openai_format_log_file: Optional file to log clean OpenAI format for terminated trajectories only
        max_concurrent_rollouts: Maximum number of concurrent rollouts to run

    Environment Variable Control:
        EP_PLAYBACK_FILE: Controls record/playback mode
        - Not set: Normal live mode
        - Set but file doesn't exist: Record mode (file will be created)
        - Set and file exists: Playback mode (uses recorded data)

    Returns:
        List of asyncio.Task objects for external handling

    Example:
        # Live mode
        results = await ep.rollout(envs, policy)

        # Create environments automatically
        results = await ep.rollout(
            "http://localhost:8000/mcp/",
            policy,
            evaluation_rows=my_evaluation_rows,
            model_id=policy.model_id,
        )

        # Recording mode
        os.environ["EP_PLAYBACK_FILE"] = "record.jsonl"
        results = await ep.rollout(envs, policy, openai_format_log_file="sft_data.jsonl")

        # Playback mode (after recording file exists)
        results = await ep.rollout(envs, policy)
    """
    # Automatically create environments if a base URL is provided
    if isinstance(envs, str):
        if evaluation_rows is None and dataset is None:
            raise ValueError("Either 'evaluation_rows' or 'dataset' must be provided when envs is a URL")

        auto_model_id = model_id or getattr(policy, "model_id", "unknown")
        envs = make(envs, evaluation_rows=evaluation_rows, dataset=dataset, model_id=auto_model_id)

    # Use the new ExecutionManager for execution
    execution_manager = ExecutionManager()

    rollout_semaphore = asyncio.Semaphore(max_concurrent_rollouts)

    tasks = execution_manager.execute_rollouts(
        envs,
        policy,
        semaphore=rollout_semaphore,
        steps=steps,
        openai_format_log_file=openai_format_log_file,
        evaluation_rows=evaluation_rows,
    )

    # Await all tasks and return concrete EvaluationRows
    # Gather returns list of EvaluationRow; use type ignore to appease Pyright when inferring coroutine types
    results: List[EvaluationRow] = await asyncio.gather(*tasks)  # type: ignore[reportUnknownArgumentType]
    return results


async def test_mcp(base_url: str, seeds: List[int]) -> Dict[str, Any]:
    """
    Test function for validating MCP server as mentioned in north star document.

    Args:
        base_url: Base URL of MCP server (e.g., "http://localhost:8000/mcp")
        seeds: List of seeds to test

    Returns:
        Test results dictionary
    """
    print(f"🧪 Testing MCP server at {base_url} with {len(seeds)} seeds...")

    results = {"total_tests": len(seeds), "successful": 0, "failed": 0, "results": []}

    for seed in seeds:
        try:
            # Create single environment
            envs = make(base_url, n=1, seeds=[seed], model_id="test-model")

            # Simple policy for testing
            policy = FireworksPolicy("test-model")

            # Run short rollout
            evaluation_rows = await rollout(envs, policy=policy, steps=10)

            if evaluation_rows and len(evaluation_rows[0].messages) > 1:
                results["successful"] += 1
                results["results"].append(
                    {
                        "seed": seed,
                        "status": "success",
                        "steps": evaluation_rows[0].get_steps(),
                        "total_reward": evaluation_rows[0].get_total_reward(),
                    }
                )
            else:
                results["failed"] += 1
                results["results"].append({"seed": seed, "status": "failed", "error": "empty_trajectory"})

        except Exception as e:
            results["failed"] += 1
            results["results"].append({"seed": seed, "status": "failed", "error": str(e)})

    success_rate = results["successful"] / results["total_tests"] * 100
    print(f"✅ Test complete: {results['successful']}/{results['total_tests']} successful ({success_rate:.1f}%)")

    return results


# Add to eval_protocol.__init__.py exports
__all__ = [
    "make",
    "rollout",
    "AnthropicPolicy",
    "FireworksPolicy",
    "OpenAIPolicy",
    "LiteLLMPolicy",
    "LLMBasePolicy",  # New base class for OpenAI integration
    "GeneralMCPVectorEnv",
    "MCPToolCall",
    "DatasetRow",
    "test_mcp",
]
