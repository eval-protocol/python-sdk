"""
MCP Environment API for reward-kit - Backward Compatibility Facade

This module has been refactored into modular components for better maintainability.
This file now serves as a backward compatibility facade.

New modular structure:
- mcp.client.connection: MCP client connection management
- mcp.execution.policy: LLMBasePolicy and FireworksPolicy for tool calling
- mcp.execution.rollout: Rollout coordination and lifecycle
- mcp.session.manager: Session and environment management

Usage remains the same:
    import eval_protocol as ep

    # Load dataset with environment configuration and prompts
    dataset = load_jsonl("dataset.jsonl")

    # Create general policy (environment-agnostic)
    policy = ep.FireworksPolicy(model_id="accounts/fireworks/models/qwen3-235b-a22b")

    # Create environments with dataset-driven configuration
    envs = ep.make("http://localhost:8000/mcp", dataset=dataset)

    # Execute tool-calling rollouts
    trajectories = await ep.rollout(envs, policy=policy, steps=512)

Key Features:
- General tool-calling interface that works with any MCP environment
- Dataset-driven configuration with system prompts and user prompt templates
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

# For legacy compatibility - import the facade functions
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Union

# Import all functionality from the new modular components
from .mcp.execution.manager import ExecutionManager
from .mcp.execution.policy import AnthropicPolicy, LLMBasePolicy, OpenAIPolicy
from .mcp.session.manager import GeneralMCPVectorEnv
from .mcp.types import DatasetRow, MCPSession, MCPToolCall, Trajectory

# Try to import FireworksPolicy - it may fail if fireworks-ai is not installed
# or if a different 'fireworks' package is installed
try:
    from .mcp.execution.policy import FireworksPolicy
except:
    # Silently skip if import fails for any reason
    pass

logger = logging.getLogger(__name__)

# Keep the old MCPVectorEnv for backward compatibility
MCPVectorEnv = GeneralMCPVectorEnv


def make(
    env_spec: str,
    dataset: Optional[List[Dict]] = None,
    n: Optional[int] = None,
    seeds: Optional[List[int]] = None,
    model_id: str = "unknown",
    user_prompt_formatter: Optional[Callable] = None,
) -> GeneralMCPVectorEnv:
    """
    Create general MCP environments driven by dataset configuration.

    Args:
        env_spec: MCP server URL
        dataset: List of dataset rows with prompts and context (preferred)
        n: Number of environments (for backward compatibility)
        seeds: List of seeds (for backward compatibility)
        model_id: Model identifier
        user_prompt_formatter: Optional callback for formatting user prompts

    Returns:
        General MCP environment that works with any MCP server

    Example:
        # New dataset-driven approach (preferred)
        dataset = load_jsonl("dataset.jsonl")
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

    # Handle dataset-driven vs legacy approaches
    if dataset is not None:
        # New dataset-driven approach
        dataset_rows = []
        sessions = []

        for row in dataset:
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

            # Create MCP session
            session = MCPSession(
                session_id=dataset_row.id,
                base_url=base_url,
                seed=dataset_row.seed,
                model_id=model_id,
                dataset_row=dataset_row,
            )
            sessions.append(session)

        return GeneralMCPVectorEnv(sessions, dataset_rows, user_prompt_formatter)

    else:
        # Legacy approach for backward compatibility
        if n is None:
            raise ValueError("Either 'dataset' or 'n' must be provided")

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

            # Create MCP session
            session = MCPSession(
                session_id=f"session_{i}",
                base_url=base_url,
                seed=seeds[i],
                model_id=model_id,
                dataset_row=dataset_row,
            )
            sessions.append(session)

        return GeneralMCPVectorEnv(sessions, dataset_rows, user_prompt_formatter)


async def rollout(
    envs: Union[GeneralMCPVectorEnv, "MCPVectorEnv"],
    policy: Union[FireworksPolicy, LLMBasePolicy, Callable],
    steps: int = 512,
    openai_format_log_file: Optional[str] = None,
    max_concurrent_rollouts: int = 8,
) -> List[Trajectory]:
    """
    Execute general rollouts using tool calling interface with automatic record/playback.

    Uses concurrent execution with semaphore-based concurrency control for efficiency.

    This works with ANY MCP environment because:
    1. Policy receives tool schemas and makes tool calls
    2. Environment prompts come from dataset
    3. No hardcoded environment logic

    Args:
        envs: GeneralMCPVectorEnv instance
        policy: Policy that takes tool schemas, observations, prompts and returns tool calls
        steps: Maximum steps per rollout
        openai_format_log_file: Optional file to log clean OpenAI format for terminated trajectories only
        max_concurrent_rollouts: Maximum number of concurrent rollouts to run

    Environment Variable Control:
        EP_PLAYBACK_FILE: Controls record/playback mode
        - Not set: Normal live mode
        - Set but file doesn't exist: Record mode (file will be created)
        - Set and file exists: Playback mode (uses recorded data)

    Returns:
        List of Trajectory objects with complete rollout data

    Example:
        # Live mode
        trajectories = await ep.rollout(envs, policy)

        # Recording mode
        os.environ["EP_PLAYBACK_FILE"] = "record.jsonl"
        trajectories = await ep.rollout(envs, policy, openai_format_log_file="sft_data.jsonl")

        # Playback mode (after recording file exists)
        trajectories = await ep.rollout(envs, policy)
    """
    # Use the new ExecutionManager for execution
    execution_manager = ExecutionManager()

    return await execution_manager.execute_rollouts(
        envs, policy, steps, openai_format_log_file, max_concurrent_rollouts
    )


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
            trajectories = await rollout(envs, policy=policy, steps=10)

            if trajectories and len(trajectories[0].observations) > 1:
                results["successful"] += 1
                results["results"].append(
                    {
                        "seed": seed,
                        "status": "success",
                        "steps": trajectories[0].steps,
                        "total_reward": trajectories[0].total_reward,
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
    "LLMBasePolicy",  # New base class for OpenAI integration
    "MCPVectorEnv",
    "GeneralMCPVectorEnv",
    "MCPToolCall",
    "DatasetRow",
    "Trajectory",
    "test_mcp",
]
