"""
Lightweight vLLM + OpenEnv Integration

Simplified integration using vLLM for inference with proper multi-turn completion splitting.
No Fireworks inference, no hot reload - just vLLM.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.utils.evaluation_row_utils import (
    filter_longest_conversation,
    multi_turn_assistant_to_ground_truth,
    assistant_to_ground_truth,
)
from trl import GRPOConfig


def create_openenv_vllm_rollout_func(
    env_factory: Callable[[], Any] | None,
    prompt_builder: Callable[[Any, int, list[str]], Any],
    action_parser: Callable[[str], Any],
    vllm_base_url: str = "http://localhost:8000",
    max_steps: int = 8,
    *,
    split_mode: str = "multi_turn",  # "multi_turn", "last_turn", "longest", or None
    completion_params: Dict[str, Any] | None = None,
    concurrency: int | None = None,
    processor_cls: Optional[Type[Any]] = OpenEnvRolloutProcessor,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    # Environment configuration
    env_client_cls: Optional[Type[Any]] = None,
    tasks: List[str] | None = None,
    miniwob_url: str | None = None,
    docker_image: str = "browsergym-env:latest",
    env_base_url: Optional[str] = None,
    request_timeout_s: float = 15.0,
    default_headers: Optional[Dict[str, str]] = None,
    provider: Any | None = None,
    docker_port: Optional[int] = None,
    env_vars: Optional[Dict[str, str]] = None,
    benchmark: str = "miniwob",
    headless: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    timeout_ms: int = 10000,
):
    """
    Build a TRL-compatible rollout_func using vLLM inference with OpenEnv.
    
    This is a lightweight version that:
    - Uses vLLM client directly (no Fireworks, no hot reload)
    - Properly splits completions using evaluation_row_utils helpers
    - Works with TRL's GRPO trainer
    
    Args:
        env_factory: Callable yielding an OpenEnv HTTPEnvClient instance
        prompt_builder: (observation, step, history) -> content for LLM
        action_parser: (llm_response: str) -> env action object
        vllm_base_url: Base URL for vLLM server (e.g., "http://localhost:8000")
        max_steps: Maximum environment steps per rollout
        split_mode: How to split completions:
            - "multi_turn": Split each assistant message as separate row (multi_turn_assistant_to_ground_truth)
            - "last_turn": Extract last assistant message as ground truth (assistant_to_ground_truth)
            - "longest": Keep only longest conversation (filter_longest_conversation)
            - None: No splitting, return all rows as-is
        completion_params: Extra completion parameters (temperature, max_tokens, etc.)
        concurrency: Max concurrent rollouts (defaults to per_device_train_batch_size)
        processor_cls: Rollout processor class (default: OpenEnvRolloutProcessor)
        processor_kwargs: Extra kwargs for processor
        env_client_cls: Environment client class
        tasks: List of task names to rotate through
        miniwob_url: MiniWoB base URL
        docker_image: Docker image for environments
        env_base_url: Direct HTTP connection to existing server
        request_timeout_s: HTTP timeout
        default_headers: HTTP headers
        provider: Docker provider
        docker_port: Host port binding
        env_vars: Environment variables for container
        benchmark: BrowserGym benchmark name
        headless: Headless browser mode
        viewport_width/height: Browser viewport size
        timeout_ms: Action timeout
    
    Returns:
        rollout_func(prompts: List[str], args: GRPOConfig, processing_class) -> Dict[str, List]
    
    Example:
        ```python
        from trl import GRPOConfig, GRPOTrainer
        from trl.extras.vllm_client import VLLMClient
        from envs.browsergym_env import BrowserGymEnv, BrowserGymAction
        
        # Start vLLM server first:
        # CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve --model Qwen/Qwen2.5-7B --tensor-parallel-size 2
        
        def make_env():
            return BrowserGymEnv.from_docker_image(
                "browsergym-env:latest",
                env_vars={"BROWSERGYM_BENCHMARK": "miniwob"}
            )
        
        def build_prompt(obs, step, history):
            return f"Step {step}\\nGoal: {obs.goal}\\n{obs.text[:500]}"
        
        def parse_action(text):
            return BrowserGymAction(action_str=text)
        
        rollout_func = create_openenv_vllm_rollout_func(
            env_factory=make_env,
            prompt_builder=build_prompt,
            action_parser=parse_action,
            vllm_base_url="http://localhost:8000",
            tasks=["click-test", "click-button", "enter-text"],
            split_mode="multi_turn",  # Split each turn for training
        )
        
        training_args = GRPOConfig(
            output_dir="outputs/vllm-training",
            per_device_train_batch_size=2,
            num_generations=4,
        )
        
        trainer = GRPOTrainer(
            model="Qwen/Qwen2.5-7B",
            args=training_args,
            train_dataset=dataset,
            rollout_func=rollout_func,
        )
        ```
    """
    
    # Import vLLM client (will be used for generation)
    try:
        from trl.extras.vllm_client import VLLMClient
    except ImportError:
        raise ImportError(
            "vLLM client not available. Install with: pip install trl[vllm]"
        )
    
    # Initialize vLLM client
    vllm_client = VLLMClient(base_url=vllm_base_url)
    
    def rollout_func(prompts: List[str], args: GRPOConfig, processing_class) -> Dict[str, List]:
        """
        Execute rollouts and return TRL-compatible results.
        
        Flow:
        1. Prompts → EvaluationRows (num_generations per prompt)
        2. Execute rollouts via OpenEnvRolloutProcessor
        3. Split completions using evaluation_row_utils
        4. Generate completions via vLLM for each split row
        5. Convert to TRL format
        """
        num_generations = getattr(args, "num_generations", 8)
        
        # 1) Build evaluation rows (one per generation per prompt)
        evaluation_rows: List[EvaluationRow] = []
        for prompt in prompts:
            for gen_idx in range(num_generations):
                evaluation_rows.append(
                    EvaluationRow(
                        messages=[Message(role="user", content=prompt)],
                        input_metadata=InputMetadata(
                            completion_params={},
                            extra={"generation_idx": gen_idx}
                        ),
                    )
                )
        
        # 2) Build processor config
        base_params: Dict[str, Any] = {
            "temperature": getattr(args, "temperature", 1.0),
            "max_tokens": getattr(args, "max_completion_length", 100),
        }
        if completion_params:
            base_params.update(completion_params)
        
        max_concurrency = concurrency if concurrency is not None else getattr(
            args, "per_device_train_batch_size", 1
        )
        
        config = RolloutProcessorConfig(
            completion_params=base_params,
            mcp_config_path="",
            semaphore=asyncio.Semaphore(max_concurrency),
            steps=max_steps,
        )
        
        # 3) Execute rollouts using OpenEnvRolloutProcessor
        Processor = processor_cls or OpenEnvRolloutProcessor
        _kwargs: Dict[str, Any] = dict(processor_kwargs or {})
        _kwargs.setdefault("env_factory", env_factory)
        _kwargs.setdefault("prompt_builder", prompt_builder)
        _kwargs.setdefault("action_parser", action_parser)
        _kwargs.setdefault("env_client_cls", env_client_cls)
        _kwargs.setdefault("tasks", tasks)
        _kwargs.setdefault("miniwob_url", miniwob_url)
        _kwargs.setdefault("docker_image", docker_image)
        _kwargs.setdefault("env_base_url", env_base_url)
        _kwargs.setdefault("request_timeout_s", request_timeout_s)
        _kwargs.setdefault("default_headers", default_headers)
        _kwargs.setdefault("provider", provider)
        _kwargs.setdefault("docker_port", docker_port)
        _kwargs.setdefault("env_vars", env_vars)
        _kwargs.setdefault("benchmark", benchmark)
        _kwargs.setdefault("headless", headless)
        _kwargs.setdefault("viewport_width", viewport_width)
        _kwargs.setdefault("viewport_height", viewport_height)
        _kwargs.setdefault("timeout_ms", timeout_ms)
        _kwargs.setdefault("num_generations", num_generations)
        
        processor = Processor(**_kwargs)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _run_all():
                tasks = processor(evaluation_rows, config)
                return await asyncio.gather(*tasks)
            
            completed_rows = loop.run_until_complete(_run_all())
        finally:
            loop.close()
        
        # 4) Split completions based on split_mode
        if split_mode == "multi_turn":
            # Split each assistant message into separate rows
            split_rows = multi_turn_assistant_to_ground_truth(completed_rows)
        elif split_mode == "last_turn":
            # Extract last assistant message as ground truth
            split_rows = assistant_to_ground_truth(completed_rows)
        elif split_mode == "longest":
            # Keep only longest conversation per rollout_id
            split_rows = filter_longest_conversation(completed_rows)
        elif split_mode is None:
            # No splitting
            split_rows = completed_rows
        else:
            raise ValueError(
                f"Invalid split_mode: {split_mode}. "
                "Must be 'multi_turn', 'last_turn', 'longest', or None"
            )
        
        print(f"[OpenEnvVLLM] Split {len(completed_rows)} rows → {len(split_rows)} rows (mode={split_mode})")
        
        # 5) Generate completions via vLLM for each split row
        # Build messages for vLLM chat endpoint
        all_messages: List[List[Dict]] = []
        for row in split_rows:
            messages = [{"role": msg.role, "content": msg.content} for msg in row.messages]
            all_messages.append(messages)
        
        # Call vLLM to generate completions
        # Check if we have conversational format
        is_conversational = all_messages and isinstance(all_messages[0], list)
        
        vllm_params = {
            "n": 1,  # One completion per split row
            "temperature": base_params["temperature"],
            "max_tokens": base_params["max_tokens"],
        }
        
        # Add any extra vLLM parameters from completion_params
        if completion_params:
            for key in ["top_p", "top_k", "min_p", "repetition_penalty"]:
                if key in completion_params:
                    vllm_params[key] = completion_params[key]
        
        if is_conversational:
            print(f"[OpenEnvVLLM] Calling vLLM chat endpoint with {len(all_messages)} conversations")
            vllm_response = vllm_client.chat(
                messages=all_messages,
                **vllm_params,
            )
        else:
            # Convert messages to prompts for generate endpoint
            prompts_for_vllm = []
            for msgs in all_messages:
                # Simple concatenation (you may want to use a chat template here)
                prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
                prompts_for_vllm.append(prompt_text)
            
            print(f"[OpenEnvVLLM] Calling vLLM generate endpoint with {len(prompts_for_vllm)} prompts")
            vllm_response = vllm_client.generate(
                prompts=prompts_for_vllm,
                **vllm_params,
            )
        
        # 6) Convert to TRL format
        prompt_ids = vllm_response["prompt_ids"]
        completion_ids = vllm_response["completion_ids"]
        logprobs = vllm_response["logprobs"]
        
        # Extract step rewards from completed rows
        step_rewards: List[List[float]] = []
        for row in split_rows:
            rewards: List[float] = []
            
            # Look for rewards in system messages (sentinel pattern)
            for msg in row.messages:
                if msg.role == "system":
                    try:
                        content = msg.content or ""
                        if isinstance(content, str) and content.startswith("__ep_step_rewards__:"):
                            import json
                            payload = content.split(":", 1)[1]
                            rewards = json.loads(payload) or []
                            break
                    except Exception:
                        pass
            
            # Fallback to execution metadata
            if not rewards and hasattr(row.execution_metadata, "extra"):
                try:
                    rewards = row.execution_metadata.extra.get("step_rewards", []) or []
                except Exception:
                    pass
            
            step_rewards.append(rewards if rewards else [0.0])
        
        # Compute statistics
        total_reward = sum(sum(r) for r in step_rewards)
        avg_reward = total_reward / len(step_rewards) if step_rewards else 0.0
        print(f"[OpenEnvVLLM] Total reward: {total_reward:.2f}, Avg: {avg_reward:.2f}")
        
        # TRL expects prompt_ids at unique-prompt level (not per-generation)
        # Deduplicate while preserving order
        seen_prompts = set()
        prompt_ids_unique = []
        for p_ids in prompt_ids:
            p_tuple = tuple(p_ids)
            if p_tuple not in seen_prompts:
                seen_prompts.add(p_tuple)
                prompt_ids_unique.append(p_ids)
        
        return {
            "prompt_ids": prompt_ids_unique,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "step_rewards": step_rewards,
        }
    
    return rollout_func

