"""
Lightweight vLLM + OpenEnv Integration

Minimal integration to use TRL's vLLM server for inference with OpenEnv BrowserGym
environments, wired into GRPO via a custom ``rollout_func``.

- Uses TRL's ``VLLMClient`` (``use_vllm=True, vllm_mode="server"``) for inference
- Uses ``OpenEnvRolloutProcessor`` to drive OpenEnv (BrowserGym-style) environments
- Supports task rotation across MiniWoB tasks
- Returns Wordle-style GRPO data: 2D token lists and 1D per-episode rewards
- No Fireworks, no hot reload, no additional providers
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable, Dict, List, Optional, Type

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig


def create_openenv_vllm_rollout_func(
    env_factory: Callable[[], Any] | None,
    prompt_builder: Callable[[Any, int, list[str]], Any],
    action_parser: Callable[[str], Any],
    vllm_base_url: str = "http://localhost:8000",
    vllm_model: str = "Qwen/Qwen2.5-7B",
    max_steps: int = 8,
    *,
    completion_params: Dict[str, Any] | None = None,
    concurrency: int | None = None,
    processor_cls: Optional[Type[Any]] = OpenEnvRolloutProcessor,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    # Environment configuration
    env_client_cls: Optional[Type[Any]] = None,
    tasks: List[str] | None = None,
    task_var: Optional[str] = None,
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
    Build a TRL-compatible ``rollout_func`` using vLLM inference with OpenEnv.

    High-level:
    - ``GRPOTrainer`` calls the returned ``rollout_func(prompts, trainer)``
    - For each prompt, we create ``num_generations`` evaluation rows
    - ``OpenEnvRolloutProcessor`` runs BrowserGym-style episodes via Docker
    - ``VLLMPolicy`` formats messages with the chat template and calls TRL's
      vLLM server using ``trainer.vllm_client``
    - We accumulate tokens across all turns of an episode and sum rewards,
      returning Wordle-style GRPO data.

    The environment side is configured via ``env_client_cls`` and the BrowserGym
    parameters (``tasks``, ``miniwob_url``, ``docker_image``, etc.).
    """
    print(f"\n{'=' * 80}", flush=True)
    print("[openenv_trl_vllm] create_openenv_vllm_rollout_func() CALLED", flush=True)
    print(f"  vllm_base_url: {vllm_base_url}", flush=True)
    print(f"  vllm_model: {vllm_model}", flush=True)
    print(f"  tasks: {tasks}", flush=True)
    print(f"  max_steps: {max_steps}", flush=True)
    print(f"{'=' * 80}", flush=True)
    sys.stdout.flush()

    # Import VLLMPolicy
    from eval_protocol.mcp.execution.vllm_policy import VLLMPolicy

    # Global-ish task rotation offset across rollout_func calls.
    # This lets us rotate tasks between GRPO steps instead of always
    # starting from tasks[0] when a new OpenEnvRolloutProcessor is created.
    task_cycle_index: int = 0

    def rollout_func(prompts: List[str], trainer) -> Dict[str, List]:
        """Execute rollouts via OpenEnv + vLLM and return GRPO-compatible results."""
        print("\n[OpenEnvVLLM] rollout_func called", flush=True)

        # Extract args from trainer
        args = trainer.args
        processing_class = trainer.processing_class

        num_generations = getattr(args, "num_generations", 8)
        print(
            f"[OpenEnvVLLM] Received {len(prompts)} prompts, {num_generations} generations each",
            flush=True,
        )

        # 1) Build evaluation rows
        evaluation_rows: List[EvaluationRow] = []
        for prompt in prompts:
            for gen_idx in range(num_generations):
                row = EvaluationRow(
                    messages=[Message(role="user", content=prompt)],
                    input_metadata=InputMetadata(completion_params={}),
                )
                row.input_metadata.generation_idx = gen_idx  # type: ignore[attr-defined]
                evaluation_rows.append(row)

        # 2) Build processor config with VLLMPolicy
        # We'll pass trainer.vllm_client to VLLMPolicy
        base_params: Dict[str, Any] = {
            "model": "dummy",  # Not used by VLLMPolicy, but needed for config
            "temperature": getattr(args, "temperature", 1.0),
            "max_tokens": getattr(args, "max_completion_length", 100),
        }
        if completion_params:
            base_params.update(completion_params)

        print(
            f"[OpenEnvVLLM] Temperature={base_params['temperature']}, max_tokens={base_params['max_tokens']}",
            flush=True,
        )
        print("[OpenEnvVLLM] Using TRL VLLMClient from trainer", flush=True)

        max_concurrency = concurrency if concurrency is not None else getattr(args, "per_device_train_batch_size", 1)
        print(
            f"[OpenEnvVLLM] Max concurrency={max_concurrency}, max_steps={max_steps}",
            flush=True,
        )

        config = RolloutProcessorConfig(
            completion_params=base_params,
            mcp_config_path="",
            semaphore=asyncio.Semaphore(max_concurrency),
            steps=max_steps,
        )

        # 3) Execute rollouts with VLLMPolicy
        print(
            f"[OpenEnvVLLM] Instantiating processor: "
            f"{processor_cls.__name__ if processor_cls else 'OpenEnvRolloutProcessor'}",
            flush=True,
        )

        # Create policy factory that uses trainer's vllm_client
        def vllm_policy_factory(model, temperature, max_tokens, base_url=None, **kwargs):
            """Factory that creates VLLMPolicy using trainer's vllm_client."""
            return VLLMPolicy(
                vllm_client=trainer.vllm_client,  # Use trainer's vLLM client!
                tokenizer=processing_class,  # Pass tokenizer for decoding
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get("top_p"),
                top_k=kwargs.get("top_k"),
                **kwargs,
            )

        Processor = processor_cls or OpenEnvRolloutProcessor
        _kwargs: Dict[str, Any] = dict(processor_kwargs or {})
        _kwargs.setdefault("env_factory", env_factory)
        _kwargs.setdefault("prompt_builder", prompt_builder)
        _kwargs.setdefault("action_parser", action_parser)
        _kwargs.setdefault("policy_factory", vllm_policy_factory)  # Pass VLLMPolicy factory!
        _kwargs.setdefault("env_client_cls", env_client_cls)

        # Rotate tasks across rollout_func calls so each GRPO step
        # primarily targets a different task, while keeping all
        # generations within a step on the same task.
        rotated_tasks = tasks
        if tasks:
            nonlocal task_cycle_index
            offset = task_cycle_index % len(tasks)
            rotated_tasks = tasks[offset:] + tasks[:offset]
            task_cycle_index = (task_cycle_index + 1) % len(tasks)
            print(
                f"[OpenEnvVLLM] Task rotation offset={offset}, rotated={rotated_tasks}",
                flush=True,
            )
        _kwargs.setdefault("tasks", rotated_tasks)
        _kwargs.setdefault("task_var", task_var)

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
        print("[OpenEnvVLLM] Processor instantiated successfully", flush=True)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:

            async def _run_all():
                tasks_list = processor(evaluation_rows, config)
                return await asyncio.gather(*tasks_list)

            completed_rows = loop.run_until_complete(_run_all())
            print(
                f"[OpenEnvVLLM] All rollouts completed: {len(completed_rows)} results",
                flush=True,
            )
        finally:
            loop.close()

        # 4) Convert to Wordle-style format (no splitting)
        # Each completed_row is one rollout with multiple turns
        # We .extend() tokens across turns, then .append() per rollout
        print(
            f"[OpenEnvVLLM] Converting {len(completed_rows)} rollouts to TRL format",
            flush=True,
        )

        tokenizer = getattr(processing_class, "tokenizer", None) or processing_class
        encode_fn = getattr(tokenizer, "encode", None)

        episode_prompt_ids: List[List[int]] = []
        episode_completion_ids: List[List[int]] = []
        episode_logprobs: List[List[float]] = []
        step_rewards_all: List[List[float]] = []

        for idx, row in enumerate(completed_rows):
            # Accumulate tokens across all turns in this rollout
            prompt_ids: List[int] = []  # .extend() for each turn
            completion_ids: List[int] = []  # .extend() for each turn
            logprobs: List[float] = []  # .extend() for each turn
            rewards: List[float] = []

            # Go through all messages and accumulate tokens
            for msg in row.messages:
                if msg.role == "user":
                    tokens = encode_fn(msg.content or "") if encode_fn else []
                    prompt_ids.extend(tokens)  # Accumulate user tokens
                elif msg.role == "assistant":
                    tokens = encode_fn(msg.content or "") if encode_fn else []
                    completion_ids.extend(tokens)  # Accumulate assistant tokens
                    logprobs.extend([0.0] * len(tokens))  # Placeholder logprobs
                elif msg.role == "system":
                    # Extract step rewards
                    try:
                        content = msg.content or ""
                        if isinstance(content, str) and content.startswith("__ep_step_rewards__:"):
                            import json

                            payload = content.split(":", 1)[1]
                            rewards = json.loads(payload) or []
                    except Exception:
                        pass

            # Fallback for rewards (if extra field exists via model_config extra="allow")
            if not rewards:
                try:
                    extra = getattr(row.execution_metadata, "extra", None)
                    if isinstance(extra, dict):
                        rewards = extra.get("step_rewards", []) or []
                except Exception:
                    pass

            # Append accumulated tokens for this episode
            episode_prompt_ids.append(prompt_ids if prompt_ids else [0])
            episode_completion_ids.append(completion_ids if completion_ids else [0])
            episode_logprobs.append(logprobs if logprobs else [0.0])
            step_rewards_all.append(rewards if rewards else [0.0])

        total_reward = sum(sum(r) for r in step_rewards_all)
        avg_reward = total_reward / len(step_rewards_all) if step_rewards_all else 0.0
        print(
            f"[OpenEnvVLLM] Total reward={total_reward:.2f}, Avg reward={avg_reward:.2f}",
            flush=True,
        )
        print(f"[OpenEnvVLLM] Returning {len(episode_prompt_ids)} episodes", flush=True)
        sys.stdout.flush()

        # Return in Wordle format
        # Tokens: 2D arrays (accumulate across turns, one list per episode)
        # Rewards: 1D arrays (one scalar per episode)
        total_rewards = [sum(r) for r in step_rewards_all]  # Sum step rewards per episode

        print(f"[OpenEnvVLLM] Episode rewards: {total_rewards}", flush=True)

        return {
            "prompt_ids": episode_prompt_ids,  # List[List[int]] - tokens per episode
            "completion_ids": episode_completion_ids,  # List[List[int]] - tokens per episode
            "logprobs": episode_logprobs,  # List[List[float]] - logprobs per episode
            "step_rewards": total_rewards,  # List[float] - total reward per episode (1D!)
        }

    print(f"[openenv_trl_vllm] Returning rollout_func (type={type(rollout_func)})", flush=True)
    sys.stdout.flush()
    return rollout_func
