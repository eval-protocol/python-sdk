"""
TRL + OpenEnv Integration Helper

This module exposes a single helper to build a TRL-compatible rollout_func
using the OpenEnvRolloutProcessor. It converts dataset prompts → EvaluationRows,
executes rollouts with concurrency, and converts results back to TRL's format.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type
import re

from eval_protocol.models import EvaluationRow, InputMetadata
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from trl import GRPOConfig


def create_openenv_rollout_func(
    env_factory: Callable[[], Any] | None,
    prompt_builder: Callable[[Any, int, list[str]], Any],
    action_parser: Callable[[str], Any],
    model: str = "gpt-4o-mini",
    max_steps: int = 8,
    *,
    completion_params: Dict[str, Any] | None = None,
    concurrency: int | None = None,
    # Allow any rollout processor to be used
    processor_cls: Optional[Type[Any]] = OpenEnvRolloutProcessor,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    # Optional environment integration (build a default env_factory if not provided)
    env_client_cls: Optional[Type[Any]] = None,
    tasks: List[str] | None = None,
    miniwob_url: str | None = None,
    docker_image: str = "browsergym-env:latest",
    # HTTP client/direct server options (match HTTPEnvClient interface)
    env_base_url: Optional[str] = None,
    request_timeout_s: float = 15.0,
    default_headers: Optional[Dict[str, str]] = None,
    # Docker provider passthrough (match HTTPEnvClient.from_docker_image)
    provider: Any | None = None,
    docker_port: Optional[int] = None,
    env_vars: Optional[Dict[str, str]] = None,
    # BrowserGym-specific convenience flags mapped to env vars
    benchmark: str = "miniwob",
    headless: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    timeout_ms: int = 10000,
):
    """
    Build a TRL-compatible rollout_func backed by a rollout processor (default: OpenEnvRolloutProcessor).

    Args:
        env_factory: Callable yielding an OpenEnv HTTPEnvClient instance. If None, a default
                     BrowserGym env factory is built using tasks/miniwob_url/docker_image.
        prompt_builder: (observation, step, history) -> Any content
            Content should be directly compatible with the LLM client
            (string or OpenAI-style content list/dict). The processor will
            not modify it.
        action_parser: (llm_response: str) -> env action object (e.g., BrowserGymAction)
        model: LLM model identifier
        max_steps: Maximum environment steps per rollout
        completion_params: Extra/override completion parameters to pass through
                           (e.g., {"temperature": 0.2, "top_p": 0.9, ...}).
                           These merge over defaults inferred from GRPO args.
        concurrency: Max number of concurrent rollouts. Defaults to
                     args.per_device_train_batch_size if not provided.
        processor_cls: Rollout processor class to instantiate. Defaults to OpenEnvRolloutProcessor.
        processor_kwargs: Extra kwargs forwarded to the processor constructor. These override any
                          automatically derived kwargs below.
        env_client_cls: Optional environment client class to instantiate (generic).
                        If provided and env_factory is None:
                          - If env_base_url is set: env_client_cls(base_url=..., request_timeout_s=..., default_headers=...)
                          - Else: env_client_cls.from_docker_image(docker_image, provider=..., **docker_kwargs)
        tasks: Optional list of BrowserGym task names to rotate over. If provided and env_factory is None,
               we will select one task per num_generations group.
        miniwob_url: MiniWoB base URL (e.g., "http://172.17.0.1:8888/miniwob") for containers.
        docker_image: Docker image to use for per-rollout BrowserGym containers.
        env_base_url: If provided, connect directly to an existing env server via HTTP.
        request_timeout_s: HTTP client timeout (seconds).
        default_headers: Default headers for HTTP requests (auth/trace).
        provider: Optional Docker provider to use for from_docker_image.
        docker_port: Optional host port binding override (provider-dependent).
        env_vars: Extra environment vars for the container; merged with BrowserGym defaults.
        benchmark: BrowserGym benchmark name ('miniwob', 'webarena', etc.) mapped to env var.
        headless: Headless mode mapped to env var.
        viewport_width/height: Browser viewport mapped to env vars.
        timeout_ms: Action timeout mapped to env var.

    Returns:
        rollout_func(prompts: List[str], args: GRPOConfig, processing_class) -> Dict[str, List]
    """

    def resolve_fireworks_model(model_str: str) -> str:
        """
        Resolve a Fireworks deployment resource to its active deployed model name.
        Accepts plain resource strings or LiteLLM-style prefixed ids, e.g.:
          - "accounts/<acct>/deployments/<id>"
          - "fireworks_ai/accounts/<acct>/deployments/<id>"
          - "fireworks_ai/accounts/fireworks/models/qwen3-8b#accounts/<acct>/deployments/<id>"
        Returns original string on any error or when resolution is not applicable.
        """
        try:
            if not isinstance(model_str, str) or not model_str:
                return model_str
            prefix = ""
            raw = model_str
            if model_str.startswith("fireworks_ai/"):
                prefix = "fireworks_ai/"
                raw = model_str[len(prefix):]
            m = re.search(r"(accounts/[^/\s]+/deployments/[^#\s]+)", raw)
            if not m and "#" in raw:
                right = raw.split("#", 1)[1]
                m = re.search(r"(accounts/[^/\s]+/deployments/[^#\s]+)", right)
            if not m:
                return model_str
            deployment_res = m.group(1)
            try:
                from fireworks.gateway import Gateway
                from fireworks.control_plane.generated.protos_grpcio.gateway.deployed_model_pb2 import (  # type: ignore
                    DeployedModel as SyncDeployedModel,
                    ListDeployedModelsRequest as SyncListDeployedModelsRequest,
                )
            except Exception:
                return model_str
            gateway = Gateway()
            req = SyncListDeployedModelsRequest(filter=f'deployment="{deployment_res}"')
            resp = gateway.list_deployed_models_sync(req)
            if getattr(resp, "total_size", 0) <= 0:
                return model_str
            deployed = resp.deployed_models[0]
            if getattr(deployed, "state", None) is not None:
                if deployed.state != SyncDeployedModel.DEPLOYED:
                    return model_str
            resolved_name = getattr(deployed, "name", None)
            if not resolved_name:
                return model_str
            return prefix + resolved_name if prefix else resolved_name
        except Exception:
            return model_str

    def rollout_func(prompts: List[str], args: GRPOConfig, processing_class) -> Dict[str, List]:
        # 1) Prompts → EvaluationRows (one per generation per prompt)
        num_generations = getattr(args, "num_generations", 8)
        evaluation_rows: List[EvaluationRow] = []
        # Build rows contiguous per prompt: for each prompt, add num_generations rows
        for prompt in prompts:
            for _ in range(num_generations):
                evaluation_rows.append(
                    EvaluationRow(
                        messages=[{"role": "user", "content": prompt}],
                        input_metadata=InputMetadata(
                            completion_params={"model": model}
                        ),
                    )
                )

        # 2) Build rollout config
        base_params: Dict[str, Any] = {
            "model": model,
            "temperature": getattr(args, "temperature", 0.0),
            "max_tokens": getattr(args, "max_completion_length", 100),
        }
        if completion_params:
            base_params.update(completion_params)

        max_concurrency = concurrency if concurrency is not None else getattr(args, "per_device_train_batch_size", 1)

        config = RolloutProcessorConfig(
            completion_params=base_params,
            mcp_config_path="",
            semaphore=asyncio.Semaphore(max_concurrency),
            steps=max_steps,
        )

        # 3) Execute rollouts
        # 3) Instantiate rollout processor (pluggable)
        Processor = processor_cls or OpenEnvRolloutProcessor  # type: ignore[assignment]
        _kwargs: Dict[str, Any] = dict(processor_kwargs or {})
        # If using OpenEnvRolloutProcessor (or compatible), supply env/prompt/action args unless overridden
        _kwargs.setdefault("env_factory", env_factory)
        _kwargs.setdefault("prompt_builder", prompt_builder)
        _kwargs.setdefault("action_parser", action_parser)
        # Environment args (only used by processors that support them)
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

        # 4) Convert EvaluationRows → TRL expected dict
        all_prompt_ids_per_row: List[List[int]] = []
        all_completion_ids: List[List[int]] = []
        all_logprobs: List[List[float]] = []
        step_rewards: List[List[float]] = []

        non_empty_rewards = 0
        total_rewards_sum = 0.0
        rows_with_rewards = 0

        # Prefer tokenizer on the processing_class if present
        tokenizer = getattr(processing_class, "tokenizer", None)
        if tokenizer is None:
            tokenizer = processing_class
        encode_fn = getattr(tokenizer, "encode", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = 0
        if encode_fn is None:
            pass

        for idx, row in enumerate(completed_rows):
            prompt_ids: List[int] = []
            completion_ids: List[int] = []
            logprobs: List[float] = []

            rewards: List[float] = []
            for msg in row.messages:
                if msg.role == "user":
                    tokens = encode_fn(msg.content or "") if encode_fn else []
                    prompt_ids.extend(tokens)
                elif msg.role == "assistant":
                    tokens = encode_fn(msg.content or "") if encode_fn else []
                    completion_ids.extend(tokens)
                    logprobs.extend([0.0] * len(tokens))  # placeholder
                elif msg.role == "system":
                    try:
                        content = msg.content or ""
                        if isinstance(content, str) and content.startswith("__ep_step_rewards__:"):
                            payload = content.split(":", 1)[1]
                            import json as _json
                            rewards = _json.loads(payload) or []
                    except Exception:
                        pass

            # Fallback to execution metadata (older processors)
            if not rewards:
                if hasattr(row.execution_metadata, "extra") and getattr(row.execution_metadata, "extra"):
                    try:
                        rewards = row.execution_metadata.extra.get("step_rewards", []) or []
                    except Exception:
                        rewards = []

            all_prompt_ids_per_row.append(prompt_ids if prompt_ids else [0])
            all_completion_ids.append(completion_ids if completion_ids else [eos_id])
            all_logprobs.append(logprobs if logprobs else [0.0])
            step_rewards.append(rewards if rewards else [0.0])

            if rewards:
                non_empty_rewards += 1
                rows_with_rewards += 1
                try:
                    total_rewards_sum += float(sum(rewards))
                except Exception:
                    pass

        if rows_with_rewards > 0:
            avg_sum = total_rewards_sum / rows_with_rewards
        else:
            avg_sum = 0.0

        # TRL expects 'prompt_ids' at the unique-prompt level in the vLLM-server path.
        # Our processor produced per-row (i.e., per-generation) entries; collapse to unique prompts.
        try:
            if num_generations > 0 and len(all_prompt_ids_per_row) % num_generations == 0:
                num_unique = len(all_prompt_ids_per_row) // num_generations
                prompt_ids_unique = [
                    all_prompt_ids_per_row[i * num_generations] for i in range(num_unique)
                ]
            else:
                # Fallback: de-duplicate while preserving order
                seen = set()
                prompt_ids_unique = []
                for p in all_prompt_ids_per_row:
                    t = tuple(p)
                    if t in seen:
                        continue
                    seen.add(t)
                    prompt_ids_unique.append(p)
        except Exception:
            prompt_ids_unique = all_prompt_ids_per_row

        return {
            "prompt_ids": prompt_ids_unique,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "step_rewards": step_rewards,
        }

    return rollout_func


