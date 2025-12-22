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
import importlib
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, cast

from eval_protocol.models import EvalMetadata, EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig


logger = logging.getLogger(__name__)


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
    env_path: Optional[str] = None,
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

    ``GRPOTrainer`` calls the returned ``rollout_func(prompts, trainer)``.
    For each prompt we run one OpenEnv episode using ``OpenEnvRolloutProcessor``
    and return Wordle-style GRPO data (2D token lists + 1D rewards).
    """
    logger.info("create_openenv_vllm_rollout_func called")
    logger.debug(
        "vllm_base_url=%s, vllm_model=%s, tasks=%s, max_steps=%s",
        vllm_base_url,
        vllm_model,
        tasks,
        max_steps,
    )

    # Import VLLMPolicy
    from eval_protocol.mcp.execution.vllm_policy import VLLMPolicy

    # Global-ish task rotation offset across rollout_func calls.
    # This lets us rotate tasks between GRPO steps instead of always
    # starting from tasks[0] when a new OpenEnvRolloutProcessor is created.
    task_cycle_index: int = 0

    def rollout_func(prompts: List[str], trainer) -> Dict[str, List]:
        """Execute rollouts via OpenEnv + vLLM and return GRPO-compatible results."""
        logger.info("OpenEnv vLLM rollout_func called")

        # Extract args from trainer
        args = trainer.args
        processing_class = trainer.processing_class

        num_generations = getattr(args, "num_generations", 8)
        eval_name = env_path or "openenv_browsergym_vllm_training"
        logger.info(
            "[OpenEnvVLLM] Received %d prompts (trainer.num_generations=%s)",
            len(prompts),
            num_generations,
        )
        logger.debug("[OpenEnvVLLM] Total rollouts to execute: %d", len(prompts))

        # Optionally load rollout processor + eval function hints from an
        # @evaluation_test via env_path
        ep_rollout_processor = None
        ep_rollout_processor_kwargs: Dict[str, Any] = {}
        ep_mcp_config_path = ""
        ep_eval_func = None

        if env_path:
            logger.info("[OpenEnvVLLM] Loading evaluation test from env_path='%s'", env_path)
            try:
                module = importlib.import_module(env_path)
            except Exception as e:
                raise ImportError(f"Failed to import env module '{env_path}': {e}") from e

            candidate_tests = [
                obj for _, obj in inspect.getmembers(module) if callable(obj) and hasattr(obj, "__ep_params__")
            ]
            if not candidate_tests:
                raise ValueError(f"No @evaluation_test functions found in '{env_path}'.")

            eval_func = candidate_tests[0]
            ep_eval_func = eval_func  # used later after rollouts complete
            ep_params = getattr(eval_func, "__ep_params__", None)
            # ep_params is an EPParameters model (Pydantic), use attribute access
            ep_rollout_processor = getattr(ep_params, "rollout_processor", None) if ep_params else None
            ep_rollout_processor_kwargs = (
                (getattr(ep_params, "rollout_processor_kwargs", None) or {}) if ep_params else {}
            )
            ep_mcp_config_path = (getattr(ep_params, "mcp_config_path", None) or "") if ep_params else ""
            logger.info(
                "[OpenEnvVLLM] Loaded eval test '%s' with rollout_processor=%s",
                getattr(eval_func, "__name__", str(eval_func)),
                type(ep_rollout_processor).__name__,
            )

        # 1) Build evaluation rows with rollout_id for logging
        import uuid

        evaluation_rows: List[EvaluationRow] = []
        for prompt_idx, prompt in enumerate(prompts):
            # One evaluation row per incoming prompt. GRPOTrainer will handle
            # grouping by `num_generations` at the trainer level; the custom
            # rollout_func must return one set of tokens per prompt.
            rollout_id = f"openenv_vllm_{uuid.uuid4().hex[:12]}"

            row = EvaluationRow(
                messages=[Message(role="user", content=prompt)],
                input_metadata=InputMetadata(
                    # Let Eval Protocol generate a stable row_id from content.
                    row_id=None,
                    completion_params={},
                ),
            )
            row.execution_metadata.rollout_id = rollout_id  # Required for ep logs!

            # Minimal eval_metadata so ep logs can group/display properly
            row.eval_metadata = EvalMetadata(
                name=eval_name,
                description=None,
                version="v1",
                status=None,
                num_runs=1,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            )

            evaluation_rows.append(row)

        logger.debug(
            "[OpenEnvVLLM] Created %d evaluation rows with rollout_ids and row_ids",
            len(evaluation_rows),
        )

        # 2) Build processor config with VLLMPolicy
        # We'll pass trainer.vllm_client to VLLMPolicy
        base_params: Dict[str, Any] = {
            "model": "dummy",  # Not used by VLLMPolicy, but needed for config
            "temperature": getattr(args, "temperature", 1.0),
            "max_tokens": getattr(args, "max_completion_length", 100),
        }
        if completion_params:
            base_params.update(completion_params)

        logger.debug(
            "[OpenEnvVLLM] Temperature=%s, max_tokens=%s",
            base_params["temperature"],
            base_params["max_tokens"],
        )
        logger.debug("[OpenEnvVLLM] Using TRL VLLMClient from trainer")

        max_concurrency = concurrency if concurrency is not None else getattr(args, "per_device_train_batch_size", 1)
        logger.debug(
            "[OpenEnvVLLM] Max concurrency=%s, max_steps=%s",
            max_concurrency,
            max_steps,
        )

        # Import default logger for local tracing
        from eval_protocol.dataset_logger import default_logger

        config = RolloutProcessorConfig(
            completion_params=base_params,
            mcp_config_path=ep_mcp_config_path or "",
            semaphore=asyncio.Semaphore(max_concurrency),
            steps=max_steps,
            logger=default_logger,
            kwargs=ep_rollout_processor_kwargs,
        )

        # 3) Execute rollouts with VLLMPolicy
        logger.debug(
            "[OpenEnvVLLM] Instantiating processor: %s",
            processor_cls.__name__ if processor_cls else "OpenEnvRolloutProcessor",
        )

        # Create policy factory that uses trainer's vllm_client or llm
        def vllm_policy_factory(model, temperature, max_tokens, base_url=None, **kwargs):
            """Factory that creates VLLMPolicy using trainer's vllm_client or llm."""
            logger.debug(
                "[VLLMPolicyFactory] Creating VLLMPolicy with temp=%s, max_tokens=%s",
                temperature,
                max_tokens,
            )
            # Check for vllm_client (server mode) or llm (colocate mode)
            vllm_client = getattr(trainer, "vllm_client", None) or getattr(trainer, "llm", None)
            if vllm_client is None:
                raise RuntimeError("Trainer has neither vllm_client (server mode) nor llm (colocate mode)")

            return VLLMPolicy(
                vllm_client=vllm_client,  # Use trainer's vLLM client!
                tokenizer=processing_class,  # Pass tokenizer for decoding
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get("top_p"),
                top_k=kwargs.get("top_k"),
                **kwargs,
            )

        Processor = processor_cls or OpenEnvRolloutProcessor
        _kwargs: Dict[str, Any] = dict(processor_kwargs or {})

        # If env_path was provided and we found an OpenEnvRolloutProcessor in the
        # evaluation test, seed processor kwargs from it so users can reuse the
        # same environment configuration for training.
        if env_path and isinstance(ep_rollout_processor, OpenEnvRolloutProcessor):
            logger.debug(
                "[OpenEnvVLLM] Seeding processor kwargs from evaluation_test rollout_processor",
            )
            _kwargs.setdefault("env_factory", getattr(ep_rollout_processor, "_provided_env_factory", None))
            _kwargs.setdefault("env_client_cls", getattr(ep_rollout_processor, "_env_client_cls", None))
            _kwargs.setdefault("tasks", getattr(ep_rollout_processor, "_tasks", None))
            _kwargs.setdefault("task_var", getattr(ep_rollout_processor, "_task_var", None))
            _kwargs.setdefault("miniwob_url", getattr(ep_rollout_processor, "_miniwob_url", None))
            _kwargs.setdefault("docker_image", getattr(ep_rollout_processor, "_docker_image", None))
            _kwargs.setdefault("env_base_url", getattr(ep_rollout_processor, "_env_base_url", None))
            _kwargs.setdefault(
                "request_timeout_s",
                getattr(ep_rollout_processor, "_request_timeout_s", None),
            )
            _kwargs.setdefault(
                "default_headers",
                getattr(ep_rollout_processor, "_default_headers", None),
            )
            _kwargs.setdefault("provider", getattr(ep_rollout_processor, "_provider", None))
            _kwargs.setdefault("docker_port", getattr(ep_rollout_processor, "_docker_port", None))
            _kwargs.setdefault("env_vars", getattr(ep_rollout_processor, "_env_vars", None))
            _kwargs.setdefault("benchmark", getattr(ep_rollout_processor, "_benchmark", None))
            _kwargs.setdefault("headless", getattr(ep_rollout_processor, "_headless", None))
            _kwargs.setdefault(
                "viewport_width",
                getattr(ep_rollout_processor, "_viewport_width", None),
            )
            _kwargs.setdefault(
                "viewport_height",
                getattr(ep_rollout_processor, "_viewport_height", None),
            )
            _kwargs.setdefault("timeout_ms", getattr(ep_rollout_processor, "_timeout_ms", None))
            _kwargs.setdefault(
                "num_generations",
                getattr(ep_rollout_processor, "_num_generations", None),
            )

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
            logger.debug(
                "[OpenEnvVLLM] Task rotation offset=%s, rotated=%s",
                offset,
                rotated_tasks,
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
        logger.debug("[OpenEnvVLLM] Processor instantiated successfully")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:

            async def _run_all() -> List[EvaluationRow]:
                tasks_list: List[asyncio.Task[EvaluationRow]] = processor(evaluation_rows, config)
                rows: List[EvaluationRow] = await asyncio.gather(*tasks_list)

                # Optionally run the @evaluation_test function on each row to
                # populate evaluation_result (score/metrics) so the same
                # reward logic can be reused across trainers.
                if env_path and ep_eval_func is not None:
                    if inspect.iscoroutinefunction(ep_eval_func):
                        eval_tasks = [ep_eval_func(row) for row in rows]
                        rows = cast(List[EvaluationRow], await asyncio.gather(*eval_tasks))
                    else:
                        rows = cast(List[EvaluationRow], [ep_eval_func(row) for row in rows])
                    logger.info(
                        "[OpenEnvVLLM] Applied eval function to %d rows from env_path='%s'",
                        len(rows),
                        env_path,
                    )

                return rows

            completed_rows: List[EvaluationRow] = loop.run_until_complete(_run_all())
            logger.info(
                "[OpenEnvVLLM] All rollouts completed: %d results",
                len(completed_rows),
            )
        finally:
            loop.close()

        # 4) Convert completed rows to TRL format (one episode per row)
        logger.info(
            "[OpenEnvVLLM] Converting %d completed rollouts to TRL format",
            len(completed_rows),
        )

        tokenizer = getattr(processing_class, "tokenizer", None) or processing_class

        episode_prompt_ids: List[List[int]] = []
        episode_completion_ids: List[List[int]] = []
        episode_logprobs: List[List[float]] = []
        step_rewards_all: List[List[float]] = []
        eval_scores: List[float] = []

        for idx, row in enumerate(completed_rows):
            logger.debug(
                "[OpenEnvVLLM] Processing rollout %d/%d: %d messages",
                idx + 1,
                len(completed_rows),
                len(row.messages),
            )

            # Prefer raw token IDs stored by the rollout processor in
            # execution_metadata.extra to avoid any re-encoding.
            prompt_ids: List[int] = []
            completion_ids: List[int] = []
            logprobs: List[float] = []  # We don't currently track per-token logprobs
            rewards: List[float] = []

            try:
                extra = getattr(row.execution_metadata, "extra", None)
                if isinstance(extra, dict):
                    prompt_ids = list(extra.get("prompt_ids", []) or [])
                    completion_ids = list(extra.get("completion_ids", []) or [])
                    rewards = [float(r) for r in (extra.get("step_rewards", []) or [])]
            except Exception:
                prompt_ids = []
                completion_ids = []
                rewards = []

            # Append accumulated tokens for this episode
            episode_prompt_ids.append(prompt_ids if prompt_ids else [0])
            episode_completion_ids.append(completion_ids if completion_ids else [0])
            episode_logprobs.append(logprobs if logprobs else [0.0])
            step_rewards_all.append(rewards if rewards else [0.0])

            # Also capture evaluation_result.score if the evaluation_test
            # populated it, so downstream trainers can reuse the exact same
            # scoring logic as the eval harness.
            score_val = 0.0
            try:
                if getattr(row, "evaluation_result", None) is not None:
                    score_attr = getattr(row.evaluation_result, "score", None)
                    if score_attr is not None:
                        score_val = float(score_attr)
            except Exception:
                score_val = 0.0
            eval_scores.append(score_val)

            ep_reward = sum(rewards) if rewards else 0.0
            logger.debug(
                "[OpenEnvVLLM] Episode %d: prompt_tokens=%d, completion_tokens=%d, reward=%.3f",
                idx + 1,
                len(prompt_ids),
                len(completion_ids),
                ep_reward,
            )

        total_reward = sum(sum(r) for r in step_rewards_all)
        avg_reward = total_reward / len(step_rewards_all) if step_rewards_all else 0.0
        logger.info(
            "[OpenEnvVLLM] ✅ All rollouts complete | total_reward=%.2f, avg_reward=%.2f",
            total_reward,
            avg_reward,
        )
        logger.info(
            "[OpenEnvVLLM] Returning %d episodes to GRPO",
            len(episode_prompt_ids),
        )

        # Return in Wordle format
        # Tokens: 2D arrays (accumulate across turns, one list per episode)
        # Rewards: 1D arrays (one scalar per episode)
        total_rewards = [sum(r) for r in step_rewards_all]  # Sum step rewards per episode

        logger.debug("[OpenEnvVLLM] Episode rewards: %s", total_rewards)

        # Validate token IDs before returning (sanity check only)
        vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 200000
        logger.debug("[OpenEnvVLLM] Validating token IDs (vocab_size=%s)...", vocab_size)
        for i, (pids, cids) in enumerate(zip(episode_prompt_ids, episode_completion_ids)):
            max_p = max(pids) if pids else 0
            max_c = max(cids) if cids else 0
            if max_p >= vocab_size or max_c >= vocab_size:
                logger.warning(
                    "[OpenEnvVLLM] Episode %d: INVALID TOKEN IDS (max_prompt_id=%s, max_completion_id=%s)",
                    i,
                    max_p,
                    max_c,
                )
            logger.debug(
                "[OpenEnvVLLM] Episode %d: prompt_len=%d, completion_len=%d, max_p_id=%d, max_c_id=%d",
                i,
                len(pids),
                len(cids),
                max_p,
                max_c,
            )

        return {
            "prompt_ids": episode_prompt_ids,  # List[List[int]] - tokens per episode
            "completion_ids": episode_completion_ids,  # List[List[int]] - tokens per episode
            "logprobs": episode_logprobs,  # List[List[float]] - logprobs per episode
            "eval_score": eval_scores,
        }

    logger.debug("[openenv_trl_vllm] Returning rollout_func (type=%s)", type(rollout_func))
    return rollout_func
