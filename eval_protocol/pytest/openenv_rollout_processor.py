"""
OpenEnv Rollout Processor

Generic processor for ANY OpenEnv environment using the standard HTTPEnvClient interface.
No environment-specific code - works with BrowserGym, Echo, TextArena, Atari, etc.

Key: OpenEnv provides a standard interface across all environments:
- All environments: HTTPEnvClient[ActionType, ObservationType]
- All have: reset() → StepResult, step(action) → StepResult, state() → State
- Client handles serialization/deserialization

This processor just calls env.reset(), env.step(), env.state() - that's it!
"""

import asyncio
import logging
import time
from itertools import count
from typing import List, Any, Dict, Callable, Generic, TypeVar, Optional, Type

from openai.types import CompletionUsage

from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

logger = logging.getLogger(__name__)


class OpenEnvRolloutProcessor(RolloutProcessor):
    """
    Generic rollout processor for ANY OpenEnv environment.

    Works with any environment that follows OpenEnv's standard interface:
    - HTTPEnvClient[ActionType, ObservationType]
    - reset() → StepResult[ObservationType]
    - step(action: ActionType) → StepResult[ObservationType]
    - state() → State

    No environment-specific code - just uses the standard interface!

    Examples:
        ```python
        # BrowserGym
        from envs.browsergym_env import BrowserGymEnv, BrowserGymAction
        def make_env():
            return BrowserGymEnv.from_docker_image(...)

        # Echo
        from envs.echo_env import EchoEnv, EchoAction
        def make_env():
            return EchoEnv.from_docker_image(...)

        # TextArena
        from envs.textarena_env import TextArenaEnv, TextArenaAction
        def make_env():
            return TextArenaEnv.from_docker_image(...)

        # Same processor works for all!
        processor = OpenEnvRolloutProcessor(
            env_factory=make_env,
            action_parser=lambda text: BrowserGymAction(action_str=text),  # or EchoAction(message=text), etc.
        )
        ```

    For TRL integration, see: trl-evalp/openenv_trl_integration.py
    """

    def __init__(
        self,
        env_factory: Optional[Callable] = None,
        prompt_builder: Callable[[Any, int, List[str]], Any] | None = None,
        action_parser: Callable[[str], Any] | None = None,
        *,
        # Policy parameter - NEW!
        policy_factory: Optional[Callable[..., Any]] = None,  # Factory to create policy from config
        # Environment construction parameters (generic HTTP client or Docker)
        env_client_cls: Optional[Type[Any]] = None,
        tasks: Optional[List[str]] = None,
        task_var: Optional[str] = None,
        miniwob_url: Optional[str] = None,
        docker_image: str = "browsergym-env:latest",
        env_base_url: Optional[str] = None,
        hub_repo_id: Optional[str] = None,
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
        num_generations: Optional[int] = None,
    ):
        """
        Initialize processor.

        Args:
            env_factory: Optional callable that creates an OpenEnv environment (HTTPEnvClient)
                        Example: lambda: BrowserGymEnv.from_docker_image(...). If not provided,
                        the processor will build one using the parameters below.
            prompt_builder: Optional function that builds the user message content from
                            (observation, step, history). It should return content
                            directly compatible with the LLM client (e.g., a string,
                            or OpenAI-style content list/dict). No additional processing
                            is performed by the processor.
            action_parser: Function that converts LLM text → Action object
                          Example: lambda text: BrowserGymAction(action_str=text)
                          Example: lambda text: EchoAction(message=text)
            env_client_cls: Optional environment HTTP client class (generic).
            tasks, task_var, miniwob_url, docker_image, env_base_url, request_timeout_s, default_headers,
            provider, docker_port, env_vars, benchmark, headless, viewport_*, timeout_ms:
                Parameters to construct default environments if env_factory is not provided.
            num_generations: Optional hint for task rotation grouping (used to mimic GRPO grouping).
        """
        self.prompt_builder = prompt_builder or (lambda obs, step, history: str(obs))
        if action_parser is None:
            raise ValueError("action_parser must be provided and return an Action object.")
        self.action_parser = action_parser
        self.policy_factory = policy_factory  # Store policy factory

        # Store env construction parameters
        self._provided_env_factory = env_factory
        self._env_client_cls = env_client_cls
        self._tasks = tasks or []
        self._task_var = task_var
        self._miniwob_url = miniwob_url
        self._docker_image = docker_image
        self._env_base_url = env_base_url
        self._hub_repo_id = hub_repo_id
        self._request_timeout_s = request_timeout_s
        self._default_headers = default_headers
        self._provider = provider
        self._docker_port = docker_port
        self._env_vars = {k: str(v) for k, v in (env_vars or {}).items()}
        self._benchmark = benchmark
        self._headless = headless
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._timeout_ms = timeout_ms
        self._num_generations = max(1, int(num_generations)) if num_generations else 1
        # Counter used for task rotation when creating environments. Uses
        # itertools.count to avoid race conditions across concurrent rollouts.
        self._env_create_counter = count()

        if self._tasks and not self._task_var:
            raise ValueError("task_var must be provided when tasks are configured.")

        # Build env_factory if not provided
        self.env_factory = self._build_env_factory()

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Process evaluation rows and return async tasks."""

        semaphore = config.semaphore
        max_steps = config.steps or 8

        logger.info("[OpenEnvRolloutProcessor] __call__ invoked with %d rows", len(rows))
        logger.info("[OpenEnvRolloutProcessor] Max steps: %d", max_steps)
        logger.debug(
            "[OpenEnvRolloutProcessor] Semaphore limit: %s",
            getattr(semaphore, "_value", "unknown"),
        )

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row with OpenEnv rollout."""
            start_time = time.perf_counter()

            logger.info("[OpenEnvRolloutProcessor] Starting rollout for row")

            # Create environment
            logger.debug("[OpenEnvRolloutProcessor] Creating environment via env_factory()")
            env = self.env_factory()
            logger.debug("[OpenEnvRolloutProcessor] Environment client created successfully")

            try:
                # Get model config
                raw_model = config.completion_params.get("model", "gpt-4o-mini")
                model = raw_model
                temperature = config.completion_params.get("temperature", 0.0)
                max_tokens = config.completion_params.get("max_tokens", 100)
                # Optional: direct routing or provider overrides (e.g., base_url, api_key, top_p, stop, etc.)
                base_url = config.completion_params.get("base_url")
                # Forward any extra completion params to LiteLLMPolicy (they will be sent per-request)
                extra_params: Dict[str, Any] = dict(config.completion_params or {})
                for _k in ("model", "temperature", "max_tokens", "base_url"):
                    try:
                        extra_params.pop(_k, None)
                    except Exception:
                        pass
                logger.info(
                    "[OpenEnvRolloutProcessor] Model='%s' temp=%s max_tokens=%s base_url=%s",
                    model,
                    temperature,
                    max_tokens,
                    base_url or "(default)",
                )

                # Create policy for generation
                if self.policy_factory is not None:
                    logger.debug("[OpenEnvRolloutProcessor] Creating policy using custom factory")
                    policy = self.policy_factory(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        base_url=base_url,
                        **extra_params,
                    )
                    logger.debug("[OpenEnvRolloutProcessor] Custom policy created successfully")
                else:
                    logger.debug("[OpenEnvRolloutProcessor] Creating LiteLLMPolicy (default)")
                    policy = LiteLLMPolicy(
                        model_id=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        base_url=base_url,
                        **extra_params,
                    )
                    logger.debug("[OpenEnvRolloutProcessor] LiteLLMPolicy created successfully")

                # Reset environment with simple transient-error retries
                reset_attempts = 3
                reset_delay = 1.0
                logger.debug("[OpenEnvRolloutProcessor] Resetting environment")
                result = None
                for i in range(reset_attempts):
                    try:
                        result = env.reset()
                        logger.debug("[OpenEnvRolloutProcessor] reset() succeeded on attempt %d", i + 1)
                        break
                    except Exception as e:
                        if i == reset_attempts - 1:
                            raise
                        time.sleep(reset_delay)
                        reset_delay *= 2.0

                if result is None:
                    raise RuntimeError("Failed to reset environment after all retry attempts")

                observation = result.observation
                logger.debug("[OpenEnvRolloutProcessor] Initial observation received")

                # Initialize tracking
                messages = list(row.messages)  # Copy initial messages
                # Inject system prompt if provided and not already present
                has_system = any(m.role == "system" for m in messages)
                system_prompt = config.completion_params.get("system_prompt")
                if system_prompt and not has_system:
                    messages.insert(0, Message(role="system", content=system_prompt))
                usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                step_rewards = []
                history: List[str] = []
                # Accumulate token IDs across all turns for training integrations
                all_prompt_ids: List[int] = []
                all_completion_ids: List[int] = []

                logger.info("[OpenEnvRolloutProcessor] Starting agent loop (max %d steps)", max_steps)

                # Agent loop: model → action → env.step → repeat
                for step in range(max_steps):
                    logger.debug("[OpenEnvRolloutProcessor] === STEP %d/%d ===", step + 1, max_steps)

                    if result.done:
                        logger.info(f"Episode done after {step} steps")
                        logger.info("[OpenEnvRolloutProcessor] Episode already done at step %d", step)
                        break

                    # Build user message content via user-provided prompt_builder
                    try:
                        logger.debug("[OpenEnvRolloutProcessor] Building prompt")
                        user_content = self.prompt_builder(observation, step + 1, history)
                        logger.debug(
                            "[OpenEnvRolloutProcessor] Prompt built (len=%d)",
                            len(str(user_content)),
                        )
                    except Exception as e:
                        logger.error(f"prompt_builder failed: {e}", exc_info=True)
                        user_content = str(observation)

                    messages.append(Message(role="user", content=user_content))
                    # Optional tracing
                    if getattr(config, "logger", None):
                        try:
                            # Log a snapshot with current messages so UI shows incremental turns
                            try:
                                row_for_log = row.model_copy(deep=True)  # pydantic v2
                            except Exception:
                                import copy as _copy

                                row_for_log = _copy.deepcopy(row)
                            row_for_log.messages = list(messages)
                            config.logger.log(row_for_log)
                        except Exception:
                            pass

                    # Call model to generate action (LiteLLM or custom policy)
                    logger.debug("[OpenEnvRolloutProcessor] Calling LLM (messages=%d)", len(messages))
                    response = await policy._make_llm_call(
                        messages=[msg.model_dump() for msg in messages],
                        tools=[],  # No tools - just text generation
                    )
                    logger.debug("[OpenEnvRolloutProcessor] LLM call completed")

                    # Update usage
                    usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
                    usage["completion_tokens"] += response["usage"]["completion_tokens"]
                    usage["total_tokens"] += response["usage"]["total_tokens"]
                    logger.debug(
                        "[OpenEnvRolloutProcessor] Tokens: prompt=%s, completion=%s",
                        response["usage"]["prompt_tokens"],
                        response["usage"]["completion_tokens"],
                    )

                    # Extract assistant message and parse into Action object
                    assistant_message = response["choices"][0]["message"]["content"]
                    preview = assistant_message if isinstance(assistant_message, str) else str(assistant_message)
                    logger.debug(
                        "[OpenEnvRolloutProcessor] Model output: '%s'",
                        preview[:120] if preview else "",
                    )

                    logger.debug("[OpenEnvRolloutProcessor] Parsing action")
                    action = self.action_parser(assistant_message)
                    label = getattr(action, "action_str", None) or str(action)
                    logger.debug("[OpenEnvRolloutProcessor] Parsed action: '%s'", label[:120])

                    # Add assistant message (original content)
                    messages.append(Message(role="assistant", content=assistant_message))

                    # Accumulate token IDs from this turn for downstream training
                    if "prompt_ids" in response and "completion_ids" in response:
                        try:
                            all_prompt_ids.extend(response["prompt_ids"])
                            all_completion_ids.extend(response["completion_ids"])
                        except Exception:
                            # Best-effort only; don't break rollouts if tokens are malformed
                            pass

                    # Execute action in environment (OpenEnv standard interface!) with transient-error retries
                    logger.debug("[OpenEnvRolloutProcessor] Executing action in environment")
                    step_attempts = 2
                    step_delay = 0.5
                    for si in range(step_attempts):
                        try:
                            result = env.step(action)
                            logger.debug("[OpenEnvRolloutProcessor] env.step() succeeded")
                            break
                        except Exception as se:
                            if si == step_attempts - 1:
                                logger.error(
                                    "[OpenEnvRolloutProcessor] env.step() failed after %d attempts: %s",
                                    step_attempts,
                                    se,
                                )
                                raise
                            time.sleep(step_delay)

                    # Collect reward (OpenEnv standard: result.reward)
                    reward = float(result.reward or 0.0)
                    step_rewards.append(reward)
                    logger.debug(
                        "[OpenEnvRolloutProcessor] Step %d: reward=%.3f, done=%s",
                        step + 1,
                        reward,
                        result.done,
                    )

                    _action_label = getattr(action, "action_str", None)
                    if not _action_label:
                        try:
                            _action_label = str(action)
                        except Exception:
                            _action_label = "<action>"
                    logger.debug(f"Step {step}: action={_action_label}, reward={reward}")

                    # Update observation (OpenEnv standard: result.observation)
                    observation = result.observation

                    # Update history for next prompt
                    error_flag = getattr(observation, "last_action_error", False)
                    history_line = (
                        f"Step {step + 1}: {_action_label} -> reward {reward:+.2f}{' ERROR' if error_flag else ''}"
                    )
                    history.append(history_line)
                    # Optional tracing
                    if getattr(config, "logger", None):
                        try:
                            # Log a snapshot with current messages so UI shows incremental turns
                            try:
                                row_for_log = row.model_copy(deep=True)  # pydantic v2
                            except Exception:
                                import copy as _copy

                                row_for_log = _copy.deepcopy(row)
                            row_for_log.messages = list(messages)
                            config.logger.log(row_for_log)
                        except Exception:
                            pass

                # Update row with results
                row.messages = messages
                row.execution_metadata.usage = CompletionUsage(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                )
                row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time

                # Attach per-step rewards and accumulated token IDs to
                # execution_metadata.extra for downstream integrations
                # (for example, TRL GRPO) instead of encoding them into
                # synthetic system messages.
                try:
                    extra = getattr(row.execution_metadata, "extra", None)
                    if not isinstance(extra, dict):
                        extra = {}
                    extra["step_rewards"] = list(step_rewards)
                    if all_prompt_ids or all_completion_ids:
                        extra["prompt_ids"] = list(all_prompt_ids)
                        extra["completion_ids"] = list(all_completion_ids)
                    row.execution_metadata.extra = extra  # type: ignore[attr-defined]
                except Exception:
                    # Non-fatal: callers can fall back if metadata is missing
                    pass

                total_reward = sum(step_rewards)
                logger.info("[OpenEnvRolloutProcessor] ✅ ROLLOUT COMPLETE")
                logger.info("[OpenEnvRolloutProcessor] Steps: %d", len(step_rewards))
                logger.info("[OpenEnvRolloutProcessor] Total reward: %.3f", total_reward)
                logger.info(
                    "[OpenEnvRolloutProcessor] Duration: %.2fs",
                    row.execution_metadata.rollout_duration_seconds,
                )
                logger.debug("[OpenEnvRolloutProcessor] Messages collected: %d", len(messages))

                logger.info(
                    f"Rollout complete: {len(step_rewards)} steps, "
                    f"total_reward={total_reward:.2f}, "
                    f"duration={row.execution_metadata.rollout_duration_seconds:.2f}s"
                )
                # Final log with complete message history
                if getattr(config, "logger", None):
                    try:
                        config.logger.log(row)
                    except Exception:
                        pass

                return row

            except Exception as e:
                logger.error(f"Error in rollout: {e}", exc_info=True)
                logger.error(
                    "[OpenEnvRolloutProcessor] ❌ ERROR in rollout: %s: %s",
                    type(e).__name__,
                    e,
                )
                raise
            finally:
                # Cleanup environment
                logger.debug("[OpenEnvRolloutProcessor] Closing environment client")
                try:
                    env.close()
                    logger.debug("[OpenEnvRolloutProcessor] Environment closed successfully")
                except Exception as close_err:
                    logger.warning(
                        "[OpenEnvRolloutProcessor] Error closing environment: %s",
                        close_err,
                    )

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await process_row(r)

        # Create and return tasks
        logger.debug("[OpenEnvRolloutProcessor] Creating %d async tasks", len(rows))
        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        logger.debug("[OpenEnvRolloutProcessor] Returning %d tasks", len(tasks))
        return tasks

    def _build_prompt(self, observation_text: str, step: int) -> str:
        """
        Build prompt for LLM from observation text.

        Generic prompt that works for any environment.
        """
        return (
            f"Step {step + 1}\n\n"
            f"Observation:\n{observation_text}\n\n"
            f"What action should be taken? Respond with a single action."
        )

    # Removed _extract_action_text: action parsing handled entirely by action_parser

    def _build_env_factory(self) -> Callable[[], Any]:
        """
        Create or return an environment factory based on the provided parameters.
        Preference order:
          1) Use provided env_factory
          2) Use generic env_client_cls with task-aware env vars (BrowserGym-style)
        """
        if self._provided_env_factory is not None:
            return self._provided_env_factory

        # If a generic client class is provided, use it
        if self._env_client_cls is not None:

            def _generic_factory():
                if self._env_base_url:
                    logger.debug(
                        "[OpenEnvRolloutProcessor] Using env_client_cls base_url=%s",
                        self._env_base_url,
                    )
                    return self._env_client_cls(  # type: ignore[call-arg]
                        base_url=self._env_base_url,
                        request_timeout_s=self._request_timeout_s,
                        default_headers=self._default_headers,
                    )

                # ------------------------------
                # Docker-based env: build env_vars with task rotation
                # ------------------------------
                docker_kwargs: Dict[str, Any] = {}

                env_vars_default: Dict[str, str] = dict(self._env_vars)

                # Select task for this env instance (if provided), grouped by num_generations
                selected_task: Optional[str] = None
                if self._tasks:
                    # Use a monotonic counter so concurrent environment creation
                    # does not reuse the same index across rollouts.
                    idx = next(self._env_create_counter)
                    group = idx // max(1, self._num_generations)
                    selected_task = self._tasks[group % len(self._tasks)]
                    if not self._task_var:
                        raise ValueError("task_var must be provided when tasks are configured.")
                    env_vars_default[self._task_var] = str(selected_task)
                    logger.debug(
                        "[OpenEnvRolloutProcessor] Task selection: idx=%d, group=%d, num_generations=%d, selected_task=%s, tasks=%s",
                        idx,
                        group,
                        self._num_generations,
                        selected_task,
                        self._tasks,
                    )

                if env_vars_default:
                    docker_kwargs["env_vars"] = env_vars_default

                if self._docker_port is not None:
                    docker_kwargs["port"] = int(self._docker_port)
                if self._hub_repo_id:
                    logger.debug(
                        "[OpenEnvRolloutProcessor] Launching from_hub repo_id='%s' ...",
                        self._hub_repo_id,
                    )
                    return self._env_client_cls.from_hub(  # type: ignore[attr-defined]
                        self._hub_repo_id,
                        provider=self._provider,
                        **docker_kwargs,
                    )
                else:
                    logger.debug(
                        "[OpenEnvRolloutProcessor] Launching from_docker_image image='%s' ...",
                        self._docker_image,
                    )
                    return self._env_client_cls.from_docker_image(  # type: ignore[attr-defined]
                        self._docker_image,
                        provider=self._provider,
                        **docker_kwargs,
                    )

            return _generic_factory

        # No fallback: require an env_factory or env_client_cls
        raise RuntimeError(
            "OpenEnvRolloutProcessor requires either env_factory or env_client_cls. "
            "Provide one of these to construct the environment."
        )
