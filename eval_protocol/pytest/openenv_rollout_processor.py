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
from typing import List, Any, Dict, Callable, Generic, TypeVar, Optional, Type
import json

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
        policy_factory: Optional[Callable] = None,  # Factory to create policy from config
        # Environment construction parameters (generic HTTP client or Docker)
        env_client_cls: Optional[Type[Any]] = None,
        tasks: Optional[List[str]] = None,
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
            tasks, miniwob_url, docker_image, env_base_url, request_timeout_s, default_headers,
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
        self._miniwob_url = miniwob_url
        self._docker_image = docker_image
        self._env_base_url = env_base_url
        self._hub_repo_id = hub_repo_id
        self._request_timeout_s = request_timeout_s
        self._default_headers = default_headers
        self._provider = provider
        self._docker_port = docker_port
        self._env_vars = env_vars or {}
        self._benchmark = benchmark
        self._headless = headless
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._timeout_ms = timeout_ms
        self._num_generations = max(1, int(num_generations)) if num_generations else 1
        self._env_create_idx: int = 0

        # Build env_factory if not provided
        self.env_factory = self._build_env_factory()
    
    def __call__(
        self, rows: List[EvaluationRow], config: RolloutProcessorConfig
    ) -> List[asyncio.Task[EvaluationRow]]:
        """Process evaluation rows and return async tasks."""
        
        semaphore = config.semaphore
        max_steps = config.steps or 8
        
        print(f"\n[OpenEnvRolloutProcessor] __call__ invoked with {len(rows)} rows", flush=True)
        print(f"[OpenEnvRolloutProcessor] Max steps: {max_steps}", flush=True)
        print(f"[OpenEnvRolloutProcessor] Semaphore limit: {semaphore._value if hasattr(semaphore, '_value') else 'unknown'}", flush=True)
        
        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row with OpenEnv rollout."""
            start_time = time.perf_counter()
            
            print(f"\n[OpenEnvRolloutProcessor] Starting rollout for row...", flush=True)
            
            # Create environment
            print("[OpenEnvRolloutProcessor] Creating environment via env_factory()...", flush=True)
            env = self.env_factory()
            print("[OpenEnvRolloutProcessor] Environment client created successfully.", flush=True)
            
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
                print(f"[OpenEnvRolloutProcessor] Model='{model}' temp={temperature} max_tokens={max_tokens} base_url={base_url or '(default)'}", flush=True)
                
                # Create policy for generation
                if self.policy_factory is not None:
                    print(f"[OpenEnvRolloutProcessor] Creating policy using custom factory...", flush=True)
                    policy = self.policy_factory(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        base_url=base_url,
                        **extra_params,
                    )
                    print(f"[OpenEnvRolloutProcessor] Custom policy created successfully", flush=True)
                else:
                    print(f"[OpenEnvRolloutProcessor] Creating LiteLLMPolicy (default)...", flush=True)
                    policy = LiteLLMPolicy(
                        model_id=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        base_url=base_url,
                        **extra_params,
                    )
                    print(f"[OpenEnvRolloutProcessor] LiteLLMPolicy created successfully", flush=True)
                
                # Reset environment with simple transient-error retries
                reset_attempts = 3
                reset_delay = 1.0
                last_exc = None
                print("[OpenEnvRolloutProcessor] Resetting environment...", flush=True)
                for i in range(reset_attempts):
                    try:
                        result = env.reset()
                        print(f"[OpenEnvRolloutProcessor] reset() succeeded on attempt {i + 1}", flush=True)
                        break
                    except Exception as e:
                        last_exc = e
                        if i == reset_attempts - 1:
                            raise
                        time.sleep(reset_delay)
                        reset_delay *= 2.0
                observation = result.observation
                print(f"[OpenEnvRolloutProcessor] Initial observation received", flush=True)
                
                
                # Initialize tracking
                messages = list(row.messages)  # Copy initial messages
                # Inject system prompt if provided and not already present
                try:
                    has_system = any(m.role == "system" for m in messages)
                except Exception:
                    has_system = False
                system_prompt = None
                try:
                    system_prompt = config.completion_params.get("system_prompt")
                except Exception:
                    system_prompt = None
                if system_prompt and not has_system:
                    messages.insert(0, Message(role="system", content=system_prompt))
                usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                step_rewards = []
                history: List[str] = []
                
                print(f"[OpenEnvRolloutProcessor] Starting agent loop (max {max_steps} steps)", flush=True)
                
                # Agent loop: model → action → env.step → repeat
                for step in range(max_steps):
                    print(f"\n[OpenEnvRolloutProcessor] === STEP {step + 1}/{max_steps} ===", flush=True)
                    
                    if result.done:
                        logger.info(f"Episode done after {step} steps")
                        print(f"[OpenEnvRolloutProcessor] Episode already done at step {step}", flush=True)
                        break
                    
                    # Build user message content via user-provided prompt_builder
                    try:
                        print(f"[OpenEnvRolloutProcessor] Building prompt...", flush=True)
                        user_content = self.prompt_builder(observation, step + 1, history)
                        print(f"[OpenEnvRolloutProcessor] Prompt built (len={len(str(user_content))})", flush=True)
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
                    
                    # Call model to generate action (LiteLLM handles multimodal!)
                    print(f"[OpenEnvRolloutProcessor] Calling LLM (messages={len(messages)})...", flush=True)
                    response = await policy._make_llm_call(
                        messages=[msg.model_dump() for msg in messages],
                        tools=None,  # No tools - just text generation
                    )
                    print(f"[OpenEnvRolloutProcessor] LLM call completed", flush=True)
                    
                    # Update usage
                    usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
                    usage["completion_tokens"] += response["usage"]["completion_tokens"]
                    usage["total_tokens"] += response["usage"]["total_tokens"]
                    print(f"[OpenEnvRolloutProcessor] Tokens: prompt={response['usage']['prompt_tokens']}, completion={response['usage']['completion_tokens']}", flush=True)
                    
                    # Extract assistant message and parse into Action object
                    assistant_message = response["choices"][0]["message"]["content"]
                    preview = assistant_message if isinstance(assistant_message, str) else str(assistant_message)
                    print(f"[OpenEnvRolloutProcessor] Model output: '{preview[:120] if preview else ''}'", flush=True)
                    
                    print(f"[OpenEnvRolloutProcessor] Parsing action...", flush=True)
                    action = self.action_parser(assistant_message)
                    label = getattr(action, "action_str", None) or str(action)
                    print(f"[OpenEnvRolloutProcessor] Parsed action: '{label[:120]}'", flush=True)
                    
                    # Add assistant message (original content)
                    messages.append(Message(role="assistant", content=assistant_message))
                    
                    # Execute action in environment (OpenEnv standard interface!) with transient-error retries
                    print(f"[OpenEnvRolloutProcessor] Executing action in environment...", flush=True)
                    step_attempts = 2
                    step_delay = 0.5
                    for si in range(step_attempts):
                        try:
                            result = env.step(action)
                            print(f"[OpenEnvRolloutProcessor] env.step() succeeded", flush=True)
                            break
                        except Exception as se:
                            if si == step_attempts - 1:
                                print(f"[OpenEnvRolloutProcessor] env.step() failed after {step_attempts} attempts: {se}", flush=True)
                                raise
                            time.sleep(step_delay)
                    
                    # Collect reward (OpenEnv standard: result.reward)
                    reward = float(result.reward or 0.0)
                    step_rewards.append(reward)
                    print(f"[OpenEnvRolloutProcessor] Step {step + 1}: reward={reward:.3f}, done={result.done}", flush=True)
                    
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
                    history_line = f"Step {step + 1}: {_action_label} -> reward {reward:+.2f}{' ERROR' if error_flag else ''}"
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
                row.execution_metadata.duration_seconds = time.perf_counter() - start_time
                
                # Store rewards for TRL reward functions via a system message sentinel
                sentinel = "__ep_step_rewards__:" + json.dumps(step_rewards)
                messages.append(Message(role="system", content=sentinel))
                
                total_reward = sum(step_rewards)
                print(f"\n[OpenEnvRolloutProcessor] ✅ ROLLOUT COMPLETE", flush=True)
                print(f"[OpenEnvRolloutProcessor] Steps: {len(step_rewards)}", flush=True)
                print(f"[OpenEnvRolloutProcessor] Total reward: {total_reward:.3f}", flush=True)
                print(f"[OpenEnvRolloutProcessor] Duration: {row.execution_metadata.duration_seconds:.2f}s", flush=True)
                print(f"[OpenEnvRolloutProcessor] Messages collected: {len(messages)}", flush=True)
                
                logger.info(
                    f"Rollout complete: {len(step_rewards)} steps, "
                    f"total_reward={total_reward:.2f}, "
                    f"duration={row.execution_metadata.duration_seconds:.2f}s"
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
                print(f"\n[OpenEnvRolloutProcessor] ❌ ERROR in rollout: {type(e).__name__}: {e}", flush=True)
                raise
            finally:
                # Cleanup environment
                print("[OpenEnvRolloutProcessor] Closing environment client...", flush=True)
                try:
                    env.close()
                    print("[OpenEnvRolloutProcessor] Environment closed successfully.", flush=True)
                except Exception as close_err:
                    print(f"[OpenEnvRolloutProcessor] Error closing environment: {close_err}", flush=True)
        
        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await process_row(r)
        
        # Create and return tasks
        print(f"[OpenEnvRolloutProcessor] Creating {len(rows)} async tasks...", flush=True)
        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        print(f"[OpenEnvRolloutProcessor] Returning {len(tasks)} tasks", flush=True)
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
                    try:
                        print(f"[OpenEnvRolloutProcessor] Using env_client_cls base_url={self._env_base_url}")
                    except Exception:
                        pass
                    return self._env_client_cls(  # type: ignore[call-arg]
                        base_url=self._env_base_url,
                        request_timeout_s=self._request_timeout_s,
                        default_headers=self._default_headers,
                    )

                # ------------------------------
                # Docker-based env: build env_vars with task rotation
                # ------------------------------
                docker_kwargs: Dict[str, Any] = {}

                # 1) Build default BrowserGym-style env vars
                env_vars_default: Dict[str, str] = {
                    "BROWSERGYM_BENCHMARK": str(self._benchmark),
                    "BROWSERGYM_HEADLESS": "true" if self._headless else "false",
                    "BROWSERGYM_VIEWPORT_WIDTH": str(self._viewport_width),
                    "BROWSERGYM_VIEWPORT_HEIGHT": str(self._viewport_height),
                    "BROWSERGYM_TIMEOUT": str(int(self._timeout_ms)),
                    # Keep obs/info flags consistent for BrowserGym
                    "BROWSERGYM_OBS_AXTREE": "1",
                    "BROWSERGYM_OBS_PRUNED_HTML": "1",
                    "BROWSERGYM_RETURN_INFO": "1",
                }

                # 2) Select task for this env instance (if provided), grouped by num_generations
                selected_task: Optional[str] = None
                if self._tasks:
                    idx = self._env_create_idx
                    self._env_create_idx = idx + 1
                    group = idx // max(1, self._num_generations)
                    selected_task = self._tasks[group % len(self._tasks)]
                    env_vars_default["BROWSERGYM_TASK_NAME"] = str(selected_task)
                    try:
                        print(
                            "[OpenEnvRolloutProcessor] Task selection:"
                            f" idx={idx}, group={group}, num_generations={self._num_generations},"
                            f" selected_task={selected_task}, tasks={self._tasks}",
                            flush=True,
                        )
                    except Exception:
                        pass

                # 3) MiniWoB URL (if provided)
                if self._miniwob_url:
                    env_vars_default["MINIWOB_URL"] = str(self._miniwob_url)

                # 4) Merge user-provided env vars (override defaults)
                if self._env_vars:
                    env_vars_default.update({k: str(v) for k, v in self._env_vars.items()})

                docker_kwargs["env_vars"] = env_vars_default

                if self._docker_port is not None:
                    docker_kwargs["port"] = int(self._docker_port)
                if self._hub_repo_id:
                    try:
                        print(f"[OpenEnvRolloutProcessor] Launching from_hub repo_id='{self._hub_repo_id}' ...")
                    except Exception:
                        pass
                    return self._env_client_cls.from_hub(  # type: ignore[attr-defined]
                        self._hub_repo_id,
                        provider=self._provider,
                        **docker_kwargs,
                    )
                else:
                    try:
                        print(f"[OpenEnvRolloutProcessor] Launching from_docker_image image='{self._docker_image}' ...")
                    except Exception:
                        pass
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
