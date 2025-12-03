import asyncio
import copy
import logging
import os
import sys
from functools import partial
from typing import Literal, Any, Optional

import chz
from datetime import datetime

# Add tinker-cookbook to path if not installed
# Assuming the directory structure:
# rft/
#   python-sdk/
#     examples/
#       tinker_math_rl/
#   tinker-cookbook/
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
tinker_path = os.path.join(repo_root, "tinker-cookbook")
if tinker_path not in sys.path:
    sys.path.append(tinker_path)

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, extract_gsm8k_final_answer
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, main

# eval_protocol imports
from eval_protocol.adapters.huggingface import create_gsm8k_adapter
from eval_protocol.integrations.tinker_cookbook import create_eval_protocol_dataset_builder, EvalProtocolEvaluator
from eval_protocol.integrations.tinker_rollout_processor import TinkerRolloutProcessor

# Import test components
from examples.tinker_math_rl.test_gsm8k_eval import test_gsm8k_tinker, get_gsm8k_input_rows

logger = logging.getLogger(__name__)


def gsm8k_row_converter(
    row: Any, group_size: int, renderer: renderers.Renderer, convo_prefix: list[renderers.Message] | None
) -> Optional[ProblemGroupBuilder]:
    """
    Convert an Eval Protocol EvaluationRow to a Tinker ProblemGroupBuilder for GSM8K.
    """
    try:
        # Extract problem and answer from EvaluationRow
        # row.messages contains the conversation. We assume the last user message is the question.
        user_msg = next((msg for msg in reversed(row.messages) if msg.role == "user"), None)
        if not user_msg:
            return None

        problem = user_msg.content
        raw_answer = row.ground_truth

        if not problem or not raw_answer:
            return None

        # Extract final answer if it looks like a GSM8K solution (contains ####)
        # Otherwise assume it is already the answer
        if "####" in raw_answer:
            answer = extract_gsm8k_final_answer(raw_answer)
        else:
            answer = raw_answer

    except Exception as e:
        logger.warning(f"Failed to parse row: {e}")
        return None

    return ProblemGroupBuilder(
        env_thunk=partial(MathEnv, problem, answer, renderer, convo_prefix=convo_prefix),
        num_envs=group_size,
    )


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training with Eval Protocol."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 512  # Increased for reasoning
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0

    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 20

    # Checkpointing
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"

    # Dataset limits
    train_limit: int = 1000
    test_limit: int = 100


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(cli_config.model_name)

    model_name_slug = cli_config.model_name.replace("/", "-")
    run_name = f"ep-gsm8k-{model_name_slug}-{cli_config.lora_rank}rank-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/ep_math_rl/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Create the builder class dynamically using the factory
    # We use create_gsm8k_adapter as the adapter factory
    # We use MathEnv.standard_fewshot_prefix as the prefix factory
    EvalProtocolDatasetBuilder = create_eval_protocol_dataset_builder(
        adapter_factory=create_gsm8k_adapter,
        row_converter=gsm8k_row_converter,
        convo_prefix_factory=MathEnv.standard_fewshot_prefix,
        train_limit=cli_config.train_limit,
        test_limit=cli_config.test_limit,
    )

    # Create the EvalProtocol Evaluator
    # Use the test_limit for the number of rows to evaluate
    eval_rows = get_gsm8k_input_rows(limit=cli_config.test_limit)

    # Need to wrap in a factory as expected by Config.evaluator_builders
    def create_eval_protocol_evaluator():
        return EvalProtocolEvaluator(
            rows=copy.deepcopy(eval_rows),
            eval_func=test_gsm8k_tinker,
            rollout_processor_cls=TinkerRolloutProcessor,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            max_tokens=cli_config.max_tokens,
            temperature=0.0,  # Greedy for eval
        )

    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=EvalProtocolDatasetBuilder(
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
        # Add our custom evaluator
        evaluator_builders=[create_eval_protocol_evaluator],
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
