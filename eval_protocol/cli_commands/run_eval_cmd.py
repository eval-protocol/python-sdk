"""
CLI command for running the full evaluation pipeline (generation + evaluation).
This script is intended to be a Hydra application.
"""

import asyncio
import logging
import sys

import hydra

# Ensure hydra.core.hydra_config is available if used for output_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import (  # Ensure MISSING is imported if used in configs
    MISSING,
    DictConfig,
    OmegaConf,
)

from eval_protocol.execution.pipeline import EvaluationPipeline

logger = logging.getLogger(__name__)


def run_evaluation_command_logic(cfg: DictConfig) -> None:
    """
    Main logic for the 'run-evaluation' command.
    """
    logger.info("Starting 'run-evaluation' command with resolved Hydra config.")

    # Make Hydra's runtime output directory available to the pipeline if needed
    # This assumes 'hydra_output_dir' is a valid field in the pipeline's config if it uses it.
    # A cleaner way is for the pipeline to be Hydra-aware or for this function to pass it explicitly.
    # For now, let's add it to the cfg object that pipeline receives.
    # Ensure the config is not frozen if we add keys, then restore its original struct state.
    was_struct = OmegaConf.is_struct(cfg)
    if was_struct:
        OmegaConf.set_struct(cfg, False)
    cfg.hydra_output_dir = HydraConfig.get().runtime.output_dir
    if was_struct:
        OmegaConf.set_struct(cfg, True)

    logger.debug(f"Full configuration for pipeline:\n{OmegaConf.to_yaml(cfg)}")

    try:
        pipeline = EvaluationPipeline(pipeline_cfg=cfg)
        all_results = asyncio.run(pipeline.run())  # Store the results
        logger.info("'run-evaluation' command finished successfully.")

        # --- Add Summary Report ---
        if all_results:
            total_samples = len(all_results)
            errors = [r for r in all_results if "error" in r and r["error"]]
            # Consider a result successful for summary if it has a score and no critical error string
            successful_evals = [
                r for r in all_results if r.get("evaluation_score") is not None and not ("error" in r and r["error"])
            ]

            num_errors = len(errors)
            num_successful = len(successful_evals)

            summary_lines = [
                "\n--- Evaluation Summary ---",
                f"Total samples processed: {total_samples}",
                f"Successful evaluations: {num_successful}",
                f"Evaluation errors: {num_errors}",
            ]

            if num_successful > 0:
                scores = [
                    r["evaluation_score"]
                    for r in successful_evals
                    if isinstance(r.get("evaluation_score"), (int, float))
                ]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    min_score = min(scores)
                    max_score = max(scores)
                    summary_lines.append(f"Average score: {avg_score:.2f}")
                    summary_lines.append(f"Min score: {min_score:.2f}")
                    summary_lines.append(f"Max score: {max_score:.2f}")

                    # Score distribution (example: 5 bins)
                    bins = [
                        0.0,
                        0.2,
                        0.4,
                        0.6,
                        0.8,
                        1.01,
                    ]  # 1.01 to include 1.0 in last bin
                    score_counts = [0] * (len(bins) - 1)
                    for score in scores:
                        for i in range(len(bins) - 1):
                            if bins[i] <= score < bins[i + 1]:
                                score_counts[i] += 1
                                break
                    summary_lines.append("Score distribution:")
                    for i in range(len(bins) - 1):
                        # Ensure bin upper bound is displayed correctly as 1.0 for the last bin
                        upper_bin_display = 1.0 if bins[i + 1] > 1.0 else bins[i + 1]
                        summary_lines.append(f"  [{bins[i]:.1f} - {upper_bin_display:.1f}): {score_counts[i]}")

            if num_errors > 0:
                summary_lines.append("\nError details (first 5):")
                for i, err_item in enumerate(errors[:5]):
                    err_id = err_item.get("id", "N/A")
                    err_msg = err_item.get("error", "Unknown error")
                    # Truncate long error messages for summary
                    if len(err_msg) > 100:
                        err_msg = err_msg[:100] + "..."
                    summary_lines.append(f"  Sample ID {err_id}: {err_msg}")

            summary_lines.append("--- End of Summary ---")

            # Use logger.info for summary to respect overall logging settings
            for line in summary_lines:
                logger.info(line)
        else:
            logger.info("No results to summarize.")

    except ValueError as ve:
        error_msg = str(ve)
        logger.error(f"Configuration or Value error in pipeline: {ve}")

        # Provide helpful suggestions based on common errors
        if "final_columns" in error_msg:
            logger.error(
                "HINT: This error suggests your dataset config has 'final_columns' which conflicts with the datasets library."
            )
            logger.error("SOLUTION: Remove 'final_columns' from your dataset config or use the simplified config.")
        elif "user_query" in error_msg and "missing" in error_msg.lower():
            logger.error("HINT: Your data is missing the 'user_query' column.")
            logger.error("SOLUTION: Run 'reward-kit validate-data --file your_data.jsonl' to check data schema.")
        elif "import" in error_msg.lower() or "module" in error_msg.lower():
            logger.error("HINT: Cannot import your reward function module.")
            logger.error("SOLUTION: Ensure your reward function file is in the current directory.")
        elif "config" in error_msg.lower() and "not found" in error_msg.lower():
            logger.error("HINT: Configuration file not found.")
            logger.error("SOLUTION: Ensure your config file is in ./conf/ directory or use --config-path.")

        sys.exit(1)  # Exit with error code for critical failures
    except Exception as e:
        error_msg = str(e)
        logger.error(f"An unexpected error occurred during the evaluation pipeline: {e}")

        # Provide helpful suggestions for common issues
        if "unexpected keyword argument" in error_msg:
            logger.error("HINT: This suggests a configuration parameter is being passed incorrectly.")
            logger.error("SOLUTION: Check your dataset config for extra parameters like 'final_columns'.")
        elif "No module named" in error_msg:
            logger.error("HINT: Cannot find Python module for reward function.")
            logger.error("SOLUTION: Ensure your reward function file is in the current directory.")
        elif "not enough values to unpack" in error_msg:
            logger.error("HINT: Data format mismatch.")
            logger.error("SOLUTION: Run 'reward-kit validate-data --file your_data.jsonl' to check data format.")

        logger.error("For more help, try:")
        logger.error("1. Run 'reward-kit validate-data --file your_data.jsonl' to check data")
        logger.error("2. Use the simplified config: --config-name simple_uipath_eval")
        logger.error("3. Check that all files are in the correct locations")

        sys.exit(1)  # Exit with error code


# This is the Hydra entry point for this command.
# It needs a config_path relative to where this script is, or an absolute one.
# If reward-kit is installed, conf might not be easily found via relative paths.
# Using `pkg://` provider is more robust for installed packages.
# For now, assume a `conf` dir at project root, and this script is in `eval_protocol/cli_commands`.
import os  # Ensure os is imported for path manipulation

# So, `config_path` would be `../../conf`.
# The `config_name` will be the primary config for this `run` command.
# Let's point directly to the example's config for now to simplify debugging Hydra pathing.
# Path from eval_protocol/cli_commands/ to examples/math_example/conf/
# Construct an absolute path or a file:// URL to make it more robust.
_RUN_EVAL_CMD_DIR = os.path.dirname(os.path.abspath(__file__))
# Default config_path for @hydra.main, relative to this file.
# Points to the project's top-level 'conf' directory.
_DEFAULT_HYDRA_CONFIG_PATH = os.path.abspath(os.path.join(_RUN_EVAL_CMD_DIR, "..", "..", "conf"))


@hydra.main(config_path=_DEFAULT_HYDRA_CONFIG_PATH, config_name=None, version_base="1.3")
def hydra_cli_entry_point(cfg: DictConfig) -> None:
    # config_path and config_name from CLI will override the defaults in the decorator.
    # If --config-name is not provided via CLI, Hydra would look for a default config
    # (e.g., config.yaml) in the _DEFAULT_HYDRA_CONFIG_PATH.
    # However, our reward-kit run command will always pass --config-path and --config-name.
    # passed to `reward-kit run` (e.g., --config-path, --config-name)
    # or by Hydra's default search behavior if not provided via CLI.
    run_evaluation_command_logic(cfg)


# This allows running `python -m eval_protocol.cli_commands.run_eval_cmd` (if __main__.py in folder)
# or if this file itself is made executable.
if __name__ == "__main__":
    # This will parse sys.argv for Hydra overrides.
    # Example: python eval_protocol/cli_commands/run_eval_cmd.py dataset=gsm8k_local_prompts generation.enabled=false
    import sys  # Required for sys.exit

    hydra_cli_entry_point()
