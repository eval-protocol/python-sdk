"""
Command-line interface for Eval Protocol.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


from eval_protocol.evaluation import create_evaluation, preview_evaluation

from .cli_commands.agent_eval_cmd import agent_eval_command
from .cli_commands.common import (
    check_agent_environment,
    check_environment,
    setup_logging,
)
from .cli_commands.deploy import deploy_command
from .cli_commands.deploy_mcp import deploy_mcp_command
from .cli_commands.logs import logs_command
from .cli_commands.preview import preview_command
from .cli_commands.run_eval_cmd import hydra_cli_entry_point


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="eval-protocol: Tools for evaluation and reward modeling")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preview command
    preview_parser = subparsers.add_parser("preview", help="Preview an evaluator with sample data")
    preview_parser.add_argument(
        "--metrics-folders",
        "-m",
        nargs="+",
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'",
    )

    # Make samples optional to allow HF dataset option
    preview_parser.add_argument(
        "--samples",
        "-s",
        required=False,
        help="Path to JSONL file containing sample data",
    )
    preview_parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples to process (default: 5)",
    )

    # Add HuggingFace dataset options
    hf_group = preview_parser.add_argument_group("HuggingFace Dataset Options")
    hf_group.add_argument(
        "--huggingface-dataset",
        "--hf",
        help="HuggingFace dataset name (e.g., 'deepseek-ai/DeepSeek-ProverBench')",
    )
    hf_group.add_argument(
        "--huggingface-split",
        default="train",
        help="Dataset split to use (default: 'train')",
    )
    hf_group.add_argument(
        "--huggingface-prompt-key",
        default="prompt",
        help="Key in the dataset containing the prompt text (default: 'prompt')",
    )
    hf_group.add_argument(
        "--huggingface-response-key",
        default="response",
        help="Key in the dataset containing the response text (default: 'response')",
    )
    hf_group.add_argument(
        "--huggingface-key-map",
        help="JSON mapping of dataset keys to Eval Protocol message keys",
    )
    preview_parser.add_argument(
        "--remote-url",
        help="URL of a remote reward function endpoint to preview against. If provided, metrics-folders might be ignored.",
    )

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Create and deploy an evaluator, or register a remote one")
    deploy_parser.add_argument("--id", required=True, help="ID for the evaluator")
    deploy_parser.add_argument(
        "--metrics-folders",
        "-m",
        nargs="+",
        required=False,  # No longer strictly required if --remote-url is used
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'. Required if not using --remote-url.",
    )
    deploy_parser.add_argument(
        "--display-name",
        help="Display name for the evaluator (defaults to ID if not provided)",
    )
    deploy_parser.add_argument("--description", help="Description for the evaluator")
    deploy_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force update if evaluator already exists",
    )

    # Add HuggingFace dataset options to deploy command
    hf_deploy_group = deploy_parser.add_argument_group("HuggingFace Dataset Options")
    hf_deploy_group.add_argument(
        "--huggingface-dataset",
        "--hf",
        help="HuggingFace dataset name (e.g., 'deepseek-ai/DeepSeek-ProverBench')",
    )
    hf_deploy_group.add_argument(
        "--huggingface-split",
        default="train",
        help="Dataset split to use (default: 'train')",
    )
    hf_deploy_group.add_argument(
        "--huggingface-prompt-key",
        default="prompt",
        help="Key in the dataset containing the prompt text (default: 'prompt')",
    )
    hf_deploy_group.add_argument(
        "--huggingface-response-key",
        default="response",
        help="Key in the dataset containing the response text (default: 'response')",
    )
    hf_deploy_group.add_argument(
        "--huggingface-key-map",
        help="JSON mapping of dataset keys to Eval Protocol message keys",
    )
    deploy_parser.add_argument(
        "--remote-url",
        help="URL of a pre-deployed remote reward function. If provided, deploys by registering this URL with Fireworks AI.",
    )

    # Deployment target options
    target_group = deploy_parser.add_argument_group("Deployment Target Options")
    target_group.add_argument(
        "--target",
        choices=["fireworks", "gcp-cloud-run", "local-serve"],
        default="fireworks",
        help="Deployment target. 'fireworks' for standard Fireworks platform deployment, 'gcp-cloud-run' for Google Cloud Run, 'local-serve' for local serving with Serveo tunneling.",
    )
    target_group.add_argument(
        "--function-ref",
        help="Reference to the reward function to deploy (e.g., 'my_module.reward_func'). Required for 'gcp-cloud-run' and 'local-serve' targets.",
    )

    # Local serving options (relevant if --target is local-serve)
    local_serve_group = deploy_parser.add_argument_group("Local Serving Options (used if --target is local-serve)")
    local_serve_group.add_argument(
        "--local-port",
        type=int,
        default=8001,
        help="Port for the local reward function server to listen on (default: 8001). Used with --target local-serve.",
    )

    # GCP deployment options
    gcp_group = deploy_parser.add_argument_group(
        "GCP Cloud Run Deployment Options (used if --target is gcp-cloud-run)"
    )
    # --function-ref is now in target_group
    gcp_group.add_argument(
        "--gcp-project",
        required=False,
        help="Google Cloud Project ID. Must be provided via CLI or rewardkit.yaml.",
    )
    gcp_group.add_argument(
        "--gcp-region",
        required=False,
        help="Google Cloud Region for deployment (e.g., 'us-central1'). Must be provided via CLI or rewardkit.yaml.",
    )
    gcp_group.add_argument(
        "--gcp-ar-repo",
        required=False,
        help="Google Artifact Registry repository name. Optional, defaults to value in rewardkit.yaml or 'eval-protocol-evaluators' if not specified.",
    )
    gcp_group.add_argument(
        "--service-account",
        help="Email of the GCP service account to run the Cloud Run service. Optional.",
    )
    gcp_group.add_argument(
        "--entry-point",
        default="reward_function",
        help="The name of the entry point function within your --function-ref module (default: reward_function). Only for gcp-cloud-run.",
    )
    gcp_group.add_argument(
        "--runtime",
        default="python311",  # Or a sensible default
        help="The Cloud Functions/Run runtime (e.g., python311). Only for gcp-cloud-run.",
    )
    gcp_group.add_argument(
        "--gcp-auth-mode",
        choices=["open", "api-key"],  # Add 'iam' later
        default=None,  # Default will be resolved in deploy_command
        help="Authentication mode for the deployed GCP Cloud Run service. "
        "'open': Publicly accessible. "
        "'api-key': Service is publicly accessible but requires an API key in requests (handled by the application). "
        "If not specified, defaults to value in rewardkit.yaml or 'api-key'. Optional.",
    )

    # Deploy MCP command
    deploy_mcp_parser = subparsers.add_parser("deploy-mcp", help="Deploy an MCP server to Google Cloud Run")
    deploy_mcp_parser.add_argument("--id", required=True, help="Unique ID for the MCP server deployment")
    deploy_mcp_parser.add_argument(
        "--mcp-server-module",
        help="Python module containing the MCP server (e.g., 'examples.frozen_lake_mcp.frozen_lake_mcp_server'). Required if --dockerfile is not provided.",
    )
    deploy_mcp_parser.add_argument(
        "--dockerfile",
        help="Path to Dockerfile to use for deployment (recommended for tested local Dockerfiles). When provided, --mcp-server-module is not required.",
    )
    deploy_mcp_parser.add_argument(
        "--gcp-project",
        help="Google Cloud Project ID. Can also be set in rewardkit.yaml",
    )
    deploy_mcp_parser.add_argument(
        "--gcp-region",
        help="Google Cloud Region (e.g., 'us-central1'). Can also be set in rewardkit.yaml",
    )
    deploy_mcp_parser.add_argument(
        "--gcp-ar-repo",
        help="Google Artifact Registry repository name. Defaults to 'eval-protocol-mcp-servers'",
    )
    deploy_mcp_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the MCP server to listen on (default: 8000)",
    )
    deploy_mcp_parser.add_argument(
        "--python-version",
        default="3.11",
        help="Python version for the container (default: 3.11)",
    )
    deploy_mcp_parser.add_argument("--requirements", help="Additional pip requirements (newline separated)")
    deploy_mcp_parser.add_argument("--env-vars", nargs="*", help="Environment variables in KEY=VALUE format")

    # Agent-eval command
    agent_eval_parser = subparsers.add_parser(
        "agent-eval", help="Run agent evaluation using the ForkableResource framework."
    )
    agent_eval_parser.add_argument(
        "--task-def",
        required=True,
        help="Path to task definition file or directory containing task definitions.",
    )
    agent_eval_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Execute tasks in parallel when multiple tasks are specified.",
    )
    agent_eval_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum number of tasks to execute in parallel (default: 3).",
    )
    agent_eval_parser.add_argument(
        "--filter",
        nargs="+",
        help="Run only tasks matching the specified task IDs.",
    )
    agent_eval_parser.add_argument(
        "--output-dir",
        default="./agent_runs",
        help="Directory to store agent evaluation run results (default: ./agent_runs).",
    )
    agent_eval_parser.add_argument(
        "--model",
        help="Override MODEL_AGENT environment variable (format: provider/model_name).",
    )
    agent_eval_parser.add_argument(
        "--num-rollouts",
        type=int,
        help="Override the number of parallel rollouts to execute for each task.",
    )

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Serve logs with file watching and real-time updates")
    logs_parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")

    # Run command (for Hydra-based evaluations)
    # This subparser intentionally defines no arguments itself.
    # All arguments after 'run' will be passed to Hydra by parse_known_args.
    subparsers.add_parser(
        "run",
        help="Run an evaluation using a Hydra configuration. All arguments after 'run' are passed to Hydra.",
    )

    # Use parse_known_args to allow Hydra to handle its own arguments
    return parser.parse_known_args(args)


def main():
    """Main entry point for the CLI"""
    try:
        from dotenv import load_dotenv

        # .env.dev for development-specific overrides, .env for general
        load_dotenv(dotenv_path=Path(".") / ".env.dev", override=True)
        load_dotenv(override=True)
    except ImportError:
        pass

    # Automatic PYTHONPATH enhancement - add current directory to Python path
    # This needs to happen early, before any module loading occurs
    current_dir = os.getcwd()
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_dir not in current_pythonpath.split(os.pathsep):
        if current_pythonpath:
            os.environ["PYTHONPATH"] = f"{current_dir}{os.pathsep}{current_pythonpath}"
        else:
            os.environ["PYTHONPATH"] = current_dir
        logger.debug(f"Added current directory to PYTHONPATH: {current_dir}")

        # Also add to sys.path so it takes effect immediately for the current process
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

    # Store original sys.argv[0] because Hydra might manipulate it
    # and we need it if we're not calling a Hydra app.
    original_script_name = sys.argv[0]
    args, remaining_argv = parse_args()  # Use parse_known_args

    setup_logging(args.verbose, getattr(args, "debug", False))

    if args.command == "preview":
        return preview_command(args)
    elif args.command == "deploy":
        return deploy_command(args)
    elif args.command == "deploy-mcp":
        return deploy_mcp_command(args)
    elif args.command == "agent-eval":
        return agent_eval_command(args)
    elif args.command == "logs":
        return logs_command(args)
    elif args.command == "run":
        # For the 'run' command, Hydra takes over argument parsing.

        # Filter out the initial '--' if present in remaining_argv, which parse_known_args might add
        hydra_specific_args = [arg for arg in remaining_argv if arg != "--"]

        # Auto-detect local conf directory and add it to config path if not explicitly provided
        has_config_path = any(arg.startswith("--config-path") for arg in hydra_specific_args)
        current_dir = os.getcwd()
        local_conf_dir = os.path.join(current_dir, "conf")

        if not has_config_path and os.path.isdir(local_conf_dir):
            logger.info(f"Auto-detected local conf directory: {local_conf_dir}")
            hydra_specific_args = [
                "--config-path",
                local_conf_dir,
            ] + hydra_specific_args

        processed_hydra_args = []
        i = 0
        while i < len(hydra_specific_args):
            arg = hydra_specific_args[i]
            if arg == "--config-path":
                processed_hydra_args.append(arg)
                i += 1
                if i < len(hydra_specific_args):
                    path_val = hydra_specific_args[i]
                    abs_path = os.path.abspath(path_val)
                    logger.debug(
                        f"Converting relative --config-path '{path_val}' (space separated) to absolute '{abs_path}'"
                    )
                    processed_hydra_args.append(abs_path)
                else:
                    logger.error("--config-path specified without a value.")
                    pass
            elif arg.startswith("--config-path="):
                flag_part, path_val = arg.split("=", 1)
                processed_hydra_args.append(flag_part)
                abs_path = os.path.abspath(path_val)
                logger.debug(
                    f"Converting relative --config-path '{path_val}' (equals separated) to absolute '{abs_path}'"
                )
                processed_hydra_args.append(abs_path)
            else:
                processed_hydra_args.append(arg)
            i += 1

        sys.argv = [sys.argv[0]] + processed_hydra_args
        logger.info(f"SYSCALL_ARGV_FOR_HYDRA (after potential abspath conversion): {sys.argv}")

        try:
            hydra_cli_entry_point()
            return 0
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Evaluation failed: {e}")

            # Provide helpful suggestions for common Hydra/config errors
            if "Cannot find primary config" in error_msg:
                logger.error("HINT: Configuration file not found.")
                logger.error("SOLUTION: Ensure you have a config file in ./conf/ directory")
                logger.error("Try: eval-protocol run --config-name simple_uipath_eval")
            elif "missing from config" in error_msg or "MissingMandatoryValue" in error_msg:
                logger.error("HINT: Required configuration values are missing.")
                logger.error("SOLUTION: Check your config file for missing required fields")
            elif "Config search path" in error_msg:
                logger.error("HINT: Hydra cannot find the configuration directory.")
                logger.error("SOLUTION: Create a ./conf directory with your config files")
            elif "ValidationError" in error_msg:
                logger.error("HINT: Configuration validation failed.")
                logger.error("SOLUTION: Run 'eval-protocol validate-data --file your_data.jsonl' to check data")

            logger.error("\nQuick fix suggestions:")
            logger.error("1. Use the simplified setup: eval-protocol run --config-name simple_uipath_eval")
            logger.error("2. Validate your data first: eval-protocol validate-data --file data.jsonl --schema agent")
            logger.error("3. Ensure you have: ./conf/simple_uipath_eval.yaml and ./uipath_reward.py")
            return 1
    else:
        temp_parser = argparse.ArgumentParser(prog=os.path.basename(original_script_name))
        temp_parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
