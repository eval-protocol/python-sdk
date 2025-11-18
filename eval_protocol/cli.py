"""
Command-line interface for Eval Protocol.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


from .cli_commands.common import setup_logging

# Re-export deploy_command for backward compatibility with tests importing from eval_protocol.cli
try:  # pragma: no cover - import-time alias for tests
    from .cli_commands import deploy as _deploy_mod

    deploy_command = _deploy_mod.deploy_command  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    # If import fails in constrained environments, tests that import it will surface the issue
    deploy_command = None  # type: ignore[assignment]

# Re-export preview_command for backward compatibility with tests importing from eval_protocol.cli
try:  # pragma: no cover - import-time alias for tests
    from .cli_commands import preview as _preview_mod

    preview_command = _preview_mod.preview_command  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    preview_command = None  # type: ignore[assignment]


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="eval-protocol: Tools for evaluation and reward modeling")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--profile",
        help="Fireworks profile to use (reads ~/.fireworks/profiles/<name>/auth.ini and settings.ini)",
    )
    parser.add_argument(
        "--server",
        help="Fireworks API server hostname or URL (e.g., dev.api.fireworks.ai or https://dev.api.fireworks.ai)",
    )

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
    logs_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    logs_parser.add_argument("--disable-elasticsearch-setup", action="store_true", help="Disable Elasticsearch setup")
    logs_parser.add_argument(
        "--use-env-elasticsearch-config",
        action="store_true",
        help="Use env vars for Elasticsearch config (requires ELASTICSEARCH_URL, ELASTICSEARCH_API_KEY, ELASTICSEARCH_INDEX_NAME)",
    )
    logs_parser.add_argument(
        "--use-fireworks",
        action="store_true",
        help="Force Fireworks tracing backend for logs UI (overrides env auto-detection)",
    )
    logs_parser.add_argument(
        "--use-elasticsearch",
        action="store_true",
        help="Force Elasticsearch backend for logs UI (overrides env auto-detection)",
    )

    # Upload command
    upload_parser = subparsers.add_parser(
        "upload",
        help="Scan for evaluation tests, select, and upload as Fireworks evaluators",
    )
    upload_parser.add_argument(
        "--path",
        default=".",
        help="Path to search for evaluation tests (default: current directory)",
    )
    upload_parser.add_argument(
        "--entry",
        help="Entrypoint of evaluation test to upload (module:function or path::function). For multiple, separate by commas.",
    )
    upload_parser.add_argument(
        "--id",
        help="Evaluator ID to use (if multiple selections, a numeric suffix is appended)",
    )
    upload_parser.add_argument(
        "--display-name",
        help="Display name for evaluator (defaults to ID)",
    )
    upload_parser.add_argument(
        "--description",
        help="Description for evaluator",
    )
    upload_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing evaluator with the same ID",
    )
    upload_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Non-interactive: upload all discovered evaluation tests",
    )
    upload_parser.add_argument(
        "--env-file",
        help="Path to .env file containing secrets to upload (default: .env in current directory)",
    )

    # Create command group
    create_parser = subparsers.add_parser(
        "create",
        help="Resource creation commands",
    )
    create_subparsers = create_parser.add_subparsers(dest="create_command")
    rft_parser = create_subparsers.add_parser(
        "rft",
        help="Create a Reinforcement Fine-tuning Job on Fireworks",
    )
    rft_parser.add_argument(
        "--evaluator",
        help="Evaluator ID or fully-qualified resource (accounts/{acct}/evaluators/{id}); if omitted, derive from local tests",
    )
    # Dataset options
    rft_parser.add_argument(
        "--dataset",
        help="Use existing dataset (ID or resource 'accounts/{acct}/datasets/{id}') to skip local materialization",
    )
    rft_parser.add_argument(
        "--dataset-jsonl",
        help="Path to JSONL to upload as a new Fireworks dataset",
    )
    rft_parser.add_argument(
        "--dataset-builder",
        help="Explicit dataset builder spec (module::function or path::function)",
    )
    rft_parser.add_argument(
        "--dataset-display-name",
        help="Display name for dataset on Fireworks (defaults to dataset id)",
    )
    # Training config and evaluator/job settings
    rft_parser.add_argument("--base-model", help="Base model resource id")
    rft_parser.add_argument("--warm-start-from", help="Addon model to warm start from")
    rft_parser.add_argument("--output-model", help="Output model id (defaults from evaluator)")
    rft_parser.add_argument("--epochs", type=int, default=1)
    rft_parser.add_argument("--batch-size", type=int, default=128000)
    rft_parser.add_argument("--learning-rate", type=float, default=3e-5)
    rft_parser.add_argument("--max-context-length", type=int, default=65536)
    rft_parser.add_argument("--lora-rank", type=int, default=16)
    rft_parser.add_argument("--gradient-accumulation-steps", type=int, help="Number of gradient accumulation steps")
    rft_parser.add_argument("--learning-rate-warmup-steps", type=int, help="Number of LR warmup steps")
    rft_parser.add_argument("--accelerator-count", type=int)
    rft_parser.add_argument("--region", help="Fireworks region enum value")
    rft_parser.add_argument("--display-name", help="RFT job display name")
    rft_parser.add_argument("--evaluation-dataset", help="Optional separate eval dataset id")
    rft_parser.add_argument("--eval-auto-carveout", dest="eval_auto_carveout", action="store_true", default=True)
    rft_parser.add_argument("--no-eval-auto-carveout", dest="eval_auto_carveout", action="store_false")
    # Rollout chunking
    rft_parser.add_argument("--chunk-size", type=int, default=100, help="Data chunk size for rollout batching")
    # Inference params
    rft_parser.add_argument("--temperature", type=float)
    rft_parser.add_argument("--top-p", type=float)
    rft_parser.add_argument("--top-k", type=int)
    rft_parser.add_argument("--max-output-tokens", type=int, default=32768)
    rft_parser.add_argument("--response-candidates-count", type=int, default=8)
    rft_parser.add_argument("--extra-body", help="JSON string for extra inference params")
    # MCP server (optional)
    rft_parser.add_argument(
        "--mcp-server",
        help="The MCP server resource name to use for the reinforcement fine-tuning job.",
    )
    # Wandb
    rft_parser.add_argument("--wandb-enabled", action="store_true")
    rft_parser.add_argument("--wandb-project")
    rft_parser.add_argument("--wandb-entity")
    rft_parser.add_argument("--wandb-run-id")
    rft_parser.add_argument("--wandb-api-key")
    # Misc
    rft_parser.add_argument("--job-id", help="Specify an explicit RFT job id")
    rft_parser.add_argument("--yes", "-y", action="store_true", help="Non-interactive mode")
    rft_parser.add_argument("--dry-run", action="store_true", help="Print planned REST calls without sending")
    rft_parser.add_argument("--force", action="store_true", help="Overwrite existing evaluator with the same ID")

    # Local test command
    local_test_parser = subparsers.add_parser(
        "local-test",
        help="Select an evaluation test and run it locally. If a Dockerfile exists, build and run via Docker; otherwise run on host.",
    )
    local_test_parser.add_argument(
        "--entry",
        help="Entrypoint to run (path::function or path). If not provided, a selector will be shown (unless --yes).",
    )
    local_test_parser.add_argument(
        "--ignore-docker",
        action="store_true",
        help="Ignore Dockerfile even if present; run pytest on host",
    )
    local_test_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Non-interactive: if multiple tests exist and no --entry, fails with guidance",
    )
    local_test_parser.add_argument(
        "--docker-build-extra",
        default="",
        help="Extra flags to pass to 'docker build' (quoted string, e.g. \"--no-cache --pull --progress=plain\")",
    )
    local_test_parser.add_argument(
        "--docker-run-extra",
        default="",
        help="Extra flags to pass to 'docker run' (quoted string, e.g. \"--env-file .env --memory=8g\")",
    )

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
        logger.debug("Added current directory to PYTHONPATH: %s", current_dir)

        # Also add to sys.path so it takes effect immediately for the current process
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

    # Pre-scan raw argv for global flags anywhere (before parsing or imports)
    raw_argv = sys.argv[1:]

    def _extract_flag_value(argv_list, flag_name):
        # Supports --flag value and --flag=value
        for i, tok in enumerate(argv_list):
            if tok == flag_name:
                if i + 1 < len(argv_list):
                    return argv_list[i + 1]
            elif tok.startswith(flag_name + "="):
                return tok.split("=", 1)[1]
        return None

    pre_profile = _extract_flag_value(raw_argv, "--profile")
    pre_server = _extract_flag_value(raw_argv, "--server")

    # Handle Fireworks profile selection early so downstream modules see the env
    profile = pre_profile
    if profile:
        try:
            os.environ["FIREWORKS_PROFILE"] = profile
            # Mirror firectl behavior: ~/.fireworks[/profiles/<profile>]
            base_dir = Path.home() / ".fireworks"
            if profile:
                base_dir = base_dir / "profiles" / profile
            os.makedirs(str(base_dir), mode=0o700, exist_ok=True)

            # Provide helpful env hints for consumers (optional)
            os.environ["FIREWORKS_AUTH_FILE"] = str(base_dir / "auth.ini")
            os.environ["FIREWORKS_SETTINGS_FILE"] = str(base_dir / "settings.ini")
            logger.debug("Using Fireworks profile '%s' at %s", profile, base_dir)
        except OSError as e:
            logger.warning("Failed to initialize Fireworks profile '%s': %s", profile, e)

        # Proactively resolve and export account_id from the active profile to avoid stale .env overrides
        try:
            from eval_protocol.auth import get_fireworks_account_id as _resolve_account_id

            resolved_account = _resolve_account_id()
            if resolved_account:
                os.environ["FIREWORKS_ACCOUNT_ID"] = resolved_account
                logger.debug("Resolved account_id from profile '%s': %s", profile, resolved_account)
        except Exception as e:  # noqa: B902
            logger.debug("Unable to resolve account_id from profile '%s': %s", profile, e)

    # Handle Fireworks server selection early
    server = pre_server
    if server:
        # Normalize to full URL if just a hostname is supplied
        normalized = server.strip()
        if not normalized.startswith("http://") and not normalized.startswith("https://"):
            normalized = f"https://{normalized}"
        os.environ["FIREWORKS_API_BASE"] = normalized
        logger.debug("Using Fireworks API base: %s", normalized)

    # Now parse args normally (so help/commands work), after globals applied
    # Store original sys.argv[0] because Hydra might manipulate it
    # and we need it if we're not calling a Hydra app.
    original_script_name = sys.argv[0]
    args, remaining_argv = parse_args()  # Use parse_known_args

    setup_logging(args.verbose, getattr(args, "debug", False))

    if args.command == "preview":
        if preview_command is None:
            raise ImportError("preview_command is unavailable")
        return preview_command(args)
    elif args.command == "deploy":
        if deploy_command is None:
            raise ImportError("deploy_command is unavailable")
        return deploy_command(args)
    elif args.command == "deploy-mcp":
        from .cli_commands.deploy_mcp import deploy_mcp_command

        return deploy_mcp_command(args)
    elif args.command == "agent-eval":
        from .cli_commands.agent_eval_cmd import agent_eval_command

        return agent_eval_command(args)
    elif args.command == "logs":
        from .cli_commands.logs import logs_command

        return logs_command(args)
    elif args.command == "upload":
        from .cli_commands.upload import upload_command

        return upload_command(args)
    elif args.command == "create":
        if args.create_command == "rft":
            from .cli_commands.create_rft import create_rft_command

            return create_rft_command(args)
        print("Error: missing subcommand for 'create'. Try: eval-protocol create rft")
        return 1
    elif args.command == "local-test":
        from .cli_commands.local_test import local_test_command

        return local_test_command(args)
    elif args.command == "run":
        # For the 'run' command, Hydra takes over argument parsing.

        # Filter out the initial '--' if present in remaining_argv, which parse_known_args might add
        hydra_specific_args = [arg for arg in remaining_argv if arg != "--"]

        # Auto-detect local conf directory and add it to config path if not explicitly provided
        has_config_path = any(arg.startswith("--config-path") for arg in hydra_specific_args)
        current_dir = os.getcwd()
        local_conf_dir = os.path.join(current_dir, "conf")

        if not has_config_path and os.path.isdir(local_conf_dir):
            logger.info("Auto-detected local conf directory: %s", local_conf_dir)
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
                        "Converting relative --config-path '%s' (space separated) to absolute '%s'",
                        path_val,
                        abs_path,
                    )
                    processed_hydra_args.append(abs_path)
                else:
                    logger.error("--config-path specified without a value.")
            elif arg.startswith("--config-path="):
                flag_part, path_val = arg.split("=", 1)
                processed_hydra_args.append(flag_part)
                abs_path = os.path.abspath(path_val)
                logger.debug(
                    "Converting relative --config-path '%s' (equals separated) to absolute '%s'",
                    path_val,
                    abs_path,
                )
                processed_hydra_args.append(abs_path)
            else:
                processed_hydra_args.append(arg)
            i += 1

        sys.argv = [sys.argv[0]] + processed_hydra_args
        logger.info("SYSCALL_ARGV_FOR_HYDRA (after potential abspath conversion): %s", sys.argv)

        try:
            from .cli_commands.run_eval_cmd import hydra_cli_entry_point

            hydra_entry = cast(Any, hydra_cli_entry_point)
            hydra_entry()  # type: ignore  # pylint: disable=no-value-for-parameter
            return 0
        except Exception as e:  # pylint: disable=broad-except
            error_msg = str(e)
            logger.error("Evaluation failed: %s", e)

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
