"""
Command-line interface for Eval Protocol.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from fireworks import Fireworks

from .cli_commands.common import setup_logging
from .cli_commands.utils import add_args_from_callable_signature

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Inspect evaluation runs locally, upload evaluators, and create reinforcement fine-tuning jobs on Fireworks"
    )
    return _configure_parser(parser)


def _configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Configure all arguments and subparsers on the given parser."""
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--server",
        help="Fireworks API server hostname or URL (e.g., dev.api.fireworks.ai or https://dev.api.fireworks.ai)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

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

    # CLI workflow flags (not part of the SDK create() signature)
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
        "--yes",
        "-y",
        action="store_true",
        help="Non-interactive: upload all discovered evaluation tests",
    )
    upload_parser.add_argument(
        "--env-file",
        help="Path to .env file containing secrets to upload (default: .env in current directory)",
    )
    upload_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing evaluator with the same ID",
    )

    # Auto-generate flags from SDK Fireworks().evaluators.create() signature
    create_evaluator_fn = Fireworks().evaluators.create

    upload_skip_fields = {
        "__top_level__": {
            "account_id",  # auto-detected
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        },
        "evaluator": {
            "commit_hash",  # should be auto-detected from git
            "source",  # not relevant for CLI flow
        },
    }
    upload_aliases = {
        "evaluator_id": ["--id"],
        "evaluator.display_name": ["--name"],
    }
    upload_help_overrides = {
        "evaluator_id": "Evaluator ID to use (if multiple selections, a numeric suffix is appended)",
        "evaluator.display_name": "Display name for evaluator (defaults to ID)",
        "evaluator.description": "Description for evaluator",
        "evaluator.requirements": "Requirements for evaluator (auto-detected from requirements.txt if not provided)",
        "evaluator.entry_point": "Pytest-style entrypoint (e.g., test_file.py::test_func). Auto-detected if not provided.",
        "evaluator.default_dataset": "Default dataset to use with this evaluator",
    }

    add_args_from_callable_signature(
        upload_parser,
        create_evaluator_fn,
        skip_fields=upload_skip_fields,
        aliases=upload_aliases,
        help_overrides=upload_help_overrides,
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

    rft_parser.add_argument("--yes", "-y", action="store_true", help="Non-interactive mode")
    rft_parser.add_argument("--dry-run", action="store_true", help="Print planned SDK call without sending")
    rft_parser.add_argument("--force", action="store_true", help="Overwrite existing evaluator with the same ID")
    rft_parser.add_argument("--skip-validation", action="store_true", help="Skip local dataset/evaluator validation")
    rft_parser.add_argument(
        "--ignore-docker",
        action="store_true",
        help="Ignore Dockerfile even if present; run pytest on host during evaluator validation",
    )
    rft_parser.add_argument(
        "--docker-build-extra",
        default="",
        metavar="",
        help="Extra flags to pass to 'docker build' when validating evaluator (quoted string, e.g. \"--no-cache --pull --progress=plain\")",
    )
    rft_parser.add_argument(
        "--docker-run-extra",
        default="",
        metavar="",
        help="Extra flags to pass to 'docker run' when validating evaluator (quoted string, e.g. \"--env-file .env --memory=8g\")",
    )

    # The flags below are Eval Protocol CLI workflow controls (not part of the Fireworks SDK `create()` signature),
    # so they can’t be auto-generated via signature introspection and must be maintained here.
    rft_parser.add_argument(
        "--source-job",
        metavar="",
        help="The source reinforcement fine-tuning job to copy configuration from. If other flags are set, they will override the source job's configuration.",
    )
    rft_parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, only errors will be printed.",
    )
    skip_fields = {
        "__top_level__": {
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
            "display_name",
            "account_id",
        },
        "training_config": {"region", "jinja_template"},
        "wandb_config": {"run_id"},
    }
    aliases = {
        "wandb_config.api_key": ["--wandb-api-key"],
        "wandb_config.project": ["--wandb-project"],
        "wandb_config.entity": ["--wandb-entity"],
        "wandb_config.enabled": ["--wandb"],
        "reinforcement_fine_tuning_job_id": ["--job-id"],
        "loss_config.kl_beta": ["--rl-kl-beta"],
        "loss_config.method": ["--rl-loss-method"],
        "node_count": ["--nodes"],
    }
    help_overrides = {
        "training_config.gradient_accumulation_steps": "The number of batches to accumulate gradients before updating the model parameters. The effective batch size will be batch-size multiplied by this value.",
        "training_config.learning_rate_warmup_steps": "The number of learning rate warmup steps for the reinforcement fine-tuning job.",
        "mcp_server": "The MCP server resource name to use for the reinforcement fine-tuning job. (Optional)",
        "loss_config.method": "RL loss method for underlying trainers. One of {grpo,dapo}.",
    }

    create_rft_job_fn = Fireworks().reinforcement_fine_tuning_jobs.create

    add_args_from_callable_signature(
        rft_parser,
        create_rft_job_fn,
        skip_fields=skip_fields,
        aliases=aliases,
        help_overrides=help_overrides,
    )

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

    # Hidden command: export-docs (for generating CLI reference documentation)
    export_docs_parser = subparsers.add_parser("export-docs", help=argparse.SUPPRESS)
    export_docs_parser.add_argument(
        "--output",
        "-o",
        default="./docs/cli-reference.md",
        help="Output markdown file path (default: ./docs/cli-reference.md)",
    )

    # Update metavar to only show visible commands (exclude those with SUPPRESS)
    _hide_suppressed_subparsers(parser)

    return parser


def _hide_suppressed_subparsers(parser: argparse.ArgumentParser) -> None:
    """Update subparsers to exclude commands with help=SUPPRESS from help output."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            # Filter _choices_actions to only visible commands
            choices_actions = getattr(action, "_choices_actions", [])
            visible_actions = [a for a in choices_actions if a.help != argparse.SUPPRESS]
            action._choices_actions = visible_actions
            # Update metavar to match
            visible_names = [a.dest for a in visible_actions]
            if visible_names:
                action.metavar = "{" + ",".join(visible_names) + "}"


def parse_args(args=None):
    """Parse command line arguments."""
    parser = build_parser()
    # Fail fast on unknown flags so typos don't silently get ignored.
    parsed, remaining = parser.parse_known_args(args)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    return parsed, remaining


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

    pre_server = _extract_flag_value(raw_argv, "--server")

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
    args, _ = parse_args()

    setup_logging(args.verbose, getattr(args, "debug", False))

    if args.command == "logs":
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
    elif args.command == "export-docs":
        from .cli_commands.export_docs import export_docs_command

        return export_docs_command(args)
    else:
        parser = build_parser()
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
