"""
Export CLI reference documentation as markdown files.

This module provides functionality to introspect the argparse-based CLI
and generate markdown documentation for each command.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_parser_info(parser: argparse.ArgumentParser) -> Dict:
    """Extract information from an ArgumentParser."""
    info = {
        "prog": parser.prog,
        "description": parser.description or "",
        "epilog": parser.epilog or "",
        "arguments": [],
        "subparsers": {},
    }

    # Extract arguments
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            # Handle subparsers
            for name, subparser in action.choices.items():
                info["subparsers"][name] = _get_parser_info(subparser)
        elif isinstance(action, argparse._HelpAction):
            # Skip help action, it's always present
            continue
        else:
            arg_info = {
                "option_strings": action.option_strings,
                "dest": action.dest,
                "help": action.help or "",
                "default": action.default,
                "required": getattr(action, "required", False),
                "type": getattr(action, "type", None),
                "choices": getattr(action, "choices", None),
                "nargs": getattr(action, "nargs", None),
                "metavar": getattr(action, "metavar", None),
            }
            # Check if help is suppressed
            if action.help != argparse.SUPPRESS:
                info["arguments"].append(arg_info)

    return info


def _format_argument_row(arg: Dict) -> str:
    """Format a single argument as a markdown table row."""
    # Build the flag/argument name
    if arg["option_strings"]:
        name = ", ".join(f"`{opt}`" for opt in arg["option_strings"])
    else:
        name = f"`{arg['dest']}`"

    # Build type info
    type_str = ""
    if arg["type"]:
        type_str = getattr(arg["type"], "__name__", str(arg["type"]))
    if arg["choices"]:
        type_str = f"choices: {arg['choices']}"

    # Format default value
    default = arg["default"]
    if default is None:
        default_str = "-"
    elif default == argparse.SUPPRESS:
        default_str = "-"
    elif isinstance(default, bool):
        default_str = str(default).lower()
    else:
        default_str = f"`{default}`"

    # Help text (escape pipe characters for markdown tables)
    help_text = (arg["help"] or "-").replace("|", "\\|")

    # Required indicator
    required = "Yes" if arg["required"] else "No"

    return f"| {name} | {type_str} | {default_str} | {required} | {help_text} |"


def _generate_command_markdown(
    name: str,
    info: Dict,
    parent_command: str = "",
    level: int = 1,
) -> str:
    """Generate markdown documentation for a single command."""
    lines = []

    # Command title
    full_command = f"{parent_command} {name}".strip() if parent_command else name
    heading = "#" * min(level, 4)
    lines.append(f"{heading} `{full_command}`")
    lines.append("")

    # Description
    if info["description"]:
        lines.append(info["description"])
        lines.append("")

    # Arguments table
    if info["arguments"]:
        lines.append("**Options:**")
        lines.append("")
        lines.append("| Option | Type | Default | Required | Description |")
        lines.append("|--------|------|---------|----------|-------------|")
        for arg in info["arguments"]:
            lines.append(_format_argument_row(arg))
        lines.append("")

    # Epilog
    if info["epilog"]:
        lines.append(info["epilog"])
        lines.append("")

    return "\n".join(lines)


def _generate_subcommand_docs(
    subparsers: Dict,
    parent_command: str,
    level: int,
) -> List[Tuple[str, str]]:
    """Generate markdown docs for all subcommands, returns list of (filename, content)."""
    docs = []

    for name, info in subparsers.items():
        full_command = f"{parent_command} {name}".strip()

        # Generate this command's doc
        content_lines = []
        content_lines.append(f"# `{full_command}`")
        content_lines.append("")

        if info["description"]:
            content_lines.append(info["description"])
            content_lines.append("")

        # Arguments table
        if info["arguments"]:
            content_lines.append("## Options")
            content_lines.append("")
            content_lines.append("| Option | Type | Default | Required | Description |")
            content_lines.append("|--------|------|---------|----------|-------------|")
            for arg in info["arguments"]:
                content_lines.append(_format_argument_row(arg))
            content_lines.append("")

        # Handle nested subparsers
        if info["subparsers"]:
            content_lines.append("## Subcommands")
            content_lines.append("")
            for subname in info["subparsers"].keys():
                sub_full = f"{full_command} {subname}"
                content_lines.append(f"- [`{sub_full}`]({name}-{subname}.md)")
            content_lines.append("")

            # Recursively generate docs for nested subcommands
            nested_docs = _generate_subcommand_docs(
                info["subparsers"],
                full_command,
                level + 1,
            )
            for nested_filename, nested_content in nested_docs:
                docs.append((f"{name}-{nested_filename}", nested_content))

        if info["epilog"]:
            content_lines.append(info["epilog"])
            content_lines.append("")

        filename = name.replace(" ", "-") + ".md"
        docs.append((filename, "\n".join(content_lines)))

    return docs


def generate_cli_docs(parser: argparse.ArgumentParser, output_dir: str) -> int:
    """
    Generate markdown documentation from an ArgumentParser.

    Args:
            parser: The root ArgumentParser instance.
            output_dir: Directory to write markdown files to.

    Returns:
            0 on success, 1 on failure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract parser info
    info = _get_parser_info(parser)

    # Generate index/overview page
    index_lines = []
    index_lines.append("# CLI Reference")
    index_lines.append("")
    index_lines.append(f"**{info['prog']}** - {info['description']}")
    index_lines.append("")

    # Global options
    if info["arguments"]:
        index_lines.append("## Global Options")
        index_lines.append("")
        index_lines.append("| Option | Type | Default | Required | Description |")
        index_lines.append("|--------|------|---------|----------|-------------|")
        for arg in info["arguments"]:
            index_lines.append(_format_argument_row(arg))
        index_lines.append("")

    # Commands section
    if info["subparsers"]:
        index_lines.append("## Commands")
        index_lines.append("")
        for name, subinfo in info["subparsers"].items():
            description = subinfo["description"] or ""
            # Truncate long descriptions for the index
            if len(description) > 100:
                description = description[:97] + "..."
            index_lines.append(f"- [`{name}`]({name}.md) - {description}")
        index_lines.append("")

    # Write index file
    index_path = output_path / "index.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    logger.info(f"Generated: {index_path}")

    # Generate individual command docs
    if info["subparsers"]:
        docs = _generate_subcommand_docs(info["subparsers"], info["prog"], 1)
        for filename, content in docs:
            file_path = output_path / filename
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"Generated: {file_path}")

    logger.info(f"CLI documentation exported to: {output_path}")
    return 0


def export_docs_command(args: argparse.Namespace) -> int:
    """
    Export CLI documentation to markdown files.

    This command introspects the CLI parser and generates markdown documentation.
    """
    # Import here to avoid circular imports
    from eval_protocol.cli import parse_args

    # Create a fresh parser by calling parse_args with empty args
    # We need to access the parser directly
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
    rft_parser.add_argument("--base-model", help="Base model resource id")
    rft_parser.add_argument("--warm-start-from", help="Addon model to warm start from")
    rft_parser.add_argument("--output-model", help="Output model id (defaults from evaluator)")
    rft_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    rft_parser.add_argument("--batch-size", type=int, default=128000, help="Training batch size")
    rft_parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    rft_parser.add_argument("--max-context-length", type=int, default=65536, help="Maximum context length")
    rft_parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    rft_parser.add_argument("--gradient-accumulation-steps", type=int, help="Number of gradient accumulation steps")
    rft_parser.add_argument("--learning-rate-warmup-steps", type=int, help="Number of LR warmup steps")
    rft_parser.add_argument("--accelerator-count", type=int, help="Number of accelerators")
    rft_parser.add_argument("--region", help="Fireworks region enum value")
    rft_parser.add_argument("--display-name", help="RFT job display name")
    rft_parser.add_argument("--evaluation-dataset", help="Optional separate eval dataset id")
    rft_parser.add_argument(
        "--eval-auto-carveout",
        dest="eval_auto_carveout",
        action="store_true",
        default=True,
        help="Enable auto carveout for evaluation (default: true)",
    )
    rft_parser.add_argument(
        "--no-eval-auto-carveout",
        dest="eval_auto_carveout",
        action="store_false",
        help="Disable auto carveout for evaluation",
    )
    rft_parser.add_argument("--chunk-size", type=int, default=100, help="Data chunk size for rollout batching")
    rft_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    rft_parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    rft_parser.add_argument("--top-k", type=int, help="Top-k sampling parameter")
    rft_parser.add_argument("--max-output-tokens", type=int, default=32768, help="Maximum output tokens")
    rft_parser.add_argument("--response-candidates-count", type=int, default=8, help="Number of response candidates")
    rft_parser.add_argument("--extra-body", help="JSON string for extra inference params")
    rft_parser.add_argument(
        "--mcp-server",
        help="The MCP server resource name to use for the reinforcement fine-tuning job.",
    )
    rft_parser.add_argument("--wandb-enabled", action="store_true", help="Enable Weights & Biases logging")
    rft_parser.add_argument("--wandb-project", help="Weights & Biases project name")
    rft_parser.add_argument("--wandb-entity", help="Weights & Biases entity")
    rft_parser.add_argument("--wandb-run-id", help="Weights & Biases run ID")
    rft_parser.add_argument("--wandb-api-key", help="Weights & Biases API key")
    rft_parser.add_argument("--job-id", help="Specify an explicit RFT job id")
    rft_parser.add_argument("--yes", "-y", action="store_true", help="Non-interactive mode")
    rft_parser.add_argument("--dry-run", action="store_true", help="Print planned REST calls without sending")
    rft_parser.add_argument("--force", action="store_true", help="Overwrite existing evaluator with the same ID")
    rft_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip local dataset and evaluator validation before creating the RFT job",
    )
    rft_parser.add_argument(
        "--ignore-docker",
        action="store_true",
        help="Ignore Dockerfile even if present; run pytest on host during evaluator validation",
    )
    rft_parser.add_argument(
        "--docker-build-extra",
        default="",
        help="Extra flags to pass to 'docker build' when validating evaluator (quoted string)",
    )
    rft_parser.add_argument(
        "--docker-run-extra",
        default="",
        help="Extra flags to pass to 'docker run' when validating evaluator (quoted string)",
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
        help="Extra flags to pass to 'docker build' (quoted string)",
    )
    local_test_parser.add_argument(
        "--docker-run-extra",
        default="",
        help="Extra flags to pass to 'docker run' (quoted string)",
    )

    output_dir = args.output_dir
    return generate_cli_docs(parser, output_dir)
