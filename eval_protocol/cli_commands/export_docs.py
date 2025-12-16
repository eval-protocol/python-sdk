"""
Export CLI reference documentation as markdown files.

This module provides functionality to introspect the argparse-based CLI
and generate markdown documentation for each command.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def _get_parser_info(parser: argparse.ArgumentParser, subparser_help: str = "") -> Dict:
    """Extract information from an ArgumentParser."""
    info = {
        "prog": parser.prog,
        "description": parser.description or "",
        "help": subparser_help,  # The help text from add_parser()
        "epilog": parser.epilog or "",
        "arguments": [],
        "subparsers": {},
    }

    # Extract arguments
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            # Handle subparsers - also extract the help text for each
            for name, subparser in action.choices.items():
                # Get the help text from the subparser action's _parser_class
                subparser_help_text = ""
                if hasattr(action, "_choices_actions"):
                    for choice_action in action._choices_actions:
                        if choice_action.dest == name:
                            subparser_help_text = choice_action.help or ""
                            break
                info["subparsers"][name] = _get_parser_info(subparser, subparser_help_text)
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


def _generate_command_section(
    name: str,
    info: Dict,
    parent_command: str,
    heading_level: int = 2,
) -> List[str]:
    """Generate markdown section for a single command."""
    lines = []
    full_command = f"{parent_command} {name}".strip()
    heading = "#" * heading_level

    lines.append(f"{heading} `{full_command}`")
    lines.append("")

    # Use help text (from add_parser) or description (from ArgumentParser)
    description = info.get("help") or info.get("description") or ""
    if description and description != argparse.SUPPRESS:
        lines.append(description)
        lines.append("")

    # Arguments table
    if info["arguments"]:
        lines.append("| Option | Type | Default | Required | Description |")
        lines.append("|--------|------|---------|----------|-------------|")
        for arg in info["arguments"]:
            lines.append(_format_argument_row(arg))
        lines.append("")

    # Handle nested subparsers recursively
    if info["subparsers"]:
        for subname, subinfo in info["subparsers"].items():
            lines.extend(
                _generate_command_section(
                    subname,
                    subinfo,
                    full_command,
                    heading_level + 1,
                )
            )

    if info["epilog"]:
        lines.append(info["epilog"])
        lines.append("")

    return lines


def generate_cli_docs(parser: argparse.ArgumentParser, output_path: str) -> int:
    """
    Generate markdown documentation from an ArgumentParser to a single file.

    Args:
        parser: The root ArgumentParser instance.
        output_path: Path to write the markdown file to.

    Returns:
        0 on success, 1 on failure.
    """
    # Extract parser info
    info = _get_parser_info(parser)

    # Filter out hidden commands (like export-docs itself)
    visible_subparsers = {
        name: subinfo
        for name, subinfo in info["subparsers"].items()
        if name != "export-docs"  # Don't document the hidden command
    }

    # Generate single page
    lines = []
    lines.append("# CLI Reference")
    lines.append("")
    lines.append(f"**{info['prog']}** - {info['description']}")
    lines.append("")

    # Global options
    if info["arguments"]:
        lines.append("## Global Options")
        lines.append("")
        lines.append("| Option | Type | Default | Required | Description |")
        lines.append("|--------|------|---------|----------|-------------|")
        for arg in info["arguments"]:
            lines.append(_format_argument_row(arg))
        lines.append("")

    # Commands section
    if visible_subparsers:
        lines.append("## Commands")
        lines.append("")
        for name, subinfo in visible_subparsers.items():
            lines.extend(_generate_command_section(name, subinfo, info["prog"], heading_level=3))

    # Write single file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated: {out}")

    return 0


def export_docs_command(args: argparse.Namespace) -> int:
    """
    Export CLI documentation to a single markdown file.

    This command introspects the CLI parser and generates markdown documentation.
    """
    # Import the parser builder from cli.py to get the actual parser
    from eval_protocol.cli import build_parser

    parser = build_parser()
    return generate_cli_docs(parser, args.output)
