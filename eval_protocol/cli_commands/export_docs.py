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


def _escape_mdx_text(text: str) -> str:
    """
    Escape text that will be emitted as the *children* of an MDX/JSX component.

    In MDX, `{` and `}` can start JS expressions even in otherwise plain text,
    which can break parsing when help strings include JSON examples.
    """
    if not text:
        return ""
    # IMPORTANT: escape '&' first to avoid double-escaping.
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
    )


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


def _format_argument_item(arg: Dict) -> List[str]:
    """Format a single argument as a Mintlify ParamField component."""
    lines = []

    # Build the flag name
    if arg["option_strings"]:
        long_opts = [o for o in arg["option_strings"] if o.startswith("--")]
        short_opts = [o for o in arg["option_strings"] if not o.startswith("--")]
        primary = long_opts[0] if long_opts else arg["option_strings"][0]
    else:
        primary = arg["dest"]
        short_opts = []

    # Map Python types to ParamField types
    type_str = ""
    if arg["type"]:
        python_type = getattr(arg["type"], "__name__", str(arg["type"]))
        type_map = {"int": "number", "float": "number", "str": "string", "bool": "boolean"}
        type_str = type_map.get(python_type, python_type)
    elif arg["default"] is not None:
        # Infer type from default
        if isinstance(arg["default"], bool):
            type_str = "boolean"
        elif isinstance(arg["default"], int):
            type_str = "number"
        elif isinstance(arg["default"], float):
            type_str = "number"
        elif isinstance(arg["default"], str):
            type_str = "string"

    # Build ParamField attributes
    attrs = [f'path="{primary}"']

    if type_str:
        attrs.append(f'type="{type_str}"')

    # Default value
    default = arg["default"]
    if default is not None and default != argparse.SUPPRESS:
        if isinstance(default, bool):
            default_str = str(default).lower()
        elif isinstance(default, str):
            # Escape quotes in string defaults
            default_str = default.replace('"', '\\"')
        else:
            default_str = str(default)
        attrs.append(f'default="{default_str}"')

    if arg["required"]:
        attrs.append("required")

    # Build description with alias mention (short + additional long aliases)
    help_text = _escape_mdx_text(arg["help"] or "")

    aliases: List[str] = []
    if arg["option_strings"]:
        aliases = [o for o in arg["option_strings"] if o != primary]

    if aliases:
        # Put long aliases first, then short ones for readability.
        long_aliases = [a for a in aliases if a.startswith("--")]
        short_aliases = [a for a in aliases if not a.startswith("--")]
        aliases_fmt = ", ".join([f"`{a}`" for a in (long_aliases + short_aliases)])
        alias_note = f"Aliases: {aliases_fmt}"
        if help_text:
            help_text = f"{help_text} ({alias_note})"
        else:
            help_text = alias_note

    # Add choices info to description
    if arg["choices"]:
        choices_str = ", ".join(f"`{c}`" for c in arg["choices"])
        choices_note = f"Choices: {choices_str}"
        if help_text:
            help_text = f"{help_text}. {choices_note}"
        else:
            help_text = choices_note

    # Generate ParamField
    lines.append(f"<ParamField {' '.join(attrs)}>")
    if help_text:
        lines.append(f"  {help_text}")
    lines.append("</ParamField>")
    lines.append("")

    return lines


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

    # Skip commands that have no arguments and only subparsers (like "ep create")
    # Instead, just render the subcommands directly at the same level
    if not info["arguments"] and info["subparsers"]:
        # Skip this level, render subcommands directly
        for subname, subinfo in info["subparsers"].items():
            lines.extend(
                _generate_command_section(
                    subname,
                    subinfo,
                    full_command,
                    heading_level,  # Keep same heading level
                )
            )
        return lines

    lines.append(f"{heading} `{full_command}`")
    lines.append("")

    # Use help text (from add_parser) or description (from ArgumentParser)
    description = info.get("help") or info.get("description") or ""
    if description and description != argparse.SUPPRESS:
        lines.append(description)
        lines.append("")

    # Arguments (no extra heading to keep TOC clean)
    if info["arguments"]:
        for arg in info["arguments"]:
            lines.extend(_format_argument_item(arg))

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

    # Generate single page with Mintlify frontmatter
    lines = []
    lines.append("---")
    lines.append("title: CLI")
    lines.append("icon: terminal")
    lines.append("---")
    lines.append("")
    lines.append(
        f"The `{info['prog']}` command-line interface can {info['description'][0].lower()}{info['description'][1:]}."
    )
    lines.append("")
    lines.append("```bash")
    lines.append(f"{info['prog']} [global options] <command> [command options]")
    lines.append("```")
    lines.append("")

    # Global options
    if info["arguments"]:
        lines.append("## Global Options")
        lines.append("")
        lines.append("These options can be used with any command:")
        lines.append("")
        for arg in info["arguments"]:
            lines.extend(_format_argument_item(arg))

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
