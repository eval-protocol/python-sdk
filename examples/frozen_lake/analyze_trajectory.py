#!/usr/bin/env python3
"""
Agent Trajectory Analyzer for Frozen Lake HTTP Rollout Evaluation

This script parses the evaluation logs and creates a human-readable
trajectory showing the agent's decision making process.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_tool_calls_from_log(log_content: str) -> List[Dict[str, Any]]:
    """Extract tool calls and their results from the log."""
    tool_calls = []

    # Find all tool call patterns
    tool_call_pattern = r"Attempting tool call: (\w+)\((.*?)\)"
    tool_result_pattern = r"Tool '(\w+)' result: (.*?)(?=INFO:|DEBUG:|ERROR:|$)"

    tool_calls_matches = re.finditer(tool_call_pattern, log_content, re.DOTALL)

    for match in tool_calls_matches:
        tool_name = match.group(1)
        tool_args = match.group(2)

        # Try to parse the arguments as JSON
        try:
            args_dict = json.loads(tool_args)
        except (json.JSONDecodeError, TypeError, ValueError):
            args_dict = {"raw": tool_args}

        tool_call = {"tool_name": tool_name, "arguments": args_dict, "result": None}

        # Find the corresponding result
        result_pattern = rf"Tool '{tool_name}' result: (.*?)(?=INFO:|DEBUG:|ERROR:|$)"
        result_match = re.search(result_pattern, log_content[match.end() :], re.DOTALL)

        if result_match:
            result_text = result_match.group(1).strip()
            # Try to parse as JSON
            try:
                tool_call["result"] = json.loads(result_text)
            except (json.JSONDecodeError, TypeError, ValueError):
                tool_call["result"] = {"raw": result_text}

        tool_calls.append(tool_call)

    return tool_calls


def extract_agent_messages(log_content: str) -> List[Dict[str, Any]]:
    """Extract the agent's reasoning and responses."""
    messages = []

    # Find OpenAI response messages
    response_pattern = r"OpenAI response message: ChatCompletionMessage\((.*?)\)"

    for match in re.finditer(response_pattern, log_content, re.DOTALL):
        message_str = match.group(1)

        # Extract thinking content
        think_match = re.search(r"content='(.*?)', refusal=", message_str, re.DOTALL)
        if think_match:
            thinking = think_match.group(1)

            # Extract <think> tags
            think_content_match = re.search(r"<think>(.*?)</think>", thinking, re.DOTALL)
            if think_content_match:
                thinking_content = think_content_match.group(1).strip()
            else:
                thinking_content = thinking

            messages.append({"type": "thinking", "content": thinking_content})

        # Extract tool calls from the message
        tool_calls_match = re.search(r"tool_calls=\[(.*?)\]", message_str, re.DOTALL)
        if tool_calls_match:
            messages.append({"type": "tool_calls", "content": tool_calls_match.group(1)})

    return messages


def extract_game_state_changes(log_content: str) -> List[Dict[str, Any]]:
    """Extract game state changes from the environment responses."""
    states = []

    # Find environment responses
    env_pattern = r"Environment: (.*?)(?=\\n|Position:|Done:)"
    visual_pattern = r"Visual State:\\n(.*?)(?=\\nPosition:|\\nDone:)"
    position_pattern = r"Position: (\[.*?\])"
    done_pattern = r"Done: (True|False)"

    # Find all environment messages
    env_matches = re.finditer(
        r"Tool 'step' result:.*?Environment: (.*?)\\nVisual State:\\n(.*?)\\nPosition: (\[.*?\])\\nDone: (True|False)",
        log_content,
        re.DOTALL,
    )

    for i, match in enumerate(env_matches):
        env_message = match.group(1)
        visual_state = match.group(2)
        position = match.group(3)
        done = match.group(4) == "True"

        states.append(
            {
                "step": i + 1,
                "message": env_message,
                "visual_state": visual_state.replace("\\n", "\n"),
                "position": position,
                "done": done,
            }
        )

    return states


def create_trajectory_report(log_file: str) -> str:
    """Create a detailed trajectory report."""

    with open(log_file, "r") as f:
        log_content = f.read()

    tool_calls = extract_tool_calls_from_log(log_content)
    agent_messages = extract_agent_messages(log_content)
    game_states = extract_game_state_changes(log_content)

    report = []
    report.append("FROZEN LAKE AGENT TRAJECTORY ANALYSIS")
    report.append("=" * 50)
    report.append("")

    # Summary
    report.append(f"📊 SUMMARY:")
    report.append(f"• Total tool calls: {len(tool_calls)}")
    report.append(f"• Total reasoning steps: {len([m for m in agent_messages if m['type'] == 'thinking'])}")
    report.append(f"• Game state changes: {len(game_states)}")
    report.append("")

    # Detailed trajectory
    report.append("🎮 DETAILED TRAJECTORY:")
    report.append("-" * 30)
    report.append("")

    for i, tool_call in enumerate(tool_calls):
        step_num = i + 1
        report.append(f"STEP {step_num}: {tool_call['tool_name'].upper()}")
        report.append(f"Arguments: {tool_call['arguments']}")

        # Add corresponding game state if available
        if i < len(game_states):
            state = game_states[i]
            report.append(f"Result: {state['message']}")
            report.append(f"Position: {state['position']}")
            report.append(f"Visual State:")
            for line in state["visual_state"].split("\n"):
                if line.strip():
                    report.append(f"  {line}")
            report.append(f"Game Done: {state['done']}")

        report.append("")

    # Agent reasoning analysis
    report.append("🧠 AGENT REASONING:")
    report.append("-" * 20)
    report.append("")

    thinking_messages = [m for m in agent_messages if m["type"] == "thinking"]
    for i, message in enumerate(thinking_messages[:3]):  # Show first 3 reasoning steps
        report.append(f"REASONING STEP {i+1}:")
        # Truncate long reasoning for readability
        content = message["content"]
        if len(content) > 500:
            content = content[:500] + "...[truncated]"
        report.append(content)
        report.append("")

    if len(thinking_messages) > 3:
        report.append(f"... and {len(thinking_messages) - 3} more reasoning steps")
        report.append("")

    # Game progression analysis
    report.append("📍 GAME PROGRESSION:")
    report.append("-" * 20)
    report.append("")

    positions = []
    for state in game_states:
        try:
            pos = eval(state["position"])  # Convert string representation to list
            positions.append(pos)
        except (SyntaxError, NameError, TypeError, ValueError):
            positions.append(state["position"])

    if positions:
        report.append("Path taken:")
        for i, pos in enumerate(positions):
            if i == 0:
                report.append(f"  Start: {pos}")
            else:
                prev_pos = positions[i - 1]
                direction = get_direction(prev_pos, pos)
                report.append(f"  Step {i}: {prev_pos} → {pos} ({direction})")

        # Final position
        if positions:
            final_pos = positions[-1]
            # Check if reached goal (typically at [3,3])
            if final_pos == [3, 3]:
                report.append(f"  🎉 GOAL REACHED at {final_pos}!")
            else:
                report.append(f"  Final position: {final_pos}")

    return "\n".join(report)


def get_direction(from_pos: List[int], to_pos: List[int]) -> str:
    """Determine the direction of movement."""
    if len(from_pos) != 2 or len(to_pos) != 2:
        return "unknown"

    row_diff = to_pos[0] - from_pos[0]
    col_diff = to_pos[1] - from_pos[1]

    if row_diff == 0 and col_diff == 1:
        return "RIGHT"
    elif row_diff == 0 and col_diff == -1:
        return "LEFT"
    elif row_diff == 1 and col_diff == 0:
        return "DOWN"
    elif row_diff == -1 and col_diff == 0:
        return "UP"
    else:
        return f"DIAGONAL({row_diff},{col_diff})"


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_trajectory.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]

    if not Path(log_file).exists():
        print(f"Error: Log file {log_file} not found")
        sys.exit(1)

    try:
        report = create_trajectory_report(log_file)

        # Save to analysis file
        analysis_file = str(Path(log_file).with_suffix(".analysis.txt"))
        with open(analysis_file, "w") as f:
            f.write(report)

        print(report)
        print(f"\n📄 Analysis saved to: {analysis_file}")

    except Exception as e:
        print(f"Error analyzing trajectory: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
