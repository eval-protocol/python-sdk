"""
Arena-Hard-Auto utility functions adapted for Eval Protocol.
"""

import os
import re
from typing import Dict, Any, Optional

from eval_protocol.models import EvaluationRow, Message

OG_ARENA_HARD_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""


# Judge model configurations for Arena-Hard-Auto style evaluation
# Each config specifies the model, parameters, and concurrency limits for LLM judges
JUDGE_CONFIGS = {
    "gpt-4.1": {
        "model": "gpt-4.1",
        "temperature": 0.0,
        "max_tokens": 16000,
    },
    "gemini-2.5-pro": {
        "model": "gemini-2.5-pro",
        "temperature": 1.0,
        "max_tokens": 32000,
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "gemini-2.5-flash": {
        "model": "gemini-2.5-flash",
        "temperature": 1.0,
        "max_tokens": 32000,
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "kimi-k2-instruct-0905": {
        "model": "accounts/fireworks/models/kimi-k2-instruct-0905",
        "temperature": 0.6,  # Kimi recommended temperature
        "max_tokens": 131000,
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/inference/v1",
    },
}

LABEL_TO_SCORE = {
    "A>>B": 1.0,
    "B<<A": 1.0,
    "A>B": 6 / 7,
    "B<A": 6 / 7,
    "A=B": 0.5,
    "B=A": 0.5,
    "A<B": 1 / 7,
    "B>A": 1 / 7,
    "A<<B": 0.0,
    "B>>A": 0.0,
}


def get_score(judgment, patterns):
    """Extract judgment score from text. From arena-hard-auto/gen_judgment.py"""
    for pattern in patterns:
        pattern = re.compile(pattern)

        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]

        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def serialize_message(msg: Message) -> str:
    parts = [f"{msg.role}: {msg.content}"]

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            parts.append(f"[Tool Call: {tool_name}({tool_args})]")

    return "\n".join(parts)


def multi_turn_assistant_to_ground_truth(data: list[EvaluationRow]) -> list[EvaluationRow]:
    """
    Split multi-turn conversations into rows, with each assistant message as ground truth.

    Args:
        data: List of EvaluationRow objects

    Returns:
        List of expanded EvaluationRow objects, one for each assistant message
    """
    expanded_rows = []
    seen_traces: set[str] = set()

    for row in data:
        messages = row.messages
        tools = row.tools
        input_metadata = row.input_metadata

        assistant_positions = []
        for i, message in enumerate(messages):
            if message.role == "assistant":
                assistant_positions.append(i)

        # Create separate evaluation rows on each assistant message (where the comparison model will respond)
        for pos in assistant_positions:
            messages_before_assistant = messages[:pos]
            assistant_message = messages[pos]

            # In this case, we trace every request, so we need to filter out duplicates
            curr_trace = "\n".join(serialize_message(m) for m in messages_before_assistant)
            if curr_trace in seen_traces:
                continue
            seen_traces.add(curr_trace)

            ground_truth_message = serialize_message(assistant_message)

            expanded_rows.append(
                EvaluationRow(
                    messages=messages_before_assistant,
                    tools=tools,
                    input_metadata=input_metadata,
                    ground_truth=ground_truth_message,
                )
            )

    return expanded_rows


def assistant_to_ground_truth(data: list[EvaluationRow]) -> list[EvaluationRow]:
    """
    Extract the last assistant message as ground truth and remove it from the conversation.

    Args:
        data: List of EvaluationRow objects

    Returns:
        List of EvaluationRow objects with last assistant message moved to ground_truth
    """
    processed_rows = []

    for row in data:
        messages = row.messages.copy()  # Don't modify original

        if messages[-1].role == "assistant":
            assistant_message = messages[-1]
            messages = messages[:-1]
            ground_truth_message = serialize_message(assistant_message)
        else:
            raise ValueError("Last message is not from assistant")

        processed_rows.append(
            EvaluationRow(
                messages=messages,
                tools=row.tools,
                input_metadata=row.input_metadata,
                ground_truth=ground_truth_message,
            )
        )

    return processed_rows


def filter_longest_conversation(data: list[EvaluationRow]) -> list[EvaluationRow]:
    """
    Filter out the longest conversation from a list of evaluation rows that share the same rollout_id.

    Args:
        data: List of EvaluationRow objects that share the same rollout_id

    Returns:
        List containing only the EvaluationRow with the most messages (longest conversation)
    """
    if not data:
        return data

    if len(data) == 1:
        return data

    # Find the row with the most messages (longest conversation)
    longest_row = max(data, key=lambda row: len(row.messages))

    return [longest_row]


async def run_single_judgment(
    question_text: str, answer_a: str, answer_b: str, tools, judge_config, client
) -> Optional[Dict[str, Any]]:
    """Run a single pairwise judgment between two answers."""
    user_prompt = f"""<|User Prompt|>
{question_text}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>

<|Available Tools|>
{tools}
<|End of Available Tools|>

{OG_ARENA_HARD_PROMPT}"""

    messages = [{"role": "user", "content": user_prompt}]

    try:
        api_params = {
            "model": judge_config["model"],
            "messages": messages,
            "temperature": judge_config["temperature"],
            "max_tokens": judge_config["max_tokens"],
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "none"

        response = await client.chat.completions.create(**api_params)
        judgment_text = response.choices[0].message.content
        if not judgment_text:
            return None

    except Exception as e:
        print(f"Error getting judgment from OpenAI: {e}")
        return None

    score = get_score(judgment_text, [r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"])
    return {"score": score, "judgment": judgment_text, "prompt": messages}
