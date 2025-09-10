"""
Arena-Hard-Auto utility functions adapted for Eval Protocol.
"""

import re

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


def split_multi_turn_rows(data: list[EvaluationRow]) -> list[EvaluationRow]:
    """
    Split multi-turn conversation rows into individual evaluation rows for each assistant message.

    Args:
        data: List of EvaluationRow objects

    Returns:
        List of expanded EvaluationRow objects, one for each assistant message
    """
    expanded_rows = []

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


def pairwise_judgment(question_text, answer_a, answer_b, tools, judge_config):
    """Pairwise judgment function. Adapted from arena-hard-auto/gen_judgment.py"""
    user_prompt = f"""<|User Prompt|>
{question_text}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""

    messages = [
        {
            "role": "system",
            "content": OG_ARENA_HARD_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    try:
        from openai import OpenAI

        client = OpenAI(api_key=judge_config["api_key"], base_url=judge_config["base_url"])

        api_params = {
            "model": judge_config["model"],
            "messages": messages,  # type: ignore
            "temperature": judge_config["temperature"],
            "max_tokens": judge_config["max_tokens"],
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = (
                "none"  # Judge can see tools to help in response, but won't actually try to call them
            )

        response = client.chat.completions.create(**api_params)

        judgment_text = response.choices[0].message.content
        if not judgment_text:
            return None

    except Exception as e:
        print(f"Error getting judgment from OpenAI: {e}")
        return None

    score = get_score(judgment_text, [r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"])

    result = {
        "score": score,
        "judgment": judgment_text,
        "prompt": messages,
    }
    return result
