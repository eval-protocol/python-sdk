"""
Arena-Hard-Auto utility functions adapted for Eval Protocol.
"""

import os
from datetime import datetime
import re
from typing import List, Dict, Any, Optional
import pandas as pd

from eval_protocol.models import EvaluationRow, Message, EvaluateResult, MetricResult
import asyncio


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
        "max_concurrency": 64,
    },
    "gemini-2.5-pro": {
        "model": "gemini-2.5-pro",
        "temperature": 1.0,
        "max_tokens": 32000,
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "max_concurrency": 16,
    },
    "gemini-2.5-flash": {
        "model": "gemini-2.5-flash",
        "temperature": 1.0,
        "max_tokens": 32000,
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "max_concurrency": 16,
    },
    "kimi-k2-instruct-0905": {
        "model": "accounts/fireworks/models/kimi-k2-instruct-0905",
        "temperature": 0.6,  # Kimi recommended temperature
        "max_tokens": 131000,
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/inference/v1",
        "max_concurrency": 64,
    },
}

# Mapping from Arena-Hard-Auto judgment labels to numerical scores
# Stronger preferences (>> or <<) get weighted more heavily (3x) than slight preferences
LABEL_TO_SCORE = {
    "A>B": [1],
    "A>>B": [1] * 3,
    "A=B": [0.5],
    "A<<B": [0] * 3,
    "A<B": [0],
    "B>A": [0],
    "B>>A": [0] * 3,
    "B=A": [0.5],
    "B<<A": [1] * 3,
    "B<A": [1],
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


def split_multi_turn_rows(data: list[EvaluationRow]) -> list[EvaluationRow]:
    """
    Split multi-turn conversation rows into individual evaluation rows for each assistant message.

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


async def pairwise_judgment_async(question_text, answer_a, answer_b, tools, judge_config, shared_client):
    """Async pairwise judgment using a shared client."""
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

        response = await shared_client.chat.completions.create(**api_params)
        judgment_text = response.choices[0].message.content
        if not judgment_text:
            return None

    except Exception as e:
        print(f"Error getting judgment from OpenAI: {e}")
        return None

    score = get_score(judgment_text, [r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"])
    return {"score": score, "judgment": judgment_text, "prompt": messages}


async def run_judgment_async(
    row: EvaluationRow, model_name: str, judge_name: str, shared_client
) -> Optional[Dict[str, Any]]:
    """Async judgment using shared client to avoid cleanup issues."""
    if not row.messages:
        return None

    question_text = "\n".join([serialize_message(msg) for msg in row.messages[:-1]])
    model_a_answer = row.ground_truth
    model_b_answer = serialize_message(row.messages[-1])

    # Run both rounds concurrently with shared client
    result1, result2 = await asyncio.gather(
        pairwise_judgment_async(
            question_text, model_a_answer, model_b_answer, row.tools, JUDGE_CONFIGS[judge_name], shared_client
        ),
        pairwise_judgment_async(
            question_text, model_b_answer, model_a_answer, row.tools, JUDGE_CONFIGS[judge_name], shared_client
        ),
    )

    games = [result1, result2]

    row.evaluation_result = EvaluateResult(
        score=0.0,
        reason=f"LLM Judge comparison: Round 1: {result1['score']}, Round 2: {result2['score']}"
        if result1 and result2
        else "Failed to get judgement scores",
        metrics={
            "round1_judgment": MetricResult(score=0.0, reason=result1["judgment"] if result1 else "Failed"),
            "round2_judgment": MetricResult(score=0.0, reason=result2["judgment"] if result2 else "Failed"),
        },
    )

    return {"model": model_name, "games": games}


def calculate_bootstrap_scores(judgments: List[Dict[str, Any]]) -> tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals for Arena-Hard-Auto style judgments.

    Converts judgment labels (A>B, A>>B, etc.) to numerical scores, performs bootstrap
    sampling to estimate score distribution, and returns mean with 90% confidence interval.

    Args:
        judgments: List of judgment dicts, each containing "games" with two rounds of scores

    Returns:
        tuple: (mean_score, lower_5th_percentile, upper_95th_percentile)
               Returns (0.0, 0.0, 0.0) if no valid scores found
    """
    # Extract scores from judgments
    scores_data = []
    for judgment in judgments:
        game1, game2 = judgment["games"]
        if game1 and game2 and game1.get("score") and game2.get("score"):
            # Convert judgment scores to numerical scores
            scores = LABEL_TO_SCORE[game2["score"]] + [1 - s for s in LABEL_TO_SCORE[game1["score"]]]
            for score in scores:
                scores_data.append(score)

    if not scores_data:
        return 0.0, 0.0, 0.0

    # Create DataFrame (single column of scores)
    battles = pd.DataFrame({"score": scores_data})

    # Bootstrap sampling for calculating relative performance to original model at fixed 50%
    bootstrap_means = [battles.sample(frac=1.0, replace=True)["score"].mean() for _ in range(100)]

    # Calculate final scores
    bootstraps = pd.Series(bootstrap_means)
    mean_score = bootstraps.mean()
    lower_score = bootstraps.quantile(0.05)
    upper_score = bootstraps.quantile(0.95)

    return mean_score, lower_score, upper_score


def push_scores_to_langfuse(rows: List[EvaluationRow], model_name: str, mean_score: float) -> None:
    """
    Push evaluation scores back to Langfuse traces for tracking and analysis.

    Creates a score entry in Langfuse for each unique trace_id found in the evaluation
    rows' session data. This allows you to see evaluation results directly in the
    Langfuse UI alongside the original traces.

    Args:
        rows: List of EvaluationRow objects with session_data containing trace IDs
        model_name: Name of the model (used as the score name in Langfuse)
        mean_score: The calculated mean score to push to Langfuse

    Note:
        Silently handles errors if Langfuse is unavailable or if rows lack session data
    """
    try:
        from eval_protocol.adapters.langfuse import create_langfuse_adapter

        langfuse = create_langfuse_adapter().client

        for trace_id in set(
            row.input_metadata.session_data["langfuse_trace_id"]
            for row in rows
            if row.evaluation_result and row.input_metadata and row.input_metadata.session_data
        ):
            if trace_id:
                langfuse.create_score(
                    trace_id=trace_id,
                    name=model_name,
                    value=mean_score,
                )
    except Exception as e:
        print(f"⚠️ Failed to push scores to Langfuse: {e}")
