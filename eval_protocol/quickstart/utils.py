"""
Arena-Hard-Auto utility functions adapted for Eval Protocol.
"""

import time
import re
import shortuuid
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import pandas as pd

from eval_protocol.models import EvaluationRow, Message

OG_ARENA_HARD_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."


def get_score(judgment, patterns):
    """Extract judgment score from text. From arena-hard-auto/gen_judgment.py"""
    for pattern in patterns:
        pattern = re.compile(pattern)

        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]

        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def pairwise_judgment(question_text, answer_a, answer_b):
    """Pairwise judgment function. Adapted from arena-hard-auto/gen_judgment.py"""
    user_prompt = f"<|User Prompt|>\n{question_text}\n\n<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"

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

    # Use OpenAI API directly
    try:
        from openai import OpenAI

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,  # type: ignore
            temperature=0.0,
            max_tokens=16000,
        )

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
