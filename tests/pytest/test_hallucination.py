"""
Hallucination detection test using LLM-as-judge.

This test demonstrates how to detect factual inaccuracies in model responses
by comparing them against provided knowledge using an LLM judge, similar to
tau's evaluate_nl_assertions approach.
"""

import json
from typing import Any, Dict, List
import pytest

import litellm

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

# Configure the judge model for LiteLLM
JUDGE_MODEL = "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"


def hallucination_dataset_adapter(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """Convert HaluEval dataset to EvaluationRow objects."""
    return [
        EvaluationRow(
            messages=[Message(role="user", content=f"Knowledge: {item['knowledge']}\n\nQuestion: {item['question']}")],
            ground_truth=item["right_answer"],
        )
        for item in data
    ]


@pytest.mark.asyncio
@evaluation_test(
    input_dataset=["tests/pytest/data/halueval_sample_dataset.jsonl"],
    dataset_adapter=hallucination_dataset_adapter,
    completion_params=[
        {"temperature": 0.0, "max_tokens": 512, "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"}
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=0.33,
    num_runs=1,
    mode="pointwise",
)
async def test_hallucination_detection(row: EvaluationRow) -> EvaluationRow:
    """
    Test for response correctness using LLM-as-judge.
    """
    messages = row.messages
    assistant_response = messages[-1].content

    if not assistant_response:
        return EvaluateResult(score=0.0, reason="❌ No assistant response found")

    correct_answer = row.ground_truth

    system_prompt = """
    TASK
    - You will be given an assistant's response and the correct answer.
    - Your job is to evaluate whether the assistant's response is factually consistent with the correct answer.
    - Grade whether the assistant got it right or wrong.

    FORMAT
    - Your response should be a JSON object with the following fields:
    - `reasoning`: a short explanation for your classification
    - `is_correct`: `true` if the assistant's response matches the correct answer, `false` otherwise

    Example response structure:
    {
        "reasoning": "<reasoning trace>",
        "is_correct": <true or false>
    }
    """

    user_prompt = f"""
    assistant_response:
    {assistant_response}

    correct_answer:
    {correct_answer}
    """

    try:
        response = await litellm.acompletion(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        result_data = json.loads(response.choices[0].message.content)
        is_correct = result_data.get("is_correct", False)
        reasoning = result_data.get("reasoning", "Could not parse reasoning")

    except Exception as e:
        # Fallback if parsing fails
        is_correct = False
        reasoning = f"Evaluation failed: {str(e)}"

    score = 1.0 if is_correct else 0.0

    if is_correct:
        assessment = "✅ Response is correct"
    else:
        assessment = "❌ Response is incorrect"

    reason = f"{assessment}\nReasoning: {reasoning}"

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=reason,
        metrics={"llm_judge": MetricResult(score=score, reason=reasoning, is_score_valid=True)},
    )

    return row
