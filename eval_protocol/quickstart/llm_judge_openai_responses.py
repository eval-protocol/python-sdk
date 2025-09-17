"""
LLM Judge quickstart that PULLS DATA FROM OpenAI Responses API and persists results locally via Eval Protocol.

This mirrors `eval_protocol/quickstart/llm_judge.py` (Langfuse source), but uses
OpenAI Responses API as the source of evaluation rows.

Env vars:
  export OPENAI_API_KEY=...             # required to fetch examples

Judge model keys:
  - Default judge is "gemini-2.5-pro" from utils; requires GEMINI_API_KEY
  - Or set judge in the code to "gpt-4.1" and export OPENAI_API_KEY

Run:
  pytest python-sdk/eval_protocol/quickstart/llm_judge_openai_responses.py -q -s
"""

import os
from typing import List

import pytest

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.quickstart import aha_judge, split_multi_turn_rows
from eval_protocol.adapters.openai_responses import OpenAIResponsesAdapter

adapter = OpenAIResponsesAdapter()
input_rows = adapter.get_evaluation_rows(
    response_ids=[
        "resp_0e1b7db5d96e92470068c99506443c819e9305e92915d2405f",
        "resp_05639dcaca074fbc0068c9946593b481908cac70075926d85c",
    ]
)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")  # pyright: ignore[reportAttributeAccessIssue]
@pytest.mark.asyncio  # pyright: ignore[reportAttributeAccessIssue]
@evaluation_test(
    input_rows=[input_rows],
    completion_params=[
        {
            "model": "fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
        },
        {
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905",
        },
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    preprocess_fn=split_multi_turn_rows,
    mode="all",
)
async def test_llm_judge_openai_responses(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    return await aha_judge(rows)
