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

import pytest

from eval_protocol import (
    evaluation_test,
    aha_judge,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    OpenAIResponsesAdapter,
    DynamicDataLoader,
    multi_turn_assistant_to_ground_truth,
)


def openai_responses_data_generator():
    adapter = OpenAIResponsesAdapter()
    return adapter.get_evaluation_rows(
        response_ids=[
            "resp_0e1b7db5d96e92470068c99506443c819e9305e92915d2405f",
            # "resp_05639dcaca074fbc0068c9946593b481908cac70075926d85c",
            # "resp_0c96a910416e87aa0068c994d0b34c81a3bda0eddf22445aec",
            # "resp_0efe023280e986f90068c994b85e088190bc8d8263fa603e02",
        ]
    )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": "fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
        },
        {
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905",
        },
    ],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[openai_responses_data_generator],
        preprocess_fn=multi_turn_assistant_to_ground_truth,
    ),
    rollout_processor=SingleTurnRolloutProcessor(),
    max_concurrent_evaluations=2,
)
async def test_llm_judge_openai_responses(row: EvaluationRow) -> EvaluationRow:
    return await aha_judge(row)
