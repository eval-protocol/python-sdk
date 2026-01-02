"""Example evaluation_test that wraps deepeval's GEval and captures logprobs.

To run this example you will need `deepeval` installed and a compatible
API key (e.g., OpenAI or Fireworks). You can override the base URL with
``EP_LLM_API_BASE`` or ``EP_LLM_BASE_URL`` and pass provider-specific
parameters through ``completion_params``. Logs are written to
``~/.eval_protocol/datasets/<YYYY-MM-DD>.jsonl`` via the local filesystem
logger so you can inspect the captured logprobs directly.

Environment variables:
    FIREWORKS_API_KEY - Required for Fireworks models
"""

import os
from typing import List

from eval_protocol.dataset_logger.local_fs_dataset_logger_adapter import LocalFSDatasetLoggerAdapter
from eval_protocol.integrations.deepeval import adapt_metric
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

try:  # pragma: no cover - optional dependency for the example
    from deepeval.metrics import GEval
    from deepeval.models import LiteLLMModel
    from deepeval.test_case import LLMTestCaseParams
except ImportError as exc:  # pragma: no cover - optional dependency for the example
    raise ImportError("Install deepeval to run this example: pip install deepeval") from exc

# Use DeepSeek via Fireworks for the GEval judge model
# Note: We need allowed_openai_params to enable top_logprobs for GEval's score normalization
judge_model = LiteLLMModel(
    model="fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
    api_key=os.environ.get("FIREWORKS_API_KEY"),
    allowed_openai_params=["top_logprobs"],  # Enable logprobs for GEval normalization
)

# Configure GEval to judge the assistant response with the full chat context.
wrapped_metric = adapt_metric(
    GEval(
        name="Helpful & Relevant",
        criteria="Evaluate the helpfulness and relevance of the model output.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge_model,
        top_logprobs=5,  # Fireworks max is 5 (default is 20)
    )
)


@evaluation_test(
    input_rows=[[EvaluationRow(messages=[{"role": "user", "content": "Say hello politely."}])]],
    completion_params=[
        {
            "model": "fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
        },
    ],
    logger=LocalFSDatasetLoggerAdapter(),
    mode="all",
)
def test_geval_with_logprobs(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Attach GEval scores while keeping the raw logprobs on the final message."""

    for row in rows:
        eval_result = wrapped_metric(
            messages=[message.model_dump(exclude_none=True) for message in row.messages],
            ground_truth="Hello!",
        )
        row.evaluation_result = eval_result

        # Logprob payload is available on the last assistant message after rollout
        # and can be forwarded to metric metadata for debugging or analysis.
        last_assistant = row.messages[-1]
        if last_assistant.logprobs:
            metric_key = next(iter(eval_result.metrics))
            eval_result.metrics[metric_key].data["logprobs"] = last_assistant.logprobs

    return rows
