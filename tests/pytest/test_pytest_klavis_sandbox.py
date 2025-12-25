import json
import logging
import os

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import KlavisSandboxRolloutProcessor, evaluation_test
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ResponseFormat(BaseModel):
    score: float
    reasoning: str


@evaluation_test(
    input_dataset=["tests/pytest/datasets/klavis_gmail_sandbox_test.jsonl"],
    rollout_processor=KlavisSandboxRolloutProcessor(
        server_name="gmail",
        # Optional: provide custom initialization data factory
        # initialize_data_factory=lambda row: {"messages": [], "drafts": []},
    ),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/deepseek-v3p2"}],
    mode="pointwise",
)
async def test_pytest_gmail_sandbox(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluate Gmail sandbox results by comparing with ground truth using LLM judge.
    
    The sandbox data is exported after agent execution and compared with expected output.
    Sandbox data is available in row.execution_metadata.metadata["sandbox_data"].
    """
    ground_truth = row.ground_truth
    sandbox_data = row.execution_metadata.extra.get("sandbox_data", {}) if row.execution_metadata.extra else {}
    final_message = row.messages[-1].content if row.messages else ""

    logger.info(f"Evaluating row {row.execution_metadata.rollout_id}")
    logger.info(f"Final message: {final_message}")
    logger.info(f"Sandbox data: {json.dumps(sandbox_data, indent=2, default=str)}")
    logger.info(f"Ground truth: {ground_truth}")

    async with AsyncOpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1"
    ) as client:
        # Use LLM to judge if the sandbox data matches the ground truth
        evaluation_prompt = f"""You are evaluating an AI agent's performance on a Gmail task.

Task: {row.messages[0].content if row.messages else 'N/A'}

Ground Truth: {ground_truth}

Agent's Final Response: {final_message}

Gmail Sandbox State After Execution:
{json.dumps(sandbox_data, indent=2, default=str)}

Evaluate whether the agent successfully completed the task by checking:
1. Did the agent understand and attempt the task?
2. Does the sandbox data reflect the expected outcome described in the ground truth?
3. Are there any emails sent/drafted that match the task requirements?

Return:
- score: 1.0 if task completed successfully, 0.5 if partially completed, 0.0 if failed
- reasoning: Explain your evaluation in 1-2 sentences
"""

        try:
            response = await client.chat.completions.create(
                model="accounts/fireworks/models/deepseek-v3p2",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise evaluator of AI agent performance. Analyze the task, execution, and results carefully.",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "ResponseFormat", "schema": ResponseFormat.model_json_schema()},
                },
                temperature=0.0,
            )

            response_text = response.choices[0].message.content
            logger.info(f"LLM judge response: {response_text}")

            parsed = json.loads(response_text or "{}")
            score = parsed.get("score", 0.0)
            reasoning = parsed.get("reasoning", "No reasoning provided")

            row.evaluation_result = EvaluateResult(
                score=score,
                reason=reasoning,
            )
        except Exception as e:
            logger.error(f"Error during LLM evaluation: {str(e)}", exc_info=True)
            row.evaluation_result = EvaluateResult(
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )

    return row
