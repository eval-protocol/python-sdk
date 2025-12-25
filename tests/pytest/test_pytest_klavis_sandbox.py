import json
import logging
import os

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import KlavisSandboxRolloutProcessor, evaluation_test
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ResponseFormat(BaseModel):
    score: float
    

def klavis_gmail_sandbox_dataset_adapter(rows: list[dict]) -> list[EvaluationRow]:
    """Dataset adapter for sandbox JSONL rows.

    Supports the new schema:
      - initialize_data: dict (passed to Klavis sandbox initializer)
      - messages: str (task instruction)
      - ground_truth: dict (expected final sandbox state)

    """
    adapted: list[EvaluationRow] = []
    system_prompt = (
        "You are a helpful assistant with access to Gmail. "
        "You can send emails, draft emails, and manage messages, etc."
    )

    for r in rows:
        if isinstance(r.get("messages"), str) and "initialize_data" in r:
            init_data = r.get("initialize_data") or {}
            task = r.get("messages") or ""
            ground_truth = r.get("ground_truth")

            row = EvaluationRow(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=task),
                ],
                ground_truth=ground_truth,
            )
            row.input_metadata.session_data = {
                "initialize_data": init_data,
                "task": task,
            }
            adapted.append(row)
        else:
            adapted.append(EvaluationRow(**r))

    return adapted


@evaluation_test(
    input_dataset=["tests/pytest/datasets/klavis_gmail_sandbox_test.jsonl"],
    rollout_processor=KlavisSandboxRolloutProcessor(
        server_name="gmail",
    ),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/kimi-k2-thinking"}],
    mode="pointwise",
    dataset_adapter=klavis_gmail_sandbox_dataset_adapter,
)
async def test_pytest_gmail_sandbox(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluate Gmail sandbox results by comparing with ground truth using LLM judge.
    
    The sandbox data is exported after agent execution and compared with expected output.
    Sandbox data is available in row.execution_metadata.extra["sandbox_data"].
    """
    ground_truth = row.ground_truth
    sandbox_data = row.execution_metadata.extra.get("sandbox_data", {}) if row.execution_metadata.extra else {}
    final_message = row.messages[-1].content if row.messages else ""
    initialize_data = (row.input_metadata.session_data or {}).get("initialize_data", {})
    task = (row.input_metadata.session_data or {}).get("task", "")

    logger.info(f"Evaluating row {row.execution_metadata.rollout_id}")
    logger.info(f"Final message: {final_message}")
    logger.info(f"Sandbox data: {json.dumps(sandbox_data, indent=2, default=str)}")
    logger.info(f"Ground truth: {ground_truth}")

    async with AsyncOpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1"
    ) as client:

        evaluation_prompt = f"""You are evaluating an AI agent's performance on a Gmail sandbox task.

Task:
{task or (row.messages[-1].content if row.messages else 'N/A')}

Initial Gmail Sandbox State (initialize_data):
{json.dumps(initialize_data, indent=2, default=str)}

Expected Final Gmail Sandbox State (ground_truth):
{json.dumps(ground_truth, indent=2, default=str)}

Gmail Sandbox State After Execution:
{json.dumps(sandbox_data, indent=2, default=str)}

Evaluate whether the agent successfully completed the task by checking:
1. Does the final sandbox state match the expected ground_truth state?
2. If there are small formatting differences, judge semantically
3. Use the initial state only as context; the key is whether the correct changes happened.

Return:
- score: 1.0 if task completed successfully, 0.5 if partially completed, 0.0 if failed

"""

        try:
            response = await client.chat.completions.create(
                model="accounts/fireworks/models/kimi-k2-thinking",
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

            row.evaluation_result = EvaluateResult(score=score)
        except Exception as e:
            logger.error(f"Error during LLM evaluation: {str(e)}", exc_info=True)
            row.evaluation_result = EvaluateResult(
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )

    return row
