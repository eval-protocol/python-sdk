from datetime import datetime
from typing import List

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import default_agent_rollout_processor, evaluation_test


@evaluation_test(
    input_messages=[
        [
            {
                "role": "user",
                "content": "Can you give a summary of the past week in the 'general, model-requests, bug-reports, questions, and feature-requests' channels. For EVERY message or thread has not been resolved, please list them at the end of your response in a table. Be sure to include the exact message, severity, and current status so far. Current Date & Time: {current_date_time}".format(
                    current_date_time=datetime.now().strftime("%B %d, %Y at %I:%M %p")
                ),
            }
        ]
    ],
    rollout_processor=default_agent_rollout_processor,
    model=["fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"],
)
def test_pytest_default_agent_rollout_processor(input_dataset: List[EvaluationRow], model) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    return input_dataset
