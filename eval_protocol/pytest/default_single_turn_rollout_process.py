from typing import List

from openai import OpenAI

from eval_protocol.auth import get_fireworks_api_base, get_fireworks_api_key
from eval_protocol.models import CompletionParams, EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.types import Dataset, ModelParam, RolloutProcessorConfig


def default_single_turn_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> List[EvaluationRow]:
    """Generate a single response from a Fireworks model."""

    api_key = get_fireworks_api_key()
    api_base = get_fireworks_api_base()
    client = OpenAI(api_key=api_key, base_url=f"{api_base}/inference/v1")

    dataset: Dataset = []
    for row in rows:
        if len(row.messages) == 0:
            raise ValueError("Messages is empty. Please provide a non-empty dataset")

        messages_payload = [{"role": m.role, "content": m.content} for m in row.messages]

        response = client.chat.completions.create(model=config.model, messages=messages_payload, **config.input_params)
        assistant_content = response.choices[0].message.content or ""
        messages = list(row.messages) + [Message(role="assistant", content=assistant_content)]
        processed = EvaluationRow(
            messages=messages,
            ground_truth=row.ground_truth,
            input_metadata=InputMetadata(completion_params=CompletionParams(model=config.model)),
        )

        dataset.append(processed)
    return dataset
