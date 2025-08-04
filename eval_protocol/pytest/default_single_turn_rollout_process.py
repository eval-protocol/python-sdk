import asyncio
from typing import List

from openai import AsyncOpenAI

from eval_protocol.auth import get_fireworks_api_base, get_fireworks_api_key
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import RolloutProcessorConfig


async def default_single_turn_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> List[EvaluationRow]:
    """Generate a single response from a Fireworks model concurrently."""

    api_key = get_fireworks_api_key()
    api_base = get_fireworks_api_base()
    client = AsyncOpenAI(api_key=api_key, base_url=f"{api_base}/inference/v1")

    async def process_row(row: EvaluationRow) -> EvaluationRow:
        """Process a single row asynchronously."""
        if len(row.messages) == 0:
            raise ValueError("Messages is empty. Please provide a non-empty dataset")

        messages_payload = [{"role": m.role, "content": m.content} for m in row.messages]

        response = await client.chat.completions.create(
            model=config.model, messages=messages_payload, tools=row.tools, **config.input_params
        )
        assistant_content = response.choices[0].message.content or ""
        tool_calls = response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else None
        messages = list(row.messages) + [
            Message(
                role="assistant",
                content=assistant_content,
                tool_calls=tool_calls,
            )
        ]

        return EvaluationRow(
            messages=messages,
            **row.model_dump(exclude={"messages"}),
        )

    # Process all rows concurrently
    tasks = [process_row(row) for row in rows]
    dataset = await asyncio.gather(*tasks)

    return dataset
