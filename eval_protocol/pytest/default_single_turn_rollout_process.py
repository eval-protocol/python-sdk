import asyncio
from typing import List

from litellm import acompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessageToolCall

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import RolloutProcessorConfig


async def default_single_turn_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> List[EvaluationRow]:
    """Generate a single response from any supported model provider using LiteLLM."""

    async def process_row(row: EvaluationRow) -> EvaluationRow:
        """Process a single row asynchronously."""
        if len(row.messages) == 0:
            raise ValueError("Messages is empty. Please provide a non-empty dataset")

        messages_payload = [{"role": m.role, "content": m.content} for m in row.messages]

        request_params = {"model": config.model, "messages": messages_payload, **config.input_params}

        if row.tools is not None:
            request_params["tools"] = row.tools

        response = await acompletion(**request_params)

        assistant_content = response.choices[0].message.content or ""
        tool_calls = response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else None

        converted_tool_calls = None
        if tool_calls:
            converted_tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function={
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                )
                for tool_call in tool_calls
            ]

        messages = list(row.messages) + [
            Message(
                role="assistant",
                content=assistant_content,
                tool_calls=converted_tool_calls,
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
