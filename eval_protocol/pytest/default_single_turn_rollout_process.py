import asyncio
from typing import List

from litellm import acompletion
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessageToolCall,
    FunctionCall,
)

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

        messages_payload = []
        for m in row.messages:
            payload = {"role": m.role}
            if m.content is not None:
                payload["content"] = m.content
            if m.name is not None:
                payload["name"] = m.name
            if m.tool_call_id is not None:
                payload["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                payload["tool_calls"] = [
                    tc.model_dump(exclude_none=True) for tc in m.tool_calls
                ]
            if m.function_call is not None:
                payload["function_call"] = m.function_call.model_dump(
                    exclude_none=True
                )
            messages_payload.append(payload)

        request_params = {"model": config.model, "messages": messages_payload, **config.input_params}

        if row.tools is not None:
            request_params["tools"] = row.tools

        response = await acompletion(**request_params)

        assistant_message = response.choices[0].message
        assistant_content = assistant_message.content or ""
        tool_calls = assistant_message.tool_calls if assistant_message.tool_calls else None
        function_call = assistant_message.function_call

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
                function_call=function_call,
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
