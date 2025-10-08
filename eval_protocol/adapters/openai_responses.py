"""Langfuse adapter for Eval Protocol.

This adapter allows pulling data from Langfuse deployments and converting it
to EvaluationRow format for use in evaluation pipelines.
"""

from collections.abc import Iterable, Sequence
import logging
from typing import List
from typing_extensions import Any

from openai.pagination import SyncCursorPage
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.responses import Response
from openai.types.responses.response_item import ResponseItem
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.responses.tool import Tool

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from .base import BaseAdapter

logger = logging.getLogger(__name__)


from openai import OpenAI


class OpenAIResponsesAdapter(BaseAdapter):
    """Adapter to pull data from OpenAI Responses API and convert to EvaluationRow format.

    This adapter can pull both chat conversations and tool calling traces from
    Langfuse deployments and convert them into the EvaluationRow format expected
    by the evaluation protocol.

    Examples:
        Basic usage:
        >>> adapter = OpenAIResponsesAdapter(
        ...     api_key="your_api_key",
        ... )
        >>> rows = list(adapter.get_evaluation_rows(respnse_ids=["response_id_1", "response_id_2"]))
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the OpenAI Responses adapter."""
        self.openai = OpenAI(api_key=api_key, base_url=base_url)

    def get_evaluation_rows(
        self,
        response_ids: List[str],
    ) -> List[EvaluationRow]:
        """Pull responses from OpenAI Responses API and convert to EvaluationRow format.

        Args:
            response_ids: List of response IDs to fetch
        Returns:
            List[EvaluationRow]: Converted evaluation rows
        """
        eval_rows: list[EvaluationRow] = []

        for response_id in response_ids:
            input_items = self.openai.responses.input_items.list(response_id=response_id)
            response = self.openai.responses.retrieve(response_id=response_id)
            eval_rows.append(self._create_evaluation_row(input_items, response))

        logger.info(
            "Successfully processed %d selected traces into %d evaluation rows", len(response_ids), len(eval_rows)
        )
        return eval_rows

    def _create_evaluation_row(self, input_items: SyncCursorPage[ResponseItem], response: Response) -> EvaluationRow:
        """Convert a response to an evaluation row."""
        messages: list[Message] = []
        if response.instructions:
            if isinstance(response.instructions, list):
                raise NotImplementedError("List of instructions is not supported")
            else:
                messages.append(Message(role="system", content=response.instructions))
        messages.extend(self._create_messages(input_items))
        if response.output_text:
            messages.append(Message(role="assistant", content=response.output_text))
        tools = self._responses_tools_to_chat_completion_tools(response.tools)
        tool_dicts = [dict(tool) for tool in tools]
        return EvaluationRow(
            messages=messages,
            tools=tool_dicts,
            input_metadata=InputMetadata(
                completion_params={
                    "model": response.model,
                    "temperature": response.temperature,
                    "max_output_tokens": response.max_output_tokens,
                    "max_tool_calls": response.max_tool_calls,
                    "parallel_tool_calls": response.parallel_tool_calls,
                    """
                    We have to manually extract the reasoning effort and summary
                    from the response.reasoning object because the openai-python
                    causes an issue with model_dump() which is used for testing.

                    https://github.com/openai/openai-python/issues/1306#issuecomment-2966267356
                    """
                    "reasoning": {
                        "effort": response.reasoning.effort,
                        "summary": response.reasoning.summary,
                    }
                    if response.reasoning
                    else None,
                    "top_logprobs": response.top_logprobs,
                    "truncation": response.truncation,
                    "top_p": response.top_p,
                }
            ),
        )

    def _responses_tools_to_chat_completion_tools(self, tools: List[Tool]) -> Sequence[ChatCompletionToolParam]:
        """Convert OpenAI Responses API tools to chat completion message function tool calls."""
        chat_completion_tools: List[ChatCompletionToolParam] = []
        for tool in tools:
            if tool.type == "function":
                chat_completion_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "parameters": tool.parameters or {},
                            "strict": tool.strict,
                            "description": tool.description or "",
                        },
                    }
                )
            else:
                raise NotImplementedError("Only function tools are supported")
        return chat_completion_tools

    def _create_messages(self, input_items: SyncCursorPage[ResponseItem]) -> Iterable[Message]:
        """Create messages from input items.

        Converts OpenAI Responses API input items to chat completion message format.
        Handles different types of response items including messages and tool calls.
        Groups parallel tool calls under a single assistant message.
        Since we iterate backwards and reverse at the end, tool call outputs should
        be added before the assistant message with tool calls.
        """
        messages: list[Message] = []
        current_tool_calls: list[ChatCompletionMessageToolCall] = []
        tool_call_outputs: list[Message] = []

        for item in input_items:
            if item.type == "message":
                # If we have accumulated tool calls, create an assistant message with them
                if current_tool_calls:
                    # Add tool call outputs first (since we reverse at the end)
                    messages.extend(tool_call_outputs)
                    tool_call_outputs = []
                    # Then add the assistant message with tool calls
                    messages.append(Message(role="assistant", tool_calls=current_tool_calls))
                    current_tool_calls = []

                # This is a message item (input or output)
                content = item.content
                for content_item in content:
                    if content_item.type == "input_text":
                        text_content = content_item.text
                        # Create new message
                        messages.append(Message(role=item.role, content=text_content))
                    else:
                        raise NotImplementedError(f"Unsupported content type: {content_item.type}")
            elif item.type == "function_call_output":
                # Collect tool call outputs to add before assistant message
                tool_call_outputs.append(
                    Message(role="tool", content=self._coerce_tool_output(item.output), tool_call_id=item.call_id)
                )
            elif item.type == "function_call":
                tool_call = ChatCompletionMessageToolCall(
                    id=item.call_id, type="function", function=Function(name=item.name, arguments=item.arguments)
                )
                current_tool_calls.append(tool_call)
            else:
                raise NotImplementedError(f"Unsupported item type: {item.type}")

        # If we have remaining tool calls, create an assistant message with them
        if current_tool_calls:
            # Add tool call outputs first (since we reverse at the end)
            messages.extend(tool_call_outputs)
            # Then add the assistant message with tool calls
            messages.append(Message(role="assistant", tool_calls=current_tool_calls))

        return reversed(messages)

    def _coerce_tool_output(self, output: Any) -> str:
        """Coerce OpenAI Responses tool output into a string for Message.content.

        The Responses API may return structured content lists. For our purposes,
        we stringify non-string outputs to satisfy the Message.content type.
        """
        if isinstance(output, str):
            return output
        try:
            # Attempt to join list of objects with any 'text' fields
            if isinstance(output, list):
                parts: list[str] = []
                for part in output:
                    text = None
                    if isinstance(part, dict):
                        text = part.get("text")
                    if text:
                        parts.append(str(text))
                    else:
                        parts.append(str(part))
                return "\n".join(parts)
            # Fallback to string conversion
            return str(output)
        except Exception:
            return str(output)
