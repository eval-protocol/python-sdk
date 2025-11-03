import asyncio
import logging
import os
import time
from typing import List

import litellm
from litellm import acompletion
from litellm.types.utils import ModelResponse, Choices
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message
from openai.types import CompletionUsage
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

logger = logging.getLogger(__name__)


class SingleTurnRolloutProcessor(RolloutProcessor):
    """Single turn rollout processor for direct LLM calls."""

    def __init__(self, *, drop_trailing_assistant_messages: bool = True) -> None:
        """
        Args:
            drop_trailing_assistant_messages: When True (default), strip any trailing
                assistant messages from the input conversation before calling the model.
                This helps when datasets include previous assistant turns and you want
                the model to answer the latest user query.
        """
        self.drop_trailing_assistant_messages = drop_trailing_assistant_messages

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Generate single turn rollout tasks and return them for external handling."""
        # Do not modify global LiteLLM cache. Disable caching per-request instead.

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row asynchronously."""
            start_time = time.perf_counter()

            if len(row.messages) == 0:
                raise ValueError("Messages is empty. Please provide a non-empty dataset")

            # Optionally drop trailing assistant messages for single-turn prompts
            messages_for_request: List[Message] = list(row.messages)
            if self.drop_trailing_assistant_messages:
                while messages_for_request and messages_for_request[-1].role == "assistant":
                    messages_for_request.pop()

            # Filter out fields that are not supported by OpenAI/LiteLLM APIs (e.g., weight, control_plane_step, reasoning_content)
            # Use the Message class method that excludes unsupported fields
            messages_payload = [message.dump_mdoel_for_chat_completion_request() for message in messages_for_request]

            request_params = {"messages": messages_payload, **config.completion_params}
            # Ensure caching is disabled only for this request (review feedback)
            request_params["cache"] = {"no-cache": True}
            # Single-level reasoning effort: expect `reasoning_effort` only
            effort_val = None

            if (
                "reasoning_effort" in config.completion_params
                and config.completion_params["reasoning_effort"] is not None
            ):
                effort_val = str(config.completion_params["reasoning_effort"])  # flat shape
            elif (
                isinstance(config.completion_params.get("extra_body"), dict)
                and "reasoning_effort" in config.completion_params["extra_body"]
                and config.completion_params["extra_body"]["reasoning_effort"] is not None
            ):
                # Accept if user passed it directly inside extra_body
                effort_val = str(config.completion_params["extra_body"]["reasoning_effort"])  # already in extra_body

            if effort_val:
                # Always under extra_body so LiteLLM forwards to provider-specific param set
                request_params.setdefault("extra_body", {})
                request_params["extra_body"]["reasoning_effort"] = effort_val
                # Ensure unsupported top-level keys are not present
                if "reasoning_effort" in request_params:
                    request_params.pop("reasoning_effort", None)

            if row.tools is not None:
                request_params["tools"] = row.tools

            if request_params.get("stream") is True:
                chunks = []
                stream = await acompletion(**request_params)

                assert isinstance(stream, CustomStreamWrapper), "Stream should be a CustomStreamWrapper"

                async for chunk in stream:  # pyright: ignore[reportGeneralTypeIssues]
                    chunks.append(chunk)
                response = litellm.stream_chunk_builder(chunks, messages_payload)
            else:
                response = await acompletion(**request_params)

            assert response is not None, "Response is None"
            assert isinstance(response, ModelResponse), "Response should be ModelResponse"
            assert isinstance(response.choices[0], Choices), "Response choice should be a Choices"

            assistant_content = response.choices[0].message.content or ""
            tool_calls = response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else None

            converted_tool_calls = None
            if tool_calls:
                converted_tool_calls = []
                for tool_call in tool_calls:
                    try:
                        converted_tool_calls.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )
                    except Exception:
                        # best-effort: fallback to dict form
                        try:
                            converted_tool_calls.append(
                                {
                                    "id": getattr(tool_call, "id", "toolcall_0"),
                                    "type": getattr(tool_call, "type", "function"),
                                    "function": {
                                        "name": getattr(getattr(tool_call, "function", None), "name", "tool"),
                                        "arguments": getattr(getattr(tool_call, "function", None), "arguments", "{}"),
                                    },
                                }
                            )
                        except Exception:
                            pass

            messages = list(messages_for_request) + [
                Message(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=converted_tool_calls,
                )
            ]
            row.execution_metadata.usage = (
                CompletionUsage(  # Note: LiteLLM sets usage dynamically via setattr(), not as a typed field
                    prompt_tokens=response.usage.prompt_tokens,  # pyright: ignore[reportAttributeAccessIssue]
                    completion_tokens=response.usage.completion_tokens,  # pyright: ignore[reportAttributeAccessIssue]
                    total_tokens=response.usage.total_tokens,  # pyright: ignore[reportAttributeAccessIssue]
                )
            )

            row.messages = messages

            row.execution_metadata.duration_seconds = time.perf_counter() - start_time

            default_logger.log(row)
            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await process_row(r)
                return result

        # Create and return tasks for external handling
        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks
