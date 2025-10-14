import asyncio
import logging
import os
import time
from typing import List

import litellm
from litellm import acompletion
from typing import Dict

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message
from openai.types import CompletionUsage
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

logger = logging.getLogger(__name__)

litellm._turn_on_debug()  # pyright: ignore[reportPrivateImportUsage]


class SingleTurnRolloutProcessor(RolloutProcessor):
    """Single turn rollout processor for direct LLM calls."""

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Generate single turn rollout tasks and return them for external handling."""
        # Do not modify global LiteLLM cache. Disable caching per-request instead.

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row asynchronously."""
            start_time = time.perf_counter()

            if len(row.messages) == 0:
                raise ValueError("Messages is empty. Please provide a non-empty dataset")

            messages_payload = [message.model_dump() for message in row.messages]

            request_params = {"messages": messages_payload, **config.completion_params}
            # Ensure caching is disabled only for this request (review feedback)
            request_params["cache"] = {"no-cache": True}
            request_params["stream"] = True  # Enable streaming
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

            # _litellm = importlib.import_module("litellm")
            # acompletion = getattr(_litellm, "acompletion")

            # Handle streaming response
            assistant_content = ""
            tool_calls = None
            usage_info = None

            stream = await acompletion(**request_params)
            async for chunk in stream:  # pyright: ignore[reportGeneralTypeIssues]
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        assistant_content += delta.content
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_calls = delta.tool_calls

                # Capture usage info from the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_info = chunk.usage

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

            messages = list(row.messages) + [
                Message(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=converted_tool_calls,
                )
            ]

            if usage_info:
                row.execution_metadata.usage = CompletionUsage(
                    prompt_tokens=usage_info.prompt_tokens,
                    completion_tokens=usage_info.completion_tokens,
                    total_tokens=usage_info.total_tokens,
                )
            else:
                # Fallback if usage info not available from streaming
                row.execution_metadata.usage = CompletionUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
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
