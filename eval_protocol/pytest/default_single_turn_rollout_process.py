import asyncio
import logging
import time
from typing import AsyncIterator, List

import litellm
from litellm import acompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessageToolCall

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import RolloutProcessorConfig

logger = logging.getLogger(__name__)


async def default_single_turn_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> AsyncIterator[EvaluationRow]:
    """Generate a single response from any supported model provider using LiteLLM."""

    # Explicitly disable LiteLLM caching to avoid reused responses across runs
    try:
        litellm.cache = None
        # Some versions expose a helper; ignore if unavailable
        if hasattr(litellm, "disable_cache"):
            litellm.disable_cache()  # type: ignore[call-arg]
    except Exception:
        pass

    async def process_row(row: EvaluationRow) -> EvaluationRow:
        """Process a single row asynchronously."""
        if len(row.messages) == 0:
            raise ValueError("Messages is empty. Please provide a non-empty dataset")

        messages_payload = [{"role": m.role, "content": m.content} for m in row.messages]

        request_params = {"model": config.model, "messages": messages_payload, **config.input_params}
        # Allow passing reasoning effort to Fireworks via LiteLLM using extra_body
        # Expected: config.input_params may contain {"reasoning": {"effort": "low|medium|high"}}
        if "reasoning" in config.input_params:
            request_params.setdefault("extra_body", {})
            request_params["extra_body"]["reasoning"] = config.input_params["reasoning"]

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

        row.messages = messages
        default_logger.log(row)
        logger.info(f"FINISHED PROCESSING ROW: {row.input_metadata.row_id} at time {time.time()}")
        return row

    # Process rows with bounded concurrency and yield as they complete
    max_concurrent = getattr(config, "max_concurrent_rollouts", 8) or 8
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
        async with semaphore:
            return await process_row(r)

    # Create all tasks
    tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]

    # Yield results as they complete (not in original order)
    try:
        while tasks:
            # Wait for at least one task to complete
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Yield completed results
            for task in done:
                try:
                    result = await task
                    yield result
                except Exception as e:
                    # Log error but continue processing other tasks
                    print(f"Error processing row: {e}")
                    # Could yield an error row or skip

            # Update tasks list to only pending tasks
            tasks = list(pending)

    finally:
        # Clean up any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
