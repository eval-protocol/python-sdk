import asyncio
import logging
import os
from typing import List

from eval_protocol.models import ChatCompletionMessageToolCall, EvaluationRow, Message
from eval_protocol.pytest.types import RolloutProcessorConfig


async def default_single_turn_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> List[EvaluationRow]:
    """Generate a single response from any supported model provider using LiteLLM."""

    # Quiet LiteLLM logs in test runs unless user overrode
    try:
        if os.environ.get("LITELLM_LOG") is None:
            os.environ["LITELLM_LOG"] = "ERROR"
        _llog = logging.getLogger("LiteLLM")
        _llog.setLevel(logging.CRITICAL)
        _llog.propagate = False
        for _h in list(_llog.handlers):
            _llog.removeHandler(_h)
    except Exception:
        pass

    # Do not modify global LiteLLM cache. Disable caching per-request instead.

    async def process_row(row: EvaluationRow) -> EvaluationRow:
        """Process a single row asynchronously."""
        if len(row.messages) == 0:
            raise ValueError("Messages is empty. Please provide a non-empty dataset")

        messages_payload = [{"role": m.role, "content": m.content} for m in row.messages]

        request_params = {"model": config.model, "messages": messages_payload, **config.input_params}
        # Ensure caching is disabled only for this request (review feedback)
        request_params["cache"] = {"no-cache": True}
        # Single-level reasoning effort: expect `reasoning_effort` only
        effort_val = None
        if isinstance(config.input_params, dict):
            if "reasoning_effort" in config.input_params:
                effort_val = str(config.input_params["reasoning_effort"])  # flat shape
            elif isinstance(config.input_params.get("extra_body"), dict) and "reasoning_effort" in config.input_params["extra_body"]:
                # Accept if user passed it directly inside extra_body
                effort_val = str(config.input_params["extra_body"]["reasoning_effort"])  # already in extra_body

        if effort_val:
            # Always under extra_body so LiteLLM forwards to provider-specific param set
            request_params.setdefault("extra_body", {})
            request_params["extra_body"]["reasoning_effort"] = effort_val
            # Ensure unsupported top-level keys are not present
            if "reasoning_effort" in request_params:
                request_params.pop("reasoning_effort", None)

        if row.tools is not None:
            request_params["tools"] = row.tools

        # Dynamic import to avoid static dependency/lint errors if LiteLLM isn't installed yet
        import importlib

        _litellm = importlib.import_module("litellm")
        acompletion = getattr(_litellm, "acompletion")
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
        config.logger.log(row)
        return row

    # Process rows with bounded concurrency if configured
    max_concurrent = getattr(config, "max_concurrent_rollouts", 8) or 8
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
        async with semaphore:
            try:
                return await process_row(r)
            except Exception as e:
                return r

    tasks = [_sem_wrapper(row) for row in rows]
    dataset = list(await asyncio.gather(*tasks))

    return dataset
