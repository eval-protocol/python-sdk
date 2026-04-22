"""Default training-aware :class:`RolloutProcessor` for Fireworks RFT.

Unlike :class:`SingleTurnRolloutProcessor`, which uses LiteLLM chat
completions and discards token-level information, this processor drives
``FireworksV1CompletionsClient`` against a Fireworks ``/v1/completions``
endpoint and surfaces the per-sample token ids / inference logprobs
needed by reinforcement-fine-tuning training (GRPO, CISPO, DAPO, GSPO).

Why this exists
---------------
RFT training loops consume token-level data per completion (prompt ids,
completion ids, inference logprobs). ``SingleTurnRolloutProcessor`` was
never meant to produce that; customers who need it have to write a
bespoke :class:`RolloutProcessor` (see the FrozenLake example in
``fw-ai/cookbook``, which is ~800 lines). This processor promotes that
bespoke pattern into an Eval Protocol default so managed Fireworks RFT
jobs can wire it up for every customer evaluator bundle without
customer code changes.

What it puts on the row
-----------------------
Besides the usual ``EvaluationRow.messages`` update (append the first
completion as an ``assistant`` turn so existing evaluators keep working),
this processor writes the following keys to
``EvaluationRow.execution_metadata.extra``:

* ``prompt_ids``        — ``list[int]`` (shared across the N completions)
* ``completion_ids``    — ``list[list[int]]`` (one per completion)
* ``inference_logprobs``— ``list[list[float]]`` aligned to completion tokens
* ``completions_text``  — ``list[str]`` (one per completion)
* ``truncated``         — ``list[bool]`` (True when ``finish_reason == 'length'``)
* ``finish_reasons``    — ``list[str]``

Shape choice
------------
Keys are ``list[list[...]]`` keyed by completion index rather than the
flattened concat convention used by
:class:`OpenEnvRolloutProcessor` (which is multi-turn and has no natural
per-completion structure). Single-turn RFT samples ``n>1`` completions
per prompt for advantage estimation, so a per-completion shape is what
the training adapter actually needs.

Ergonomics
----------
The processor reads all sampling knobs (``model``, ``temperature``,
``max_tokens``, ``n``) from ``config.completion_params``, matching
:class:`SingleTurnRolloutProcessor`. Customer evaluator bundles don't
need to reference this class — the managed RFT launcher swaps it in.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from openai.types import CompletionUsage

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

logger = logging.getLogger(__name__)


def _as_list_of_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert ``EvaluationRow.messages`` into the dict shape the client expects."""
    return [m.dump_mdoel_for_chat_completion_request() for m in messages]


def _append_extra(row: EvaluationRow, updates: dict[str, Any]) -> None:
    """Merge *updates* into ``row.execution_metadata.extra`` without clobbering."""
    current = row.execution_metadata.extra if row.execution_metadata else None
    merged: dict[str, Any] = dict(current) if current else {}
    merged.update(updates)
    row.execution_metadata.extra = merged


class FireworksTrainingRolloutProcessor(RolloutProcessor):
    """Single-turn rollout with token-level outputs attached for RFT training.

    Args:
        drop_trailing_assistant_messages: When True (default), strip trailing
            assistant messages from the input conversation before sampling
            — matches :class:`SingleTurnRolloutProcessor` behaviour.
        tokenizer_name_or_path: Override for HuggingFace tokenizer lookup.
            When not set, the model id from ``completion_params["model"]`` is
            used (via ``FireworksV1CompletionsClient``'s default behaviour).
        api_key: Override for the Fireworks API key. Defaults to the
            ``FIREWORKS_API_KEY`` env var at first use.
        base_url: Override for the Fireworks API base URL. Defaults to the
            ``FIREWORKS_BASE_URL`` env var if set, else the SDK default.
    """

    def __init__(
        self,
        *,
        drop_trailing_assistant_messages: bool = True,
        tokenizer_name_or_path: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.drop_trailing_assistant_messages = drop_trailing_assistant_messages
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self._api_key = api_key
        self._base_url = base_url
        # One client per model id per processor instance; cached lazily in setup().
        self._clients: dict[str, Any] = {}

    def setup(self) -> None:
        """Validate the Fireworks SDK / tokenizer deps up front."""
        # Defer the heavy import to setup so processor construction is cheap.
        from eval_protocol.integrations.fireworks_v1_completions_client import (  # noqa: F401
            FireworksV1CompletionsClient,
        )

    async def acleanup(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                logger.debug("FireworksV1CompletionsClient.close() failed", exc_info=True)
        self._clients.clear()

    def _client_for(self, *, model_id: str, temperature: float, max_tokens: int) -> Any:
        """Get-or-create a ``FireworksV1CompletionsClient`` for *model_id*."""
        cached = self._clients.get(model_id)
        if cached is not None:
            return cached

        from eval_protocol.integrations.fireworks_v1_completions_client import (
            FireworksV1CompletionsClient,
        )

        client = FireworksV1CompletionsClient(
            model_id=model_id,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            api_key=self._api_key or os.getenv("FIREWORKS_API_KEY"),
            base_url=self._base_url or os.getenv("FIREWORKS_BASE_URL"),
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
        )
        self._clients[model_id] = client
        return client

    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        """Return one asyncio.Task per input row."""

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            if not row.messages:
                raise ValueError("EvaluationRow.messages is empty")

            completion_params = dict(config.completion_params or {})
            row.input_metadata.completion_params = completion_params

            model_id = completion_params.get("model")
            if not model_id:
                raise ValueError("completion_params.model is required")
            temperature = float(completion_params.get("temperature", 1.0))
            max_tokens = int(completion_params.get("max_tokens", 256))
            completions_per_prompt = int(completion_params.get("n", 1))
            if completions_per_prompt < 1:
                raise ValueError(f"n must be >= 1, got {completions_per_prompt}")

            messages_for_request: list[Message] = list(row.messages)
            if self.drop_trailing_assistant_messages:
                while messages_for_request and messages_for_request[-1].role == "assistant":
                    messages_for_request.pop()

            client = self._client_for(model_id=str(model_id), temperature=temperature, max_tokens=max_tokens)

            prompt_messages = _as_list_of_dicts(messages_for_request)
            prompt_token_ids = client.build_prompt_token_ids(messages=prompt_messages, tools=row.tools)

            # Fire N parallel calls against the *same* prompt_token_ids. Each
            # call produces one completion. We sample in parallel because
            # Fireworks /v1/completions handles n=1 most reliably; requesting
            # n>1 sometimes collapses to a single choice on partial failures,
            # and we'd rather surface per-completion retry behaviour.
            async def _one_completion() -> dict[str, Any]:
                return await client.create_completion_from_prompt_ids(
                    prompt_token_ids=prompt_token_ids, tools=row.tools
                )

            results = await asyncio.gather(*[_one_completion() for _ in range(completions_per_prompt)])

            completion_ids: list[list[int]] = []
            completions_text: list[str] = []
            inference_logprobs: list[list[float]] = []
            truncated: list[bool] = []
            finish_reasons: list[str] = []

            for result in results:
                completion_ids.append(list(result.get("completion_ids") or []))
                inference_logprobs.append(list(result.get("completion_logprobs") or []))
                finish_reason = str(result.get("finish_reason") or "unknown")
                finish_reasons.append(finish_reason)
                truncated.append(finish_reason == "length")
                # Prefer the parsed assistant content if the client produced it;
                # fall back to the raw choice text.
                choice = (result.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = str(message.get("content") or "")
                completions_text.append(text)

            first_result = results[0]
            prompt_ids = list(first_result.get("prompt_ids") or prompt_token_ids)
            first_message = (first_result.get("choices") or [{}])[0].get("message") or {}
            first_tool_calls = first_message.get("tool_calls")

            # Append the first completion as the assistant turn so that
            # existing evaluators that inspect ``last_assistant_message`` keep
            # working without modification.
            row.messages = list(messages_for_request) + [
                Message(
                    role="assistant",
                    content=completions_text[0] if completions_text else "",
                    tool_calls=first_tool_calls,
                    logprobs=inference_logprobs[0] if inference_logprobs else None,
                )
            ]

            row.execution_metadata.finish_reason = finish_reasons[0] if finish_reasons else None
            row.execution_metadata.tool_call_count = len(first_tool_calls) if first_tool_calls else 0
            row.execution_metadata.usage = CompletionUsage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=sum(len(ids) for ids in completion_ids),
                total_tokens=len(prompt_ids) + sum(len(ids) for ids in completion_ids),
            )
            row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time

            _append_extra(
                row,
                {
                    "prompt_ids": prompt_ids,
                    "completion_ids": completion_ids,
                    "inference_logprobs": inference_logprobs,
                    "completions_text": completions_text,
                    "truncated": truncated,
                    "finish_reasons": finish_reasons,
                },
            )

            default_logger.log(row)
            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await process_row(r)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]
