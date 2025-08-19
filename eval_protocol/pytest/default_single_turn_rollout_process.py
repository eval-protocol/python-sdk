import asyncio
import logging
import os
import time
from typing import List

from litellm import acompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessageToolCall

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

logger = logging.getLogger(__name__)


class SingleTurnRolloutProcessor(RolloutProcessor):
    """Single turn rollout processor for direct LLM calls."""

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Generate single turn rollout tasks and return them for external handling."""

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

            request_params = {"messages": messages_payload, **config.completion_params}
            # Ensure caching is disabled only for this request (review feedback)
            request_params["cache"] = {"no-cache": True}
            # Single-level reasoning effort: expect `reasoning_effort` only
            effort_val = None

            if "reasoning_effort" in config.completion_params:
                effort_val = str(config.completion_params["reasoning_effort"])  # flat shape
            elif (
                isinstance(config.completion_params.get("extra_body"), dict)
                and "reasoning_effort" in config.completion_params["extra_body"]
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

            # Dynamic import to avoid static dependency/lint errors if LiteLLM isn't installed yet
            import importlib

            _litellm = importlib.import_module("litellm")
            acompletion = getattr(_litellm, "acompletion")
            # Ensure LiteLLM forwards OpenAI logprobs params if present
            try:
                cp = request_params
                allow = set(cp.get("allowed_openai_params", []) or [])
                if "logprobs" in cp or (isinstance(cp.get("extra_body"), dict) and "logprobs" in cp["extra_body"]):
                    allow.update(["logprobs"])  # Fireworks uses integer `logprobs`
                    cp["allowed_openai_params"] = list(allow)
            except Exception:
                pass

            # Optional: try Fireworks /completions for reliable first-token logprobs
            async def _try_fireworks_completions_for_logprobs(messages_payload, params):
                try:
                    import aiohttp, os as _os
                except Exception:
                    return "", None
                model_id = params.get("model") or ""
                if not isinstance(model_id, str) or "fireworks_ai/" not in model_id:
                    return "", None
                if not bool(os.getenv("FIREWORKS_API_KEY")):
                    return "", None
                sys_txt = "\n".join([m["content"] for m in messages_payload if m["role"] == "system" and m.get("content")])
                user_txt = "\n".join([m["content"] for m in messages_payload if m["role"] == "user" and m.get("content")])
                prompt = ((sys_txt + "\n\n") if sys_txt else "") + user_txt
                payload = {
                    "model": model_id.replace("fireworks_ai/", ""),
                    "prompt": prompt,
                    "max_tokens": int(params.get("max_tokens", 1) or 1),
                    "temperature": float(params.get("temperature", 0.0) or 0.0),
                    "logprobs": 5,
                }
                api_base = params.get("base_url") or "https://api.fireworks.ai/inference/v1"
                url = f"{api_base}/completions"
                headers = {
                    "Authorization": f"Bearer {_os.getenv('FIREWORKS_API_KEY')}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=payload, headers=headers, timeout=60) as resp:
                            if resp.status != 200:
                                return "", None
                            data = await resp.json()
                    choice = (data.get("choices") or [{}])[0]
                    text = choice.get("text") or ""
                    lp = choice.get("logprobs") or {}
                    top = None
                    if isinstance(lp.get("top_logprobs"), list) and lp["top_logprobs"]:
                        top = lp["top_logprobs"][0]
                    letter_lp = {}
                    if isinstance(top, dict):
                        for k, v in top.items():
                            norm = str(k).strip()
                            if norm in {"A", "B", "C", "D"}:
                                letter_lp[norm] = float(v)
                    return text, (letter_lp or None)
                except Exception:
                    return "", None

            use_fw_completions = bool(config.completion_params.get("use_fireworks_completions_for_logprobs"))
            # Auto-enable for Fireworks chat models when possible to improve logprobs coverage
            try:
                if not use_fw_completions and isinstance(request_params.get("model"), str) and "fireworks_ai/" in request_params.get("model", ""):
                    use_fw_completions = True
            except Exception:
                pass
            fw_text, fw_letter_lp = ("", None)
            if use_fw_completions:
                fw_text, fw_letter_lp = await _try_fireworks_completions_for_logprobs(messages_payload, request_params)

            response = None
            try:
                # Increase robustness with retries
                request_params.setdefault("num_retries", 3)
                request_params.setdefault("retry_strategy", "exponential_backoff_retry")
                response = await acompletion(**request_params)
                logprobs_fallback = False
            except Exception:
                # Fallback: retry without logprobs to at least get a letter prediction
                fallback = dict(request_params)
                try:
                    fallback.pop("logprobs", None)
                    fallback.pop("top_logprobs", None)
                    if isinstance(fallback.get("extra_body"), dict):
                        fallback["extra_body"].pop("logprobs", None)
                        fallback["extra_body"].pop("top_logprobs", None)
                    if isinstance(fallback.get("allowed_openai_params"), list):
                        fallback["allowed_openai_params"] = [
                            p for p in fallback["allowed_openai_params"] if p not in ("logprobs", "top_logprobs")
                        ]
                except Exception:
                    pass
                response = await acompletion(**fallback)
                logprobs_fallback = True

            assistant_content = response.choices[0].message.content or (fw_text or "") or ""
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

            # Attempt to extract top-token logprobs for the first generated token and map to A/B/C/D
            def _extract_letter_logprobs(resp_obj) -> dict:
                letters = {"A", "B", "C", "D"}
                extracted: dict[str, float] = {}

                # Try direct attribute locations first
                choice0 = None
                try:
                    choice0 = resp_obj.choices[0]
                except Exception:
                    choice0 = None

                candidates = []
                # Common shapes: choice.logprobs or choice.message.logprobs
                if choice0 is not None:
                    lp = getattr(choice0, "logprobs", None)
                    if lp is not None:
                        candidates.append(lp)
                    msg = getattr(choice0, "message", None)
                    if msg is not None:
                        msg_lp = getattr(msg, "logprobs", None)
                        if msg_lp is not None:
                            candidates.append(msg_lp)

                # If we have Pydantic model, try model_dump to inspect raw dict
                try:
                    if hasattr(resp_obj, "model_dump"):
                        raw = resp_obj.model_dump()
                    elif hasattr(resp_obj, "dict"):
                        raw = resp_obj.dict()  # type: ignore[attr-defined]
                    else:
                        raw = None
                except Exception:
                    raw = None

                if raw:
                    try:
                        raw_lp = (
                            raw.get("choices", [{}])[0].get("logprobs")
                            or raw.get("choices", [{}])[0].get("message", {}).get("logprobs")
                        )
                        if raw_lp is not None:
                            candidates.append(raw_lp)
                    except Exception:
                        pass

                # Fallback: recursively search for token/logprob pairs (e.g., provider-specific completion_logprobs)
                def _walk(obj):
                    if isinstance(obj, dict):
                        # Direct token/logprob
                        if ("token" in obj and "logprob" in obj) or ("text" in obj and "logprob" in obj):
                            return [obj]
                        found = []
                        for v in obj.values():
                            res = _walk(v)
                            if res:
                                found.extend(res)
                        return found
                    elif isinstance(obj, list):
                        found = []
                        for it in obj:
                            res = _walk(it)
                            if res:
                                found.extend(res)
                        return found
                    return []

                if raw:
                    try:
                        recents = _walk(raw)
                        # Group into a pseudo top_logprobs set
                        if recents:
                            candidates.append({"top_logprobs": recents})
                    except Exception:
                        pass

                # Normalize known formats to list of token entries with token/logprob/top_logprobs
                token_entries = []
                for cand in candidates:
                    try:
                        # OpenAI-style: {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": [{...}]}]}
                        content_list = getattr(cand, "content", None) or (cand.get("content") if isinstance(cand, dict) else None)
                        if content_list and isinstance(content_list, list) and len(content_list) > 0:
                            token_entries.append(content_list[0])
                            continue
                        # Fireworks variants may inline top_logprobs at choice level
                        top_lp = getattr(cand, "top_logprobs", None) or (cand.get("top_logprobs") if isinstance(cand, dict) else None)
                        token = getattr(cand, "token", None) or (cand.get("token") if isinstance(cand, dict) else None)
                        logprob = getattr(cand, "logprob", None) or (cand.get("logprob") if isinstance(cand, dict) else None)
                        if top_lp or (token and logprob is not None):
                            token_entries.append({"token": token, "logprob": logprob, "top_logprobs": top_lp})
                    except Exception:
                        continue

                # Pull candidates from top_logprobs list if present
                for ent in token_entries:
                    top = None
                    try:
                        top = ent.get("top_logprobs") if isinstance(ent, dict) else getattr(ent, "top_logprobs", None)
                    except Exception:
                        top = None

                    # Sometimes the first token itself is returned + top list; include both
                    pool = []
                    if top and isinstance(top, list):
                        pool.extend(top)
                    # Include the token entry itself
                    pool.append(ent)

                    for t in pool:
                        try:
                            tk = None
                            if isinstance(t, dict):
                                tk = t.get("token") or t.get("text")
                            else:
                                tk = getattr(t, "token", None)
                            lpv = t.get("logprob") if isinstance(t, dict) else getattr(t, "logprob", None)
                            if not tk or lpv is None:
                                continue
                            # Normalize whitespace around tokens (e.g., " A")
                            norm = str(tk).strip()
                            if norm in letters:
                                # Keep max logprob if duplicates appear across pools
                                if norm not in extracted or float(lpv) > extracted[norm]:
                                    extracted[norm] = float(lpv)
                        except Exception:
                            continue

                return extracted

            letter_logprobs = fw_letter_lp or _extract_letter_logprobs(response)

            messages = list(row.messages) + [
                Message(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=converted_tool_calls,
                    # Store extracted letter logprobs in control_plane_step for downstream analysis
                    control_plane_step={
                        "letter_logprobs": letter_logprobs if letter_logprobs else None,
                        "logprobs_fallback": logprobs_fallback,
                        "used_fw_completions": use_fw_completions,
                    },
                )
            ]

            row.messages = messages
            default_logger.log(row)
            return row

        # Process rows with bounded concurrency
        max_concurrent = getattr(config, "max_concurrent_rollouts", 8) or 8
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await process_row(r)
                return result

        # Create and return tasks for external handling
        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks
