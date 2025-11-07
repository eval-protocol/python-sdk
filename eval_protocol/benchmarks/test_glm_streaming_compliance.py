"""Benchmarks for GLM streaming regressions (structured output + tool calls)."""

import json
from typing import Any

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    Message,
    MetricResult,
    ChatCompletionContentPartTextParam,
)
from eval_protocol.pytest.default_single_turn_rollout_process import (
    SingleTurnRolloutProcessor,
)
from eval_protocol.pytest.evaluation_test import evaluation_test


DEFAULT_MODEL_ID = "fireworks_ai/accounts/fireworks/models/glm-4p6"
DEFAULT_MAX_TOKENS = 1024


def _coerce_content_to_str(
    content: str | list[ChatCompletionContentPartTextParam] | None,
) -> str:
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            text_val = getattr(part, "text", None)
            if text_val:
                texts.append(text_val)
        return "".join(texts)
    if content is None:
        return ""
    return str(content)


def _safe_json_loads(payload: str) -> Any | None:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


STRUCTURED_SYSTEM_PROMPT = "You are a weather assistant. Respond with a JSON object matching the provided schema."

STRUCTURED_RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or location name",
                "enum": ["London", "New York"],
            },
            "temperature": {
                "type": "number",
                "description": "Temperature in Celsius",
            },
            "conditions": {
                "type": "string",
                "description": "Weather conditions description",
            },
        },
        "required": ["location", "temperature", "conditions"],
    },
}

STRUCTURED_OUTPUT_ROW = EvaluationRow(
    messages=[
        Message(role="system", content=STRUCTURED_SYSTEM_PROMPT),
        Message(role="user", content="What is the weather like in London?"),
    ]
)
STRUCTURED_OUTPUT_ROW.input_metadata.dataset_info = {
    "case": "glm-structured-output-streaming",
}


TOOL_SYSTEM_PROMPT = (
    "You are a weather assistant. If tools are available, always call them to gather data before responding."
)

WEATHER_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "additionalProperties": False,
                "required": ["location", "unit"],
            },
        },
    }
]

TOOL_CALL_ROW = EvaluationRow(
    messages=[
        Message(role="system", content=TOOL_SYSTEM_PROMPT),
        Message(role="user", content="What is the weather like in Boston in fahrenheit?"),
    ],
    tools=WEATHER_TOOL_DEFINITION,
)
TOOL_CALL_ROW.input_metadata.dataset_info = {
    "case": "glm-tool-call-streaming",
}


@evaluation_test(
    input_rows=[[STRUCTURED_OUTPUT_ROW]],
    completion_params=[
        {
            "model": DEFAULT_MODEL_ID,
            "stream": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "response_format": STRUCTURED_RESPONSE_FORMAT,
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=1.0,
    num_runs=1,
    mode="pointwise",
)
def test_glm_streaming_structured_output(row: EvaluationRow) -> EvaluationRow:
    """Ensure structured output arrives in assistant content when streaming."""

    assistant_msg = row.last_assistant_message()
    if assistant_msg is None:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason="No assistant message produced",
            metrics={},
        )
        return row

    content_str = _coerce_content_to_str(assistant_msg.content)
    reasoning_str = assistant_msg.reasoning_content or ""
    parsed_content = _safe_json_loads(content_str)
    parsed_reasoning = _safe_json_loads(reasoning_str) if reasoning_str else None

    required_fields = {"location", "temperature", "conditions"}

    content_is_json = parsed_content is not None
    required_keys_present = content_is_json and required_fields <= set(parsed_content.keys())
    temperature_is_number = content_is_json and isinstance(parsed_content.get("temperature"), (int, float))
    location_valid = content_is_json and parsed_content.get("location") in {"London", "New York"}
    reasoning_contains_payload = parsed_reasoning is not None
    finish_reason = row.execution_metadata.finish_reason
    finish_reason_expected = finish_reason == "stop"

    all_checks_passed = (
        content_str.strip()
        and content_is_json
        and required_keys_present
        and temperature_is_number
        and location_valid
        and not reasoning_contains_payload
        and finish_reason_expected
    )

    metrics = {
        "content_is_json": MetricResult(
            score=1.0 if content_is_json else 0.0,
            is_score_valid=True,
            reason="Assistant content parsed as JSON" if content_is_json else "Failed to parse JSON",
            data={"content": content_str},
        ),
        "required_keys_present": MetricResult(
            score=1.0 if required_keys_present else 0.0,
            is_score_valid=content_is_json,
            reason=("All required keys present" if required_keys_present else "Missing required keys"),
            data={"parsed_content": parsed_content},
        ),
        "temperature_is_number": MetricResult(
            score=1.0 if temperature_is_number else 0.0,
            is_score_valid=content_is_json,
            reason="Temperature is numeric" if temperature_is_number else "Temperature not numeric",
            data={"temperature": parsed_content.get("temperature") if parsed_content else None},
        ),
        "reasoning_contains_payload": MetricResult(
            score=0.0 if reasoning_contains_payload else 1.0,
            is_score_valid=True,
            reason="Reasoning is empty" if not reasoning_contains_payload else "Payload leaked to reasoning",
            data={"reasoning": reasoning_str},
        ),
        "finish_reason_stop": MetricResult(
            score=1.0 if finish_reason_expected else 0.0,
            is_score_valid=True,
            reason=(
                "finish_reason is stop" if finish_reason_expected else f"Unexpected finish_reason: {finish_reason}"
            ),
            data={"finish_reason": finish_reason},
        ),
    }

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=(
            "Structured output returned in assistant content"
            if all_checks_passed
            else "Structured output missing or malformed"
        ),
        metrics=metrics,
    )
    return row


@evaluation_test(
    input_rows=[[TOOL_CALL_ROW]],
    completion_params=[
        {
            "model": DEFAULT_MODEL_ID,
            "stream": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_glm_streaming_tool_call(row: EvaluationRow) -> EvaluationRow:
    """Ensure streaming tool calls settle with finish_reason=tool_calls and a single call."""

    assistant_msg = row.last_assistant_message()
    if assistant_msg is None:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason="No assistant message produced",
            metrics={},
        )
        return row

    tool_calls = assistant_msg.tool_calls or []
    tool_calls_for_metrics: list[Any] = []
    for tc in tool_calls:
        if hasattr(tc, "model_dump"):
            try:
                tool_calls_for_metrics.append(tc.model_dump(exclude_none=True))
            except Exception:
                tool_calls_for_metrics.append(str(tc))
        elif isinstance(tc, dict):
            tool_calls_for_metrics.append(tc)
        else:
            tool_calls_for_metrics.append(str(tc))
    finish_reason = row.execution_metadata.finish_reason
    tool_call_count = row.execution_metadata.tool_call_count

    has_tool_call = len(tool_calls) > 0
    exactly_one_tool_call = len(tool_calls) == 1
    finish_reason_tool_calls = finish_reason == "tool_calls"
    tool_call_count_matches = tool_call_count == len(tool_calls)

    tool_call_arguments_valid = False
    if exactly_one_tool_call:
        tool_call = tool_calls[0]
        function_block = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        arguments_payload = function_block.get("arguments")
        parsed_arguments = _safe_json_loads(arguments_payload) if isinstance(arguments_payload, str) else None
        tool_call_arguments_valid = (
            isinstance(parsed_arguments, dict)
            and parsed_arguments.get("location", "").lower() == "boston"
            and parsed_arguments.get("unit") == "fahrenheit"
        )
    else:
        parsed_arguments = None

    all_checks_passed = (
        has_tool_call
        and exactly_one_tool_call
        and finish_reason_tool_calls
        and tool_call_arguments_valid
        and tool_call_count_matches
    )

    metrics = {
        "has_tool_call": MetricResult(
            score=1.0 if has_tool_call else 0.0,
            is_score_valid=True,
            reason="Assistant produced at least one tool call" if has_tool_call else "No tool calls returned",
            data={"tool_call_count": len(tool_calls)},
        ),
        "single_tool_call": MetricResult(
            score=1.0 if exactly_one_tool_call else 0.0,
            is_score_valid=has_tool_call,
            reason=("Exactly one tool call" if exactly_one_tool_call else "Unexpected number of tool calls"),
            data={"tool_calls": tool_calls_for_metrics},
        ),
        "finish_reason_tool_calls": MetricResult(
            score=1.0 if finish_reason_tool_calls else 0.0,
            is_score_valid=True,
            reason=(
                "finish_reason is tool_calls"
                if finish_reason_tool_calls
                else f"Unexpected finish_reason: {finish_reason}"
            ),
            data={"finish_reason": finish_reason},
        ),
        "tool_call_arguments_valid": MetricResult(
            score=1.0 if tool_call_arguments_valid else 0.0,
            is_score_valid=exactly_one_tool_call,
            reason=("Tool call arguments valid" if tool_call_arguments_valid else "Tool call arguments invalid"),
            data={"arguments": parsed_arguments},
        ),
        "tool_call_count_matches": MetricResult(
            score=1.0 if tool_call_count_matches else 0.0,
            is_score_valid=True,
            reason=(
                "tool_call_count matches returned calls"
                if tool_call_count_matches
                else f"tool_call_count mismatch (metadata={tool_call_count}, actual={len(tool_calls)})"
            ),
            data={"metadata_tool_call_count": tool_call_count},
        ),
    }

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=(
            "Streaming tool call completed correctly" if all_checks_passed else "Streaming tool call behaviour invalid"
        ),
        metrics=metrics,
    )
    return row
