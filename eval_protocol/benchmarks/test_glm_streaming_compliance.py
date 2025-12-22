"""Benchmarks for output streaming compliance (structured output + tool calls)."""

import json
import os
import re
from typing import Any

import pytest

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    Message,
    MetricResult,
)
from eval_protocol.pytest.default_single_turn_rollout_process import (
    SingleTurnRolloutProcessor,
)
from eval_protocol.pytest.evaluation_test import evaluation_test


DEFAULT_MODEL_ID = "fireworks_ai/accounts/fireworks/models/glm-4p6"
DEFAULT_MAX_TOKENS = 10000

# Feature flags from environment variables
# EP_SUPPORTS_MULTIPLE_TOOL_CALLS: "1" to include multiple tool call tests, "0" to skip
SUPPORTS_MULTIPLE_TOOL_CALLS = os.getenv("EP_SUPPORTS_MULTIPLE_TOOL_CALLS", "1") == "1"
# EP_SUPPORTS_REASONING: "1" to include reasoning tests and pass reasoning_effort, "0" to skip reasoning tests
SUPPORTS_REASONING = os.getenv("EP_SUPPORTS_REASONING", "1") == "1"


def _coerce_content_to_str(
    content: str | list[Any] | None,
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

PEER_SIMPLE_STREAM_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "Write a short explanation of how binary search works. Keep it under 200 words.",
        }
    ],
    "temperature": 0.3,
    "top_p": 1,
    "max_tokens": 400,
}

PEER_JSON_STREAM_PAYLOAD = {
    "messages": [{"role": "user", "content": "What is the weather like in London?"}],
    "max_tokens": 25344,
    "temperature": 1,
    "top_p": 1,
    "response_format": {
        "type": "json_object",
        "schema": STRUCTURED_RESPONSE_FORMAT["schema"],
    },
}

PEER_TOOL_BRACE_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "Call test_brace_bug with param1='test_value', param2=42, and param3=true",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "test_brace_bug",
                "description": "A test function to validate JSON brace handling in tool arguments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "A string parameter",
                        },
                        "param2": {
                            "type": "integer",
                            "description": "An integer parameter",
                        },
                        "param3": {
                            "type": "boolean",
                            "description": "A boolean parameter",
                        },
                    },
                    "required": ["param1", "param2", "param3"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "temperature": 0.1,
    "top_p": 1,
}

PEER_TOOL_MULTI_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "Process data {'users': [{'name': 'John', 'age': 30}], 'total': 1} with count 5 and enabled true",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "process_data",
                "description": "Process complex data structures",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "filters": {"type": "array"},
                        "count": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                    },
                    "required": ["data"],
                },
            },
        }
    ],
    "stream": True,
    "temperature": 0.1,
}

MULTI_TOOL_CALLS_PAYLOAD = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. When multiple tools are needed, call them all in one response.",
        },
        {
            "role": "user",
            "content": (
                "What's the weather in Boston and San Francisco (in Fahrenheit)? Also check the air quality in Boston."
            ),
        },
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
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
                    "required": ["location", "unit"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_air_quality",
                "description": "Get the current air quality in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. Boston, MA",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        },
    ],
    "tool_choice": "required",
    "temperature": 0.2,
    "top_p": 1,
    "stream": True,
}

PEER_TOOL_MISSING_REQUIRED_PARAM_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "View the file at /tmp/test.txt with view_range [160, 210]",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "view",
                "description": "View a file or directory",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file or directory to view",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["file", "directory"],
                            "description": "Type of the path (file or directory)",
                        },
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Line range to view [start, end]",
                        },
                    },
                    "required": ["path", "type"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "required",
    "temperature": 0.1,
    "stream": True,
}

PEER_TOOL_STRING_INSTEAD_ARRAY_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "View the file at /tmp/test.txt with view_range [160, 210]",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "view",
                "description": "View a file or directory",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file or directory to view",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["file", "directory"],
                            "description": "Type of the path (file or directory)",
                        },
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Line range to view [start, end]",
                        },
                    },
                    "required": ["path", "type", "view_range"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "required",
    "temperature": 0.1,
    "stream": True,
}

PEER_TOOL_NAMING_ERROR_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "View the file at /tmp/test.txt",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "view",
                "description": "View a file or directory",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file or directory to view",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["file", "directory"],
                            "description": "Type of the path (file or directory)",
                        },
                    },
                    "required": ["path", "type"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "required",
    "temperature": 0.1,
    "stream": True,
}

PEER_TOOL_COMMAND_EXECUTION_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "Launch a process to run 'ls -la' in the /tmp directory",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "launch_process",
                "description": "Launch a shell process",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory for the command",
                        },
                        "env": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Environment variables",
                        },
                    },
                    "required": ["command", "cwd"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "required",
    "temperature": 0.1,
    "stream": True,
}

PEER_TOOL_PARAMETER_FORMAT_ERRORS_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "Process data with count 60, enabled true, and timeout 30",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "process_data",
                "description": "Process data with various parameters",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "Number of items to process",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Whether processing is enabled",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds",
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "retry": {"type": "boolean"},
                                "max_attempts": {"type": "integer"},
                            },
                            "description": "Processing options",
                        },
                    },
                    "required": ["count", "enabled"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "required",
    "temperature": 0.1,
    "stream": True,
}


def _build_row_from_payload(case: str, payload: dict[str, Any]) -> EvaluationRow:
    messages = [
        Message(role=message["role"], content=message.get("content", "")) for message in payload.get("messages", [])
    ]
    row = EvaluationRow(messages=messages, tools=payload.get("tools"))
    row.input_metadata.dataset_info = {"case": case}
    return row


def _build_completion_params_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": DEFAULT_MODEL_ID,
        "stream": True,
        "return_reasoning_with_separate_field": True,
        "raw_output": True,  # Include raw model output for debugging
    }
    # Only include reasoning_effort if model supports it
    if SUPPORTS_REASONING:
        params["reasoning_effort"] = "none"  # Default: no reasoning unless explicitly requested

    passthrough_keys = {"temperature", "top_p", "max_tokens", "response_format"}
    # Only passthrough reasoning_effort if model supports it
    if SUPPORTS_REASONING:
        passthrough_keys.add("reasoning_effort")

    for key in passthrough_keys:
        if key in payload:
            params[key] = payload[key]
    return params


def _maybe_add_reasoning_effort(params: dict[str, Any], effort: str = "low") -> dict[str, Any]:
    """Conditionally add reasoning_effort to params if model supports it."""
    if SUPPORTS_REASONING:
        params["reasoning_effort"] = effort
    return params


def _normalize_tool_call(tc: Any) -> tuple[str | None, dict[str, Any] | None]:
    """Convert LiteLLM tool call objects/dicts into (name, arguments dict)."""

    record: dict[str, Any]
    if hasattr(tc, "model_dump"):
        try:
            record = tc.model_dump(exclude_none=True)
        except Exception:
            return (None, None)
    elif isinstance(tc, dict):
        record = tc
    else:
        return (None, None)

    fn = record.get("function") or {}
    name = fn.get("name")
    args_raw = fn.get("arguments")
    if isinstance(args_raw, str):
        args = _safe_json_loads(args_raw)
    else:
        args = args_raw if isinstance(args_raw, dict) else None
    return (name, args if isinstance(args, dict) else None)


def _collect_tool_calls(tool_calls: list[Any] | None) -> list[tuple[str | None, dict[str, Any] | None]]:
    return [_normalize_tool_call(tc) for tc in (tool_calls or [])]


XML_TAG_PATTERN = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
FORBIDDEN_TAG_KEYWORDS = ("think", "tool_call", "tool_calls", "tool_call_section")
DEBUG_RESPONSES = os.getenv("EP_DEBUG_LOG_RESPONSES") == "1"


def _unique_preserving(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        if item not in seen:
            seen[item] = None
    return list(seen.keys())


def _scan_xml_tags(text: str) -> list[str]:
    if not text:
        return []
    return _unique_preserving([match.group(0) for match in XML_TAG_PATTERN.finditer(text)])


def _scan_forbidden_tags(content: str, reasoning: str) -> tuple[list[str], list[str]]:
    combined = "\n".join(part for part in (content, reasoning) if part)
    xml_tags = _scan_xml_tags(combined)
    forbidden = [tag for tag in xml_tags if any(keyword in tag.lower() for keyword in FORBIDDEN_TAG_KEYWORDS)]
    return forbidden, xml_tags


def _detect_reasoning_leakage(content: str, reasoning: str) -> list[str]:
    """
    Detect thinking/reasoning phrases in content when reasoning_content exists.

    Returns list of detected thinking patterns that should be in reasoning, not content.
    Only checks for very clear reasoning indicators, not common phrases.
    """
    if not reasoning:
        # No reasoning present, so no leakage possible
        return []

    if not content:
        return []

    # Only check for VERY clear reasoning/thinking indicators
    # Avoid common phrases that might appear in normal responses
    thinking_patterns = [
        r"<think>",  # XML thinking tags
        r"</think>",
        r"\bStep \d+:",  # "Step 1:", "Step 2:", etc. (clear reasoning structure)
        r"\bThinking:",  # Explicit thinking label
        r"\bReasoning:",  # Explicit reasoning label
        r"\bMy thought process:",  # Explicit meta-reasoning
        r"\bLet me think",  # Explicit thinking phrase (not just "Let me")
        r"\bI need to think",  # Explicit thinking
    ]

    detected = []
    for pattern in thinking_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            detected.append(pattern)

    return detected


def _augment_metrics_with_common_checks(
    metrics: dict[str, MetricResult],
    finish_reason: Any,
    content: str,
    reasoning: str,
) -> tuple[bool, bool, bool, bool]:
    finish_reason_str = ""
    if finish_reason is not None:
        finish_reason_str = str(finish_reason).strip()
    finish_reason_present = finish_reason_str != ""

    forbidden_tags, xml_tags = _scan_forbidden_tags(content, reasoning)
    no_forbidden_tags = not forbidden_tags
    no_xml_tags = not xml_tags

    # Check for reasoning leakage into content
    reasoning_leakage = _detect_reasoning_leakage(content, reasoning)
    no_reasoning_leakage = not reasoning_leakage

    metrics["finish_reason_not_null"] = MetricResult(
        score=1.0 if finish_reason_present else 0.0,
        is_score_valid=True,
        reason="finish_reason present" if finish_reason_present else "finish_reason missing or empty",
        data={"finish_reason": finish_reason},
    )
    metrics["no_forbidden_tags"] = MetricResult(
        score=1.0 if no_forbidden_tags else 0.0,
        is_score_valid=True,
        reason="No forbidden tags detected" if no_forbidden_tags else "Forbidden tags detected",
        data={"matches": forbidden_tags, "count": len(forbidden_tags)},
    )
    metrics["no_xml_tags"] = MetricResult(
        score=1.0 if no_xml_tags else 0.0,
        is_score_valid=True,
        reason="No XML-like tags detected" if no_xml_tags else "XML-like tags detected",
        data={"matches": xml_tags, "count": len(xml_tags)},
    )
    metrics["no_reasoning_leakage"] = MetricResult(
        score=1.0 if no_reasoning_leakage else 0.0,
        is_score_valid=True,  # Always valid - if no reasoning, then no leakage possible (score=1.0)
        reason="No thinking phrases in content"
        if no_reasoning_leakage
        else f"Thinking phrases detected in content: {reasoning_leakage}",
        data={"detected_patterns": reasoning_leakage, "has_reasoning": bool(reasoning)},
    )

    return finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage


def _debug_log_assistant_message(test_name: str, assistant_message: Message | None, finish_reason: Any) -> None:
    if not DEBUG_RESPONSES:
        return
    print(f"[EP][DEBUG] test={test_name} finish_reason={finish_reason!r}")
    if assistant_message is None:
        print("  assistant_message: <none>")
        return
    content = _coerce_content_to_str(assistant_message.content)
    reasoning = (assistant_message.reasoning_content or "").strip() if assistant_message.reasoning_content else ""
    tool_calls = assistant_message.tool_calls or []
    print(f"  content: {content[:400]!r}")
    print(f"  reasoning: {reasoning[:400]!r}")
    if tool_calls:
        try:
            serialized = []
            for tc in tool_calls:
                if hasattr(tc, "model_dump"):
                    serialized.append(tc.model_dump(exclude_none=True))
                elif isinstance(tc, dict):
                    serialized.append(tc)
                else:
                    serialized.append(str(tc))
            print(f"  tool_calls: {serialized}")
        except Exception as exc:  # pragma: no cover - debug helper
            print(f"  tool_calls: <error serializing> {exc!r}")
    else:
        print("  tool_calls: []")


@evaluation_test(
    input_rows=[[STRUCTURED_OUTPUT_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "stream": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "response_format": STRUCTURED_RESPONSE_FORMAT,
                "raw_output": True,  # Include raw model output for debugging
            },
            "none",  # No reasoning expected for structured output
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=1.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_structured_output(row: EvaluationRow) -> EvaluationRow:
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
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("structured_output", assistant_msg, finish_reason)

    content_str = _coerce_content_to_str(assistant_msg.content)
    reasoning_str = assistant_msg.reasoning_content or ""
    parsed_content = _safe_json_loads(content_str)
    parsed_reasoning = _safe_json_loads(reasoning_str) if reasoning_str else None

    required_fields = {"location", "temperature", "conditions"}

    content_present = bool(content_str.strip())
    content_is_json = parsed_content is not None
    required_keys_present = content_is_json and required_fields <= set(parsed_content.keys())
    temperature_is_number = content_is_json and isinstance(parsed_content.get("temperature"), (int, float))
    location_valid = content_is_json and parsed_content.get("location") in {"London", "New York"}
    reasoning_contains_payload = parsed_reasoning is not None
    finish_reason_expected = finish_reason == "stop"

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

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        content_present
        and content_is_json
        and required_keys_present
        and temperature_is_number
        and location_valid
        and not reasoning_contains_payload
        and finish_reason_expected
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

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


_SIMPLE_STREAM_ROW = _build_row_from_payload("peer-simple-stream", PEER_SIMPLE_STREAM_PAYLOAD)


@evaluation_test(
    input_rows=[[_SIMPLE_STREAM_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_SIMPLE_STREAM_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_simple_completion(row: EvaluationRow) -> EvaluationRow:
    """Ensure plain streaming completion returns content without leaking reasoning."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("simple_completion", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""

    has_content = bool(content_str.strip())
    finish_reason_stop = finish_reason == "stop"
    reasoning_empty = reasoning_str == ""
    has_tool_calls = bool(assistant_msg and assistant_msg.tool_calls)

    metrics = {
        "has_content": MetricResult(
            score=1.0 if has_content else 0.0,
            is_score_valid=True,
            reason="Assistant content present" if has_content else "Assistant content empty",
            data={"preview": content_str[:120]},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected finish_reason: {finish_reason}",
        ),
        "reasoning_empty": MetricResult(
            score=1.0 if reasoning_empty else 0.0,
            is_score_valid=True,
            reason="Reasoning is empty" if reasoning_empty else "Unexpected reasoning output",
            data={"reasoning": reasoning_str},
        ),
        "tool_calls_absent": MetricResult(
            score=1.0 if not has_tool_calls else 0.0,
            is_score_valid=True,
            reason="No tool calls emitted" if not has_tool_calls else "Unexpected tool calls emitted",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        has_content
        and finish_reason_stop
        and reasoning_empty
        and not has_tool_calls
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Simple streaming completion compliant"
        if all_checks_passed
        else "Simple completion failed compliance checks",
        metrics=metrics,
    )
    return row


_PEER_JSON_ROW = _build_row_from_payload("peer-json-stream", PEER_JSON_STREAM_PAYLOAD)


@evaluation_test(
    input_rows=[[_PEER_JSON_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_JSON_STREAM_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_json_preservation(row: EvaluationRow) -> EvaluationRow:
    """Validate peer JSON streaming payload keeps structure in assistant content."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("peer_json", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    parsed_content = _safe_json_loads(content_str)
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""

    content_is_json = parsed_content is not None
    includes_required = content_is_json and set(parsed_content.keys()) >= {"location", "temperature", "conditions"}
    reasoning_empty = reasoning_str == ""
    finish_reason_stop = finish_reason == "stop"

    metrics = {
        "content_is_json": MetricResult(
            score=1.0 if content_is_json else 0.0,
            is_score_valid=True,
            reason="Assistant content parsed as JSON" if content_is_json else "Failed to parse JSON",
            data={"content": content_str},
        ),
        "required_fields": MetricResult(
            score=1.0 if includes_required else 0.0,
            is_score_valid=content_is_json,
            reason="All required fields present" if includes_required else "Missing required fields",
            data={"parsed_content": parsed_content},
        ),
        "reasoning_empty": MetricResult(
            score=1.0 if reasoning_empty else 0.0,
            is_score_valid=True,
            reason="Reasoning is empty" if reasoning_empty else "Unexpected reasoning output",
            data={"reasoning": reasoning_str},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        content_is_json
        and includes_required
        and reasoning_empty
        and finish_reason_stop
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="JSON structure preserved" if all_checks_passed else "JSON structure missing or response malformed",
        metrics=metrics,
    )
    return row


@evaluation_test(
    input_rows=[[TOOL_CALL_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "stream": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "raw_output": True,  # Include raw model output for debugging
            },
            "none",  # No reasoning expected for tool calls
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_single_tool_call(row: EvaluationRow) -> EvaluationRow:
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

    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_call", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content)
    reasoning_str = (assistant_msg.reasoning_content or "").strip()
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
    tool_call_count = row.execution_metadata.tool_call_count

    has_tool_call = len(tool_calls) > 0
    exactly_one_tool_call = len(tool_calls) == 1
    finish_reason_tool_calls = finish_reason == "tool_calls"
    tool_call_count_matches = tool_call_count == len(tool_calls)

    tool_call_arguments_valid = False
    parsed_arguments = None
    if exactly_one_tool_call:
        # Use helper to normalize tool call (handles both dict and Pydantic objects)
        name, parsed_arguments = _normalize_tool_call(tool_calls[0])
        tool_call_arguments_valid = (
            isinstance(parsed_arguments, dict)
            and ("boston" in (parsed_arguments.get("location") or "").lower())
            and parsed_arguments.get("unit") == "fahrenheit"
        )

    base_checks_passed = (
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

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        base_checks_passed and finish_reason_present and no_forbidden_tags and no_xml_tags and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Streaming tool call completed correctly"
        if all_checks_passed
        else "Streaming tool call behaviour invalid",
        metrics=metrics,
    )
    return row


_PEER_TOOL_BRACE_ROW = _build_row_from_payload("peer-tool-brace-bug", PEER_TOOL_BRACE_PAYLOAD)


@evaluation_test(
    input_rows=[[_PEER_TOOL_BRACE_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_TOOL_BRACE_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_tool_json_validity(row: EvaluationRow) -> EvaluationRow:
    """Ensure streamed tool arguments remain valid JSON (no truncated braces)."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_brace_arguments", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    tool_calls = assistant_msg.tool_calls or [] if assistant_msg else []

    # Use helper to normalize tool calls (handles both dict and Pydantic objects)
    normalized_calls = _collect_tool_calls(tool_calls)
    parsed_arguments: list[Any] = [args for name, args in normalized_calls]
    valid_arguments = all(isinstance(args, dict) for args in parsed_arguments)

    finish_reason_ok = finish_reason in {"tool_calls", "stop"}

    metrics = {
        "tool_call_present": MetricResult(
            score=1.0 if tool_calls else 0.0,
            is_score_valid=True,
            reason="Tool calls emitted" if tool_calls else "No tool calls emitted",
            data={"tool_calls": tool_calls},
        ),
        "arguments_json": MetricResult(
            score=1.0 if (tool_calls and valid_arguments) else 0.0,
            is_score_valid=bool(tool_calls),
            reason="Arguments parsed as JSON" if valid_arguments else "Arguments not valid JSON",
            data={"parsed_arguments": parsed_arguments},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason_ok else 0.0,
            is_score_valid=True,
            reason="finish_reason acceptable" if finish_reason_ok else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        tool_calls
        and valid_arguments
        and finish_reason_ok
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Tool arguments preserved as JSON" if all_checks_passed else "Tool argument JSON malformed",
        metrics=metrics,
    )
    return row


_PEER_TOOL_MULTI_ROW = _build_row_from_payload("peer-tool-multi", PEER_TOOL_MULTI_PAYLOAD)


@evaluation_test(
    input_rows=[[_PEER_TOOL_MULTI_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_TOOL_MULTI_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_tool_complex_arguments(row: EvaluationRow) -> EvaluationRow:
    """Validate complex tool arguments are preserved when streaming."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_multi", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    valid_call = None
    for name, args in calls:
        if name == "process_data" and isinstance(args, dict):
            valid_call = args
            break

    finish_reason_tool_calls = finish_reason == "tool_calls"

    metrics = {
        "tool_calls_count": MetricResult(
            score=1.0 if calls else 0.0,
            is_score_valid=True,
            reason="Tool calls emitted" if calls else "No tool calls emitted",
            data={"tool_call_count": len(calls)},
        ),
        "process_data_arguments_valid": MetricResult(
            score=1.0 if isinstance(valid_call, dict) else 0.0,
            is_score_valid=bool(calls),
            reason="process_data arguments parsed" if valid_call else "process_data arguments missing/invalid",
            data={"arguments": valid_call},
        ),
        "finish_reason_tool_calls": MetricResult(
            score=1.0 if finish_reason_tool_calls else 0.0,
            is_score_valid=True,
            reason="finish_reason is tool_calls"
            if finish_reason_tool_calls
            else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        valid_call
        and finish_reason_tool_calls
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="process_data call preserved" if all_checks_passed else "process_data call missing or invalid",
        metrics=metrics,
    )
    return row


_MULTI_TOOL_CALLS_ROW = _build_row_from_payload("multi-tool-calls", MULTI_TOOL_CALLS_PAYLOAD)


@pytest.mark.skipif(
    not SUPPORTS_MULTIPLE_TOOL_CALLS,
    reason="Model does not support multiple tool calls (EP_SUPPORTS_MULTIPLE_TOOL_CALLS=0)",
)
@evaluation_test(
    input_rows=[[_MULTI_TOOL_CALLS_ROW]],
    completion_params=[_build_completion_params_from_payload(MULTI_TOOL_CALLS_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_multiple_tool_calls(row: EvaluationRow) -> EvaluationRow:
    """Ensure multiple tool calls survive a single streamed assistant turn."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("kimi_multi_tool_calls", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    names = [name for name, _ in calls if name]
    has_weather = names.count("get_current_weather") >= 2
    has_air_quality = "get_air_quality" in names

    metrics = {
        "tool_calls_count": MetricResult(
            score=1.0 if len(calls) >= 3 else 0.0,
            is_score_valid=True,
            reason="Three or more tool calls" if len(calls) >= 3 else "Fewer than three tool calls",
            data={"tool_calls": names},
        ),
        "includes_weather_calls": MetricResult(
            score=1.0 if has_weather else 0.0,
            is_score_valid=True,
            reason="Weather tool called for multiple cities" if has_weather else "Insufficient weather tool calls",
        ),
        "includes_air_quality": MetricResult(
            score=1.0 if has_air_quality else 0.0,
            is_score_valid=True,
            reason="Air quality tool called" if has_air_quality else "Air quality tool missing",
        ),
        "finish_reason_tool_calls": MetricResult(
            score=1.0 if finish_reason == "tool_calls" else 0.0,
            is_score_valid=True,
            reason=(
                "finish_reason is tool_calls"
                if finish_reason == "tool_calls"
                else f"Unexpected finish_reason: {finish_reason}"
            ),
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        len(calls) >= 3
        and has_weather
        and has_air_quality
        and finish_reason == "tool_calls"
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Multiple tool calls emitted"
        if all_checks_passed
        else "Expected multi-tool response missing or invalid",
        metrics=metrics,
    )
    return row


_PEER_TOOL_REQUIRED_PARAMS_ROW = _build_row_from_payload(
    "peer-tool-required-params", PEER_TOOL_MISSING_REQUIRED_PARAM_PAYLOAD
)


@evaluation_test(
    input_rows=[[_PEER_TOOL_REQUIRED_PARAMS_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_TOOL_MISSING_REQUIRED_PARAM_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_tool_required_params_present(row: EvaluationRow) -> EvaluationRow:
    """Verify that tool calls include all required parameters during streaming."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_required_params", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    required_params_present = False
    arguments = None
    for _, args in calls:
        if args:
            arguments = args
            # Check that required 'type' param is present and valid
            required_params_present = "type" in args and args.get("type") in {"file", "directory"}

    metrics = {
        "tool_call_emitted": MetricResult(
            score=1.0 if calls else 0.0,
            is_score_valid=True,
            reason="Tool call emitted" if calls else "No tool call emitted",
        ),
        "required_params_present": MetricResult(
            score=1.0 if required_params_present else 0.0,
            is_score_valid=bool(calls),
            reason="All required parameters present"
            if required_params_present
            else "Required parameter missing or invalid",
            data={"arguments": arguments},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason in {"tool_calls", "stop"} else 0.0,
            is_score_valid=True,
            reason="finish_reason acceptable"
            if finish_reason in {"tool_calls", "stop"}
            else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        required_params_present
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="All required parameters included in tool call"
        if all_checks_passed
        else "Required parameters missing or response invalid",
        metrics=metrics,
    )
    return row


_PEER_TOOL_STRING_ARRAY_ROW = _build_row_from_payload(
    "peer-tool-string-instead-of-array", PEER_TOOL_STRING_INSTEAD_ARRAY_PAYLOAD
)


@evaluation_test(
    input_rows=[[_PEER_TOOL_STRING_ARRAY_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_TOOL_STRING_INSTEAD_ARRAY_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_tool_array_parameters(row: EvaluationRow) -> EvaluationRow:
    """Check streamed arguments keep view_range as an integer array."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_view_range_format", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    view_range_valid = False
    view_range_value = None
    for _, args in calls:
        if args and "view_range" in args:
            view_range_value = args["view_range"]
            if (
                isinstance(view_range_value, list)
                and len(view_range_value) == 2
                and all(isinstance(item, int) for item in view_range_value)
            ):
                view_range_valid = True

    metrics = {
        "tool_call_emitted": MetricResult(
            score=1.0 if calls else 0.0,
            is_score_valid=True,
            reason="Tool call emitted" if calls else "No tool call emitted",
        ),
        "view_range_valid": MetricResult(
            score=1.0 if view_range_valid else 0.0,
            is_score_valid=bool(calls),
            reason="view_range is integer array" if view_range_valid else "view_range malformed",
            data={"view_range": view_range_value},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason in {"tool_calls", "stop"} else 0.0,
            is_score_valid=True,
            reason="finish_reason acceptable"
            if finish_reason in {"tool_calls", "stop"}
            else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        view_range_valid and finish_reason_present and no_forbidden_tags and no_xml_tags and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="view_range array preserved" if all_checks_passed else "view_range malformed",
        metrics=metrics,
    )
    return row


_PEER_TOOL_NAMING_ROW = _build_row_from_payload("peer-tool-naming-error", PEER_TOOL_NAMING_ERROR_PAYLOAD)


@evaluation_test(
    input_rows=[[_PEER_TOOL_NAMING_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_TOOL_NAMING_ERROR_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_tool_naming_fields(row: EvaluationRow) -> EvaluationRow:
    """Confirm tool arguments include required naming fields."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_naming", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    args_valid = False
    arguments = None
    for _, args in calls:
        if args:
            arguments = args
            args_valid = "type" in args and args.get("type") in {"file", "directory"}

    metrics = {
        "tool_call_emitted": MetricResult(
            score=1.0 if calls else 0.0,
            is_score_valid=True,
            reason="Tool call emitted" if calls else "No tool call emitted",
        ),
        "naming_valid": MetricResult(
            score=1.0 if args_valid else 0.0,
            is_score_valid=bool(calls),
            reason="Tool arguments include type" if args_valid else "Tool arguments missing type",
            data={"arguments": arguments},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason in {"tool_calls", "stop"} else 0.0,
            is_score_valid=True,
            reason="finish_reason acceptable"
            if finish_reason in {"tool_calls", "stop"}
            else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        args_valid and finish_reason_present and no_forbidden_tags and no_xml_tags and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Tool naming fields intact" if all_checks_passed else "Tool naming fields missing or response invalid",
        metrics=metrics,
    )
    return row


_PEER_TOOL_COMMAND_ROW = _build_row_from_payload("peer-tool-command-execution", PEER_TOOL_COMMAND_EXECUTION_PAYLOAD)


@evaluation_test(
    input_rows=[[_PEER_TOOL_COMMAND_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_TOOL_COMMAND_EXECUTION_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_tool_command_execution(row: EvaluationRow) -> EvaluationRow:
    """Validate command execution tool receives the correct parameters."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_command", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    command_valid = False
    arguments = None
    for name, args in calls:
        if name == "launch_process" and args:
            arguments = args
            command_valid = args.get("command") == "ls -la" and args.get("cwd") == "/tmp"

    metrics = {
        "launch_process_call": MetricResult(
            score=1.0 if arguments else 0.0,
            is_score_valid=True,
            reason="launch_process called" if arguments else "launch_process missing",
            data={"arguments": arguments},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason in {"tool_calls", "stop"} else 0.0,
            is_score_valid=True,
            reason="finish_reason acceptable"
            if finish_reason in {"tool_calls", "stop"}
            else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        command_valid and finish_reason_present and no_forbidden_tags and no_xml_tags and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Command execution arguments correct"
        if all_checks_passed
        else "Command execution arguments incorrect or response invalid",
        metrics=metrics,
    )
    return row


_PEER_TOOL_PARAMETER_ROW = _build_row_from_payload(
    "peer-tool-parameter-format-errors", PEER_TOOL_PARAMETER_FORMAT_ERRORS_PAYLOAD
)


@evaluation_test(
    input_rows=[[_PEER_TOOL_PARAMETER_ROW]],
    completion_params=[_build_completion_params_from_payload(PEER_TOOL_PARAMETER_FORMAT_ERRORS_PAYLOAD)],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_streaming_tool_parameter_types(row: EvaluationRow) -> EvaluationRow:
    """Ensure streamed parameters respect expected JSON types."""

    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_parameter_formats", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    args = None
    types_valid = False
    for name, payload_args in calls:
        if name == "process_data" and isinstance(payload_args, dict):
            args = payload_args
            types_valid = isinstance(payload_args.get("count"), int) and isinstance(payload_args.get("enabled"), bool)

    metrics = {
        "process_data_call": MetricResult(
            score=1.0 if args else 0.0,
            is_score_valid=True,
            reason="process_data call present" if args else "process_data call missing",
        ),
        "types_valid": MetricResult(
            score=1.0 if types_valid else 0.0,
            is_score_valid=bool(args),
            reason="Numeric/boolean fields have correct types" if types_valid else "Type mismatch in arguments",
            data={"arguments": args},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason in {"tool_calls", "stop"} else 0.0,
            is_score_valid=True,
            reason="finish_reason acceptable"
            if finish_reason in {"tool_calls", "stop"}
            else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        types_valid and finish_reason_present and no_forbidden_tags and no_xml_tags and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Parameter types maintained" if all_checks_passed else "Parameter types incorrect or response invalid",
        metrics=metrics,
    )
    return row


# ============================================================================
# Reasoning Effort Tests
# ============================================================================

REASONING_DISABLED_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is 2+2? Explain your answer."),
    ]
)
REASONING_DISABLED_ROW.input_metadata.dataset_info = {
    "test_name": "reasoning_disabled",
    "description": "Verify reasoning_content is empty when reasoning_effort=none",
}


@pytest.mark.skipif(
    not SUPPORTS_REASONING,
    reason="Model does not support reasoning_effort parameter (EP_SUPPORTS_REASONING=0)",
)
@evaluation_test(
    input_rows=[[REASONING_DISABLED_ROW]],
    completion_params=[
        {
            "model": DEFAULT_MODEL_ID,  # Reasoning-capable model
            "reasoning_effort": "none",  # Explicitly disable reasoning
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
            "raw_output": True,  # Include raw model output for debugging
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_reasoning_effort_none_no_reasoning(row: EvaluationRow) -> EvaluationRow:
    """
    Verify that when reasoning_effort=none, reasoning_content is empty.

    Tests that reasoning-capable models respect the reasoning_effort=none parameter.
    """
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""

    has_content = bool(content_str.strip())
    reasoning_empty = reasoning_str == ""
    finish_reason_stop = finish_reason == "stop"

    metrics = {
        "has_content": MetricResult(
            score=1.0 if has_content else 0.0,
            is_score_valid=True,
            reason="Content present" if has_content else "No content",
            data={"content_preview": content_str[:100]},
        ),
        "reasoning_empty": MetricResult(
            score=1.0 if reasoning_empty else 0.0,
            is_score_valid=True,
            reason="reasoning_content is empty (as expected)"
            if reasoning_empty
            else f"Unexpected reasoning_content: {reasoning_str[:100]}",
            data={"reasoning_length": len(reasoning_str)},
        ),
        "finish_reason_stop": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        has_content
        and reasoning_empty
        and finish_reason_stop
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    # Build detailed failure reason
    failure_reasons = []
    if not has_content:
        failure_reasons.append("no content")
    if not reasoning_empty:
        failure_reasons.append(f"reasoning_content present (len={len(reasoning_str)})")
    if not finish_reason_stop:
        failure_reasons.append(f"finish_reason={finish_reason}")
    if not finish_reason_present:
        failure_reasons.append("finish_reason null")
    if not no_forbidden_tags:
        failure_reasons.append("forbidden tags detected")
    if not no_xml_tags:
        failure_reasons.append("XML tags detected")

    reason = (
        "reasoning_effort=none respected" if all_checks_passed else f"Compliance failed: {', '.join(failure_reasons)}"
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=reason,
        metrics=metrics,
    )
    return row


REASONING_ENABLED_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(
            role="user",
            content="Solve this problem: If Alice has 3 apples and Bob gives her 5 more, how many does she have?",
        ),
    ]
)
REASONING_ENABLED_ROW.input_metadata.dataset_info = {
    "test_name": "reasoning_enabled",
    "description": "Verify reasoning_content is present when reasoning_effort=low",
}


@pytest.mark.skipif(
    not SUPPORTS_REASONING,
    reason="Model does not support reasoning_effort parameter (EP_SUPPORTS_REASONING=0)",
)
@evaluation_test(
    input_rows=[[REASONING_ENABLED_ROW]],
    completion_params=[
        {
            "model": DEFAULT_MODEL_ID,  # Reasoning-capable model
            "reasoning_effort": "low",  # Enable reasoning
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
            "raw_output": True,  # Include raw model output for debugging
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_reasoning_effort_low_has_reasoning(row: EvaluationRow) -> EvaluationRow:
    """
    Verify that when reasoning_effort=low, reasoning_content is present.

    Tests that reasoning-capable models generate reasoning when requested.
    """
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""

    has_content = bool(content_str.strip())
    reasoning_present = bool(reasoning_str)
    finish_reason_stop = finish_reason == "stop"

    metrics = {
        "has_content": MetricResult(
            score=1.0 if has_content else 0.0,
            is_score_valid=True,
            reason="Content present" if has_content else "No content",
            data={"content_preview": content_str[:100]},
        ),
        "reasoning_present": MetricResult(
            score=1.0 if reasoning_present else 0.0,
            is_score_valid=True,
            reason="reasoning_content present (as expected)"
            if reasoning_present
            else "reasoning_content missing when reasoning_effort=low",
            data={"reasoning_length": len(reasoning_str), "reasoning_preview": reasoning_str[:200]},
        ),
        "finish_reason_stop": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        has_content
        and reasoning_present
        and finish_reason_stop
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    # Build detailed failure reason
    failure_reasons = []
    if not has_content:
        failure_reasons.append("no content")
    if not reasoning_present:
        failure_reasons.append("reasoning_content missing")
    if not finish_reason_stop:
        failure_reasons.append(f"finish_reason={finish_reason}")
    if not finish_reason_present:
        failure_reasons.append("finish_reason null")
    if not no_forbidden_tags:
        failure_reasons.append("forbidden tags detected")
    if not no_xml_tags:
        failure_reasons.append("XML tags detected")
    if not no_reasoning_leakage:
        failure_reasons.append("thinking phrases in content")

    reason = (
        "reasoning_effort=low produces reasoning_content"
        if all_checks_passed
        else f"Compliance failed: {', '.join(failure_reasons)}"
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=reason,
        metrics=metrics,
    )
    return row


# ============================================================================
# Tools + Reasoning Combination Test
# ============================================================================

TOOLS_WITH_REASONING_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant with access to tools."),
        Message(
            role="user",
            content="What's the weather in San Francisco? Think through which tool to use and why.",
        ),
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ],
)
TOOLS_WITH_REASONING_ROW.input_metadata.dataset_info = {
    "test_name": "tools_with_reasoning",
    "description": "Verify tools and reasoning work together in streaming",
}


@evaluation_test(
    input_rows=[[TOOLS_WITH_REASONING_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,  # Reasoning-capable model
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": 0.0,
                "stream": True,
                "raw_output": True,  # Include raw model output for debugging
            },
            "low",
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_streaming_tools_with_reasoning(row: EvaluationRow) -> EvaluationRow:
    """
    Verify that streaming works correctly when BOTH tools and reasoning are present.

    Requirements:
    - reasoning_content should be present (if SUPPORTS_REASONING)
    - tool_calls should be present
    - finish_reason should be "tool_calls"
    - No XML tags or reasoning leakage
    """
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    reasoning_present = bool(reasoning_str)
    has_tool_calls = len(calls) > 0
    finish_reason_tool_calls = finish_reason == "tool_calls"

    # Validate tool call has required params
    tool_call_valid = False
    if has_tool_calls:
        for name, args in calls:
            if name == "get_current_weather" and isinstance(args, dict):
                location = args.get("location", "")
                if "san francisco" in location.lower() or "sf" in location.lower():
                    tool_call_valid = True
                    break

    metrics = {
        "has_tool_calls": MetricResult(
            score=1.0 if has_tool_calls else 0.0,
            is_score_valid=True,
            reason="Tool calls present" if has_tool_calls else "No tool calls",
            data={"tool_call_count": len(calls)},
        ),
        "finish_reason_tool_calls": MetricResult(
            score=1.0 if finish_reason_tool_calls else 0.0,
            is_score_valid=True,
            reason="finish_reason is tool_calls"
            if finish_reason_tool_calls
            else f"Unexpected finish_reason: {finish_reason}",
        ),
        "tool_call_valid": MetricResult(
            score=1.0 if tool_call_valid else 0.0,
            is_score_valid=has_tool_calls,
            reason="Tool call has correct location" if tool_call_valid else "Tool call missing required params",
            data={"tool_calls": [{"name": name, "args": args} for name, args in calls]},
        ),
    }

    # Only add reasoning_present metric if model supports reasoning
    if SUPPORTS_REASONING:
        metrics["reasoning_present"] = MetricResult(
            score=1.0 if reasoning_present else 0.0,
            is_score_valid=True,
            reason="reasoning_content present" if reasoning_present else "reasoning_content missing",
            data={"reasoning_length": len(reasoning_str), "reasoning_preview": reasoning_str[:200]},
        )

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    # Build pass criteria - reasoning check is conditional
    all_checks_passed = (
        has_tool_calls
        and finish_reason_tool_calls
        and tool_call_valid
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )
    # Only require reasoning if model supports it
    if SUPPORTS_REASONING:
        all_checks_passed = all_checks_passed and reasoning_present

    # Build detailed failure reason
    failure_reasons = []
    if SUPPORTS_REASONING and not reasoning_present:
        failure_reasons.append("reasoning_content missing")
    if not has_tool_calls:
        failure_reasons.append("no tool calls")
    if not finish_reason_tool_calls:
        failure_reasons.append(f"finish_reason={finish_reason}")
    if not tool_call_valid:
        failure_reasons.append("tool params invalid")
    if not finish_reason_present:
        failure_reasons.append("finish_reason null")
    if not no_forbidden_tags:
        failure_reasons.append("forbidden tags detected")
    if not no_xml_tags:
        failure_reasons.append("XML tags detected")
    if not no_reasoning_leakage:
        failure_reasons.append("thinking phrases in content")

    reason = (
        "Tools + reasoning work together in streaming"
        if all_checks_passed
        else f"Compliance failed: {', '.join(failure_reasons)}"
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=reason,
        metrics=metrics,
    )
    return row


# ============================================================================
# Streaming Consistency Test (Shadow Test)
# ============================================================================

STREAMING_CONSISTENCY_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Count from 1 to 5 and explain why you're counting."),
    ]
)
STREAMING_CONSISTENCY_ROW.input_metadata.dataset_info = {
    "test_name": "streaming_consistency",
    "description": "Shadow test: verify stream=true produces identical output to stream=false",
}


@evaluation_test(
    input_rows=[[STREAMING_CONSISTENCY_ROW]],
    completion_params=[
        {
            "model": os.getenv("EP_MODEL", DEFAULT_MODEL_ID),
            "max_tokens": os.getenv("EP_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            "temperature": 0.0,  # Deterministic for consistency
            "stream": False,  # Will be overridden by custom rollout
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
async def test_streaming_output_consistency(row: EvaluationRow) -> EvaluationRow:
    """
    Shadow stress test for streaming consistency.

    Strategy:
    1. Run request with stream=false, capture output
    2. Run request with stream=true + forced_generation (same tokens), concat chunks
    3. Verify concatenated streaming output matches non-streaming output

    This catches bugs like: "you're5" (streaming) vs "you're 5" (non-streaming)
    """
    from openai import AsyncOpenAI

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason="FIREWORKS_API_KEY not set",
        )
        return row

    model = os.getenv("EP_MODEL", DEFAULT_MODEL_ID)
    messages = [msg.model_dump() for msg in row.messages]

    try:
        async with AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1",
        ) as client:
            # Step 1: Get non-streaming output
            response_non_stream = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=DEFAULT_MAX_TOKENS,
                stream=False,
            )

            non_stream_content = response_non_stream.choices[0].message.content or ""
            non_stream_tool_calls = response_non_stream.choices[0].message.tool_calls
            non_stream_finish = response_non_stream.choices[0].finish_reason

            # Step 2: Get streaming output with forced generation
            # Extract token IDs from non-streaming response if available
            # (Note: OpenAI API doesn't expose token IDs directly, so we'll just verify
            # that streaming with same params produces same output)
            stream_response = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=DEFAULT_MAX_TOKENS,
                stream=True,
            )

            # Concatenate streaming chunks
            stream_content_parts = []
            stream_finish = None
            stream_tool_calls = None

            async for chunk in stream_response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        stream_content_parts.append(delta.content)
                    if delta.tool_calls:
                        stream_tool_calls = delta.tool_calls
                    if chunk.choices[0].finish_reason:
                        stream_finish = chunk.choices[0].finish_reason

            stream_content = "".join(stream_content_parts)

            # Step 3: Compare outputs
            content_match = non_stream_content == stream_content
            finish_match = non_stream_finish == stream_finish

            # Tool calls comparison (basic check)
            tool_calls_match = True
            if non_stream_tool_calls and stream_tool_calls:
                tool_calls_match = len(non_stream_tool_calls) == len(stream_tool_calls)
            elif non_stream_tool_calls or stream_tool_calls:
                tool_calls_match = False

            metrics = {
                "content_match": MetricResult(
                    score=1.0 if content_match else 0.0,
                    is_score_valid=True,
                    reason="Content identical"
                    if content_match
                    else f"Content mismatch: non-stream='{non_stream_content[:100]}' vs stream='{stream_content[:100]}'",
                    data={
                        "non_stream_length": len(non_stream_content),
                        "stream_length": len(stream_content),
                    },
                ),
                "finish_reason_match": MetricResult(
                    score=1.0 if finish_match else 0.0,
                    is_score_valid=True,
                    reason=f"finish_reason: {non_stream_finish}"
                    if finish_match
                    else f"Mismatch: {non_stream_finish} vs {stream_finish}",
                ),
                "tool_calls_match": MetricResult(
                    score=1.0 if tool_calls_match else 0.0,
                    is_score_valid=True,
                    reason="Tool calls consistent" if tool_calls_match else "Tool call count mismatch",
                ),
            }

            all_match = content_match and finish_match and tool_calls_match

            row.evaluation_result = EvaluateResult(
                score=1.0 if all_match else 0.0,
                is_score_valid=True,
                reason="Streaming output matches non-streaming" if all_match else "Streaming inconsistency detected",
                metrics=metrics,
            )

    except Exception as e:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason=f"Test execution failed: {e}",
        )

    return row


# ============================================================================
# Non-Streaming Tests (Mirror of Streaming Tests)
# ============================================================================


# Test 1: Structured Output (Non-Streaming)
@evaluation_test(
    input_rows=[[STRUCTURED_OUTPUT_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "stream": False,  # Non-streaming
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "response_format": STRUCTURED_RESPONSE_FORMAT,
                "raw_output": True,  # Include raw model output for debugging
            },
            "none",
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=1.0,
    num_runs=1,
    mode="pointwise",
)
def test_non_streaming_structured_output(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Validate structured output with JSON schema."""
    assistant_msg = row.last_assistant_message()
    if assistant_msg is None:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason="No assistant message produced",
            metrics={},
        )
        return row
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("structured_output_non_stream", assistant_msg, finish_reason)

    content_str = _coerce_content_to_str(assistant_msg.content)
    reasoning_str = assistant_msg.reasoning_content or ""
    parsed_content = _safe_json_loads(content_str)
    parsed_reasoning = _safe_json_loads(reasoning_str) if reasoning_str else None

    required_fields = {"location", "temperature", "conditions"}

    content_present = bool(content_str.strip())
    content_is_json = parsed_content is not None
    required_keys_present = content_is_json and required_fields <= set(parsed_content.keys())
    temperature_is_number = content_is_json and isinstance(parsed_content.get("temperature"), (int, float))
    location_valid = content_is_json and parsed_content.get("location") in {"London", "New York"}
    reasoning_contains_payload = parsed_reasoning is not None
    finish_reason_expected = finish_reason == "stop"

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

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        content_present
        and content_is_json
        and required_keys_present
        and temperature_is_number
        and location_valid
        and not reasoning_contains_payload
        and finish_reason_expected
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=(
            "Structured output returned in assistant content"
            if all_checks_passed
            else "Structured output missing or response malformed"
        ),
        metrics=metrics,
    )
    return row


# Test 2: Simple Completion (Non-Streaming)
_SIMPLE_COMPLETION_NON_STREAM_ROW = _build_row_from_payload("simple-completion-non-stream", PEER_SIMPLE_STREAM_PAYLOAD)


@evaluation_test(
    input_rows=[[_SIMPLE_COMPLETION_NON_STREAM_ROW]],
    completion_params=[
        {
            **_build_completion_params_from_payload(PEER_SIMPLE_STREAM_PAYLOAD),
            "stream": False,  # Override to non-streaming
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_non_streaming_simple_completion(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Validate plain text completion."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("simple_completion_non_stream", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""

    has_content = bool(content_str.strip())
    finish_reason_stop = finish_reason == "stop"
    reasoning_empty = reasoning_str == ""
    has_tool_calls = bool(assistant_msg and assistant_msg.tool_calls)

    metrics = {
        "has_content": MetricResult(
            score=1.0 if has_content else 0.0,
            is_score_valid=True,
            reason="Assistant content present" if has_content else "Assistant content empty",
            data={"preview": content_str[:120]},
        ),
        "finish_reason": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected finish_reason: {finish_reason}",
        ),
        "reasoning_empty": MetricResult(
            score=1.0 if reasoning_empty else 0.0,
            is_score_valid=True,
            reason="Reasoning is empty" if reasoning_empty else "Unexpected reasoning output",
            data={"reasoning": reasoning_str},
        ),
        "tool_calls_absent": MetricResult(
            score=1.0 if not has_tool_calls else 0.0,
            is_score_valid=True,
            reason="No tool calls emitted" if not has_tool_calls else "Unexpected tool calls emitted",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        has_content
        and finish_reason_stop
        and reasoning_empty
        and not has_tool_calls
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Simple completion compliant" if all_checks_passed else "Simple completion failed compliance checks",
        metrics=metrics,
    )
    return row


# Test 3: Tool Call (Non-Streaming)
TOOL_CALL_NON_STREAM_ROW = EvaluationRow(
    messages=[
        Message(role="system", content=TOOL_SYSTEM_PROMPT),
        Message(role="user", content="What's the weather in Boston in Fahrenheit?"),
    ],
    tools=WEATHER_TOOL_DEFINITION,
)
TOOL_CALL_NON_STREAM_ROW.input_metadata.dataset_info = {
    "test_name": "tool_call_non_stream",
    "description": "Non-streaming tool call test",
}


@evaluation_test(
    input_rows=[[TOOL_CALL_NON_STREAM_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "stream": False,  # Non-streaming
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "raw_output": True,  # Include raw model output for debugging
            },
            "none",
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_non_streaming_single_tool_call(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Validate single tool call."""
    assistant_msg = row.last_assistant_message()
    if assistant_msg is None:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason="No assistant message produced",
            metrics={},
        )
        return row

    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("tool_call_non_stream", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content)
    reasoning_str = (assistant_msg.reasoning_content or "").strip()
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
    tool_call_count = row.execution_metadata.tool_call_count

    has_tool_call = len(tool_calls) > 0
    exactly_one_tool_call = len(tool_calls) == 1
    finish_reason_tool_calls = finish_reason == "tool_calls"
    tool_call_count_matches = tool_call_count == len(tool_calls)

    tool_call_arguments_valid = False
    parsed_arguments = None
    if exactly_one_tool_call:
        # Use helper to normalize tool call (handles both dict and Pydantic objects)
        name, parsed_arguments = _normalize_tool_call(tool_calls[0])
        tool_call_arguments_valid = (
            isinstance(parsed_arguments, dict)
            and ("boston" in (parsed_arguments.get("location") or "").lower())
            and parsed_arguments.get("unit") == "fahrenheit"
        )

    base_checks_passed = (
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

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        base_checks_passed and finish_reason_present and no_forbidden_tags and no_xml_tags and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Tool call completed correctly" if all_checks_passed else "Tool call behaviour invalid",
        metrics=metrics,
    )
    return row


# Test 4: Multiple Tool Calls (Non-Streaming)
_MULTI_TOOL_CALLS_NON_STREAM_ROW = _build_row_from_payload("multi-tool-calls-non-stream", MULTI_TOOL_CALLS_PAYLOAD)


@pytest.mark.skipif(
    not SUPPORTS_MULTIPLE_TOOL_CALLS,
    reason="Model does not support multiple tool calls (EP_SUPPORTS_MULTIPLE_TOOL_CALLS=0)",
)
@evaluation_test(
    input_rows=[[_MULTI_TOOL_CALLS_NON_STREAM_ROW]],
    completion_params=[
        {
            **_build_completion_params_from_payload(MULTI_TOOL_CALLS_PAYLOAD),
            "stream": False,  # Override to non-streaming
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
)
def test_non_streaming_multiple_tool_calls(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Validate multiple tool calls."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason
    _debug_log_assistant_message("multi_tool_calls_non_stream", assistant_msg, finish_reason)
    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    names = [name for name, _ in calls if name]
    has_weather = names.count("get_current_weather") >= 2
    has_air_quality = "get_air_quality" in names

    metrics = {
        "tool_calls_count": MetricResult(
            score=1.0 if len(calls) >= 3 else 0.0,
            is_score_valid=True,
            reason="Three or more tool calls" if len(calls) >= 3 else "Fewer than three tool calls",
            data={"tool_calls": names},
        ),
        "includes_weather_calls": MetricResult(
            score=1.0 if has_weather else 0.0,
            is_score_valid=True,
            reason="Weather tool called for multiple cities" if has_weather else "Insufficient weather tool calls",
        ),
        "includes_air_quality": MetricResult(
            score=1.0 if has_air_quality else 0.0,
            is_score_valid=True,
            reason="Air quality tool called" if has_air_quality else "Air quality tool missing",
        ),
        "finish_reason_tool_calls": MetricResult(
            score=1.0 if finish_reason == "tool_calls" else 0.0,
            is_score_valid=True,
            reason=(
                "finish_reason is tool_calls"
                if finish_reason == "tool_calls"
                else f"Unexpected finish_reason: {finish_reason}"
            ),
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        len(calls) >= 3
        and has_weather
        and has_air_quality
        and finish_reason == "tool_calls"
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Multiple tool calls emitted"
        if all_checks_passed
        else "Expected multi-tool response missing or invalid",
        metrics=metrics,
    )
    return row


# Test 5: Reasoning Disabled (Non-Streaming)
REASONING_DISABLED_NON_STREAM_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is 2+2? Explain your answer."),
    ]
)
REASONING_DISABLED_NON_STREAM_ROW.input_metadata.dataset_info = {
    "test_name": "reasoning_disabled_non_stream",
    "description": "Non-streaming: verify reasoning_content empty when reasoning_effort=none",
}


@pytest.mark.skipif(
    not SUPPORTS_REASONING,
    reason="Model does not support reasoning_effort parameter (EP_SUPPORTS_REASONING=0)",
)
@evaluation_test(
    input_rows=[[REASONING_DISABLED_NON_STREAM_ROW]],
    completion_params=[
        {
            "model": DEFAULT_MODEL_ID,
            "reasoning_effort": "none",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0,
            "stream": False,  # Non-streaming
            "raw_output": True,  # Include raw model output for debugging
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_reasoning_effort_none_no_reasoning_non_stream(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Verify reasoning_content empty when reasoning_effort=none."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""

    has_content = bool(content_str.strip())
    reasoning_empty = reasoning_str == ""
    finish_reason_stop = finish_reason == "stop"

    metrics = {
        "has_content": MetricResult(
            score=1.0 if has_content else 0.0,
            is_score_valid=True,
            reason="Content present" if has_content else "No content",
            data={"content_preview": content_str[:100]},
        ),
        "reasoning_empty": MetricResult(
            score=1.0 if reasoning_empty else 0.0,
            is_score_valid=True,
            reason="reasoning_content is empty (as expected)"
            if reasoning_empty
            else f"Unexpected reasoning_content: {reasoning_str[:100]}",
            data={"reasoning_length": len(reasoning_str)},
        ),
        "finish_reason_stop": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        has_content
        and reasoning_empty
        and finish_reason_stop
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    # Build detailed failure reason
    failure_reasons = []
    if not has_content:
        failure_reasons.append("no content")
    if not reasoning_empty:
        failure_reasons.append(f"reasoning_content present (len={len(reasoning_str)})")
    if not finish_reason_stop:
        failure_reasons.append(f"finish_reason={finish_reason}")
    if not finish_reason_present:
        failure_reasons.append("finish_reason null")
    if not no_forbidden_tags:
        failure_reasons.append("forbidden tags detected")
    if not no_xml_tags:
        failure_reasons.append("XML tags detected")

    reason = (
        "reasoning_effort=none respected" if all_checks_passed else f"Compliance failed: {', '.join(failure_reasons)}"
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=reason,
        metrics=metrics,
    )
    return row


# Test 6: Reasoning Enabled (Non-Streaming)
REASONING_ENABLED_NON_STREAM_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(
            role="user",
            content="Solve this problem: If Alice has 3 apples and Bob gives her 5 more, how many does she have?",
        ),
    ]
)
REASONING_ENABLED_NON_STREAM_ROW.input_metadata.dataset_info = {
    "test_name": "reasoning_enabled_non_stream",
    "description": "Non-streaming: verify reasoning_content present when reasoning_effort=low",
}


@pytest.mark.skipif(
    not SUPPORTS_REASONING,
    reason="Model does not support reasoning_effort parameter (EP_SUPPORTS_REASONING=0)",
)
@evaluation_test(
    input_rows=[[REASONING_ENABLED_NON_STREAM_ROW]],
    completion_params=[
        {
            "model": DEFAULT_MODEL_ID,
            "reasoning_effort": "low",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0,
            "stream": False,  # Non-streaming
            "raw_output": True,  # Include raw model output for debugging
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_reasoning_effort_low_has_reasoning_non_stream(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Verify reasoning_content present when reasoning_effort=low."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""

    has_content = bool(content_str.strip())
    reasoning_present = bool(reasoning_str)
    finish_reason_stop = finish_reason == "stop"

    metrics = {
        "has_content": MetricResult(
            score=1.0 if has_content else 0.0,
            is_score_valid=True,
            reason="Content present" if has_content else "No content",
            data={"content_preview": content_str[:100]},
        ),
        "reasoning_present": MetricResult(
            score=1.0 if reasoning_present else 0.0,
            is_score_valid=True,
            reason="reasoning_content present (as expected)"
            if reasoning_present
            else "reasoning_content missing when reasoning_effort=low",
            data={"reasoning_length": len(reasoning_str), "reasoning_preview": reasoning_str[:200]},
        ),
        "finish_reason_stop": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected finish_reason: {finish_reason}",
        ),
    }

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    all_checks_passed = (
        has_content
        and reasoning_present
        and finish_reason_stop
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )

    # Build detailed failure reason
    failure_reasons = []
    if not has_content:
        failure_reasons.append("no content")
    if not reasoning_present:
        failure_reasons.append("reasoning_content missing")
    if not finish_reason_stop:
        failure_reasons.append(f"finish_reason={finish_reason}")
    if not finish_reason_present:
        failure_reasons.append("finish_reason null")
    if not no_forbidden_tags:
        failure_reasons.append("forbidden tags detected")
    if not no_xml_tags:
        failure_reasons.append("XML tags detected")
    if not no_reasoning_leakage:
        failure_reasons.append("thinking phrases in content")

    reason = (
        "reasoning_effort=low produces reasoning_content"
        if all_checks_passed
        else f"Compliance failed: {', '.join(failure_reasons)}"
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=reason,
        metrics=metrics,
    )
    return row


# Test 7: Tools + Reasoning (Non-Streaming)
TOOLS_WITH_REASONING_NON_STREAM_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant with access to tools."),
        Message(
            role="user",
            content="What's the weather in San Francisco? Think through which tool to use and why.",
        ),
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ],
)
TOOLS_WITH_REASONING_NON_STREAM_ROW.input_metadata.dataset_info = {
    "test_name": "tools_with_reasoning_non_stream",
    "description": "Non-streaming: verify tools and reasoning work together",
}


@evaluation_test(
    input_rows=[[TOOLS_WITH_REASONING_NON_STREAM_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": 0.0,
                "stream": False,  # Non-streaming
                "raw_output": True,  # Include raw model output for debugging
            },
            "low",
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_non_streaming_tools_with_reasoning(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Verify tools and reasoning work together."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    reasoning_present = bool(reasoning_str)
    has_tool_calls = len(calls) > 0
    finish_reason_tool_calls = finish_reason == "tool_calls"

    # Validate tool call has required params
    tool_call_valid = False
    if has_tool_calls:
        for name, args in calls:
            if name == "get_current_weather" and isinstance(args, dict):
                location = args.get("location", "")
                if "san francisco" in location.lower() or "sf" in location.lower():
                    tool_call_valid = True
                    break

    metrics = {
        "has_tool_calls": MetricResult(
            score=1.0 if has_tool_calls else 0.0,
            is_score_valid=True,
            reason="Tool calls present" if has_tool_calls else "No tool calls",
            data={"tool_call_count": len(calls)},
        ),
        "finish_reason_tool_calls": MetricResult(
            score=1.0 if finish_reason_tool_calls else 0.0,
            is_score_valid=True,
            reason="finish_reason is tool_calls"
            if finish_reason_tool_calls
            else f"Unexpected finish_reason: {finish_reason}",
        ),
        "tool_call_valid": MetricResult(
            score=1.0 if tool_call_valid else 0.0,
            is_score_valid=has_tool_calls,
            reason="Tool call has correct location" if tool_call_valid else "Tool call missing required params",
            data={"tool_calls": [{"name": name, "args": args} for name, args in calls]},
        ),
    }

    # Only add reasoning_present metric if model supports reasoning
    if SUPPORTS_REASONING:
        metrics["reasoning_present"] = MetricResult(
            score=1.0 if reasoning_present else 0.0,
            is_score_valid=True,
            reason="reasoning_content present" if reasoning_present else "reasoning_content missing",
            data={"reasoning_length": len(reasoning_str), "reasoning_preview": reasoning_str[:200]},
        )

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    # Build pass criteria - reasoning check is conditional
    all_checks_passed = (
        has_tool_calls
        and finish_reason_tool_calls
        and tool_call_valid
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )
    # Only require reasoning if model supports it
    if SUPPORTS_REASONING:
        all_checks_passed = all_checks_passed and reasoning_present

    # Build detailed failure reason
    failure_reasons = []
    if SUPPORTS_REASONING and not reasoning_present:
        failure_reasons.append("reasoning_content missing")
    if not has_tool_calls:
        failure_reasons.append("no tool calls")
    if not finish_reason_tool_calls:
        failure_reasons.append(f"finish_reason={finish_reason}")
    if not tool_call_valid:
        failure_reasons.append("tool params invalid")
    if not finish_reason_present:
        failure_reasons.append("finish_reason null")
    if not no_forbidden_tags:
        failure_reasons.append("forbidden tags detected")
    if not no_xml_tags:
        failure_reasons.append("XML tags detected")
    if not no_reasoning_leakage:
        failure_reasons.append("thinking phrases in content")

    reason = (
        "Tools + reasoning work together in streaming"
        if all_checks_passed
        else f"Compliance failed: {', '.join(failure_reasons)}"
    )

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason=reason,
        metrics=metrics,
    )
    return row


# ============================================================================
# Missing Permutations: Reasoning + Structured JSON
# ============================================================================

STRUCTURED_OUTPUT_WITH_REASONING_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful math assistant."),
        Message(
            role="user",
            content="Solve this problem step by step: If a train travels 120 km in 2 hours, what is its average speed? Return your answer as JSON with 'speed_kmh' and 'explanation' fields.",
        ),
    ]
)
STRUCTURED_OUTPUT_WITH_REASONING_ROW.input_metadata.dataset_info = {
    "test_name": "structured_output_with_reasoning_stream",
    "description": "Streaming: structured JSON + reasoning",
}

STRUCTURED_JSON_SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "speed_kmh": {"type": "number", "description": "Speed in km/h"},
            "explanation": {"type": "string", "description": "Brief explanation"},
        },
        "required": ["speed_kmh", "explanation"],
    },
}


@evaluation_test(
    input_rows=[[STRUCTURED_OUTPUT_WITH_REASONING_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "stream": True,
                "response_format": STRUCTURED_JSON_SCHEMA,
                "temperature": 0.0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "raw_output": True,  # Include raw model output for debugging
            },
            "low",
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_streaming_structured_output_with_reasoning(row: EvaluationRow) -> EvaluationRow:
    """Validate structured JSON output with reasoning in streaming mode."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    parsed_content = _safe_json_loads(content_str)

    content_is_json = parsed_content is not None
    has_required_keys = content_is_json and {"speed_kmh", "explanation"} <= set(parsed_content.keys())
    speed_is_number = content_is_json and isinstance(parsed_content.get("speed_kmh"), (int, float))
    reasoning_present = bool(reasoning_str)
    finish_reason_stop = finish_reason == "stop"

    metrics = {
        "content_is_json": MetricResult(
            score=1.0 if content_is_json else 0.0,
            is_score_valid=True,
            reason="Content is valid JSON" if content_is_json else "Failed to parse JSON",
            data={"content": content_str[:200]},
        ),
        "has_required_keys": MetricResult(
            score=1.0 if has_required_keys else 0.0,
            is_score_valid=content_is_json,
            reason="Required keys present" if has_required_keys else "Missing required keys",
            data={"parsed": parsed_content},
        ),
        "speed_is_number": MetricResult(
            score=1.0 if speed_is_number else 0.0,
            is_score_valid=content_is_json,
            reason="speed_kmh is numeric" if speed_is_number else "speed_kmh not numeric",
        ),
        "finish_reason_stop": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected: {finish_reason}",
        ),
    }

    # Only add reasoning_present metric if model supports reasoning
    if SUPPORTS_REASONING:
        metrics["reasoning_present"] = MetricResult(
            score=1.0 if reasoning_present else 0.0,
            is_score_valid=True,
            reason="reasoning_content present" if reasoning_present else "reasoning_content missing",
            data={"reasoning_length": len(reasoning_str), "reasoning_preview": reasoning_str[:200]},
        )

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    # Build pass criteria - reasoning check is conditional
    all_checks_passed = (
        content_is_json
        and has_required_keys
        and speed_is_number
        and finish_reason_stop
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )
    # Only require reasoning if model supports it
    if SUPPORTS_REASONING:
        all_checks_passed = all_checks_passed and reasoning_present

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Structured JSON + reasoning work together" if all_checks_passed else "JSON or reasoning invalid",
        metrics=metrics,
    )
    return row


# Non-streaming version
STRUCTURED_OUTPUT_WITH_REASONING_NON_STREAM_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful math assistant."),
        Message(
            role="user",
            content="Solve this problem step by step: If a train travels 120 km in 2 hours, what is its average speed? Return your answer as JSON with 'speed_kmh' and 'explanation' fields.",
        ),
    ]
)
STRUCTURED_OUTPUT_WITH_REASONING_NON_STREAM_ROW.input_metadata.dataset_info = {
    "test_name": "structured_output_with_reasoning_non_stream",
    "description": "Non-streaming: structured JSON + reasoning",
}


@evaluation_test(
    input_rows=[[STRUCTURED_OUTPUT_WITH_REASONING_NON_STREAM_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "stream": False,
                "response_format": STRUCTURED_JSON_SCHEMA,
                "temperature": 0.0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "raw_output": True,  # Include raw model output for debugging
            },
            "low",
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_non_streaming_structured_output_with_reasoning(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Validate structured JSON output with reasoning."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    parsed_content = _safe_json_loads(content_str)

    content_is_json = parsed_content is not None
    has_required_keys = content_is_json and {"speed_kmh", "explanation"} <= set(parsed_content.keys())
    speed_is_number = content_is_json and isinstance(parsed_content.get("speed_kmh"), (int, float))
    reasoning_present = bool(reasoning_str)
    finish_reason_stop = finish_reason == "stop"

    metrics = {
        "content_is_json": MetricResult(
            score=1.0 if content_is_json else 0.0,
            is_score_valid=True,
            reason="Content is valid JSON" if content_is_json else "Failed to parse JSON",
            data={"content": content_str[:200]},
        ),
        "has_required_keys": MetricResult(
            score=1.0 if has_required_keys else 0.0,
            is_score_valid=content_is_json,
            reason="Required keys present" if has_required_keys else "Missing required keys",
            data={"parsed": parsed_content},
        ),
        "speed_is_number": MetricResult(
            score=1.0 if speed_is_number else 0.0,
            is_score_valid=content_is_json,
            reason="speed_kmh is numeric" if speed_is_number else "speed_kmh not numeric",
        ),
        "finish_reason_stop": MetricResult(
            score=1.0 if finish_reason_stop else 0.0,
            is_score_valid=True,
            reason="finish_reason is stop" if finish_reason_stop else f"Unexpected: {finish_reason}",
        ),
    }

    # Only add reasoning_present metric if model supports reasoning
    if SUPPORTS_REASONING:
        metrics["reasoning_present"] = MetricResult(
            score=1.0 if reasoning_present else 0.0,
            is_score_valid=True,
            reason="reasoning_content present" if reasoning_present else "reasoning_content missing",
            data={"reasoning_length": len(reasoning_str), "reasoning_preview": reasoning_str[:200]},
        )

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    # Build pass criteria - reasoning check is conditional
    all_checks_passed = (
        content_is_json
        and has_required_keys
        and speed_is_number
        and finish_reason_stop
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )
    # Only require reasoning if model supports it
    if SUPPORTS_REASONING:
        all_checks_passed = all_checks_passed and reasoning_present

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Structured JSON + reasoning work together" if all_checks_passed else "JSON or reasoning invalid",
        metrics=metrics,
    )
    return row


# ============================================================================
# Missing Permutations: Reasoning + Multiple Tools
# ============================================================================

MULTIPLE_TOOLS_WITH_REASONING_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant with access to tools."),
        Message(
            role="user",
            content="Get the weather for Boston, San Francisco, and Seattle (all in Fahrenheit). Think about which cities to check and explain your approach.",
        ),
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
        }
    ],
)
MULTIPLE_TOOLS_WITH_REASONING_ROW.input_metadata.dataset_info = {
    "test_name": "multiple_tools_with_reasoning_stream",
    "description": "Streaming: multiple tool calls + reasoning",
}


@pytest.mark.skipif(
    not SUPPORTS_MULTIPLE_TOOL_CALLS,
    reason="Model does not support multiple tool calls (EP_SUPPORTS_MULTIPLE_TOOL_CALLS=0)",
)
@evaluation_test(
    input_rows=[[MULTIPLE_TOOLS_WITH_REASONING_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "stream": True,
                "temperature": 0.0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "raw_output": True,  # Include raw model output for debugging
            },
            "low",
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_streaming_multiple_tools_with_reasoning(row: EvaluationRow) -> EvaluationRow:
    """Validate multiple tool calls with reasoning in streaming mode."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    reasoning_present = bool(reasoning_str)
    has_multiple_tools = len(calls) >= 3
    finish_reason_tool_calls = finish_reason == "tool_calls"

    # Check that all 3 cities are covered
    cities_covered = set()
    for name, args in calls:
        if name == "get_current_weather" and isinstance(args, dict):
            location = (args.get("location") or "").lower()
            if "boston" in location:
                cities_covered.add("boston")
            if "san francisco" in location or "sf" in location:
                cities_covered.add("san_francisco")
            if "seattle" in location:
                cities_covered.add("seattle")

    all_cities_covered = len(cities_covered) == 3

    metrics = {
        "has_multiple_tools": MetricResult(
            score=1.0 if has_multiple_tools else 0.0,
            is_score_valid=True,
            reason=f"{len(calls)} tool calls (expected 3+)" if has_multiple_tools else f"Only {len(calls)} tool calls",
            data={"tool_call_count": len(calls)},
        ),
        "all_cities_covered": MetricResult(
            score=1.0 if all_cities_covered else 0.0,
            is_score_valid=has_multiple_tools,
            reason="All 3 cities covered" if all_cities_covered else f"Only {len(cities_covered)} cities covered",
            data={"cities": list(cities_covered), "tool_calls": [{"name": n, "args": a} for n, a in calls]},
        ),
        "finish_reason_tool_calls": MetricResult(
            score=1.0 if finish_reason_tool_calls else 0.0,
            is_score_valid=True,
            reason="finish_reason is tool_calls" if finish_reason_tool_calls else f"Unexpected: {finish_reason}",
        ),
    }

    # Only add reasoning_present metric if model supports reasoning
    if SUPPORTS_REASONING:
        metrics["reasoning_present"] = MetricResult(
            score=1.0 if reasoning_present else 0.0,
            is_score_valid=True,
            reason="reasoning_content present" if reasoning_present else "reasoning_content missing",
            data={"reasoning_length": len(reasoning_str), "reasoning_preview": reasoning_str[:200]},
        )

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    # Build pass criteria - reasoning check is conditional
    all_checks_passed = (
        has_multiple_tools
        and all_cities_covered
        and finish_reason_tool_calls
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )
    # Only require reasoning if model supports it
    if SUPPORTS_REASONING:
        all_checks_passed = all_checks_passed and reasoning_present

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Multiple tools + reasoning work together"
        if all_checks_passed
        else "Multiple tools or reasoning invalid",
        metrics=metrics,
    )
    return row


# Non-streaming version
MULTIPLE_TOOLS_WITH_REASONING_NON_STREAM_ROW = EvaluationRow(
    messages=[
        Message(role="system", content="You are a helpful assistant with access to tools."),
        Message(
            role="user",
            content="Get the weather for Boston, San Francisco, and Seattle (all in Fahrenheit). Think about which cities to check and explain your approach.",
        ),
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
        }
    ],
)
MULTIPLE_TOOLS_WITH_REASONING_NON_STREAM_ROW.input_metadata.dataset_info = {
    "test_name": "multiple_tools_with_reasoning_non_stream",
    "description": "Non-streaming: multiple tool calls + reasoning",
}


@pytest.mark.skipif(
    not SUPPORTS_MULTIPLE_TOOL_CALLS,
    reason="Model does not support multiple tool calls (EP_SUPPORTS_MULTIPLE_TOOL_CALLS=0)",
)
@evaluation_test(
    input_rows=[[MULTIPLE_TOOLS_WITH_REASONING_NON_STREAM_ROW]],
    completion_params=[
        _maybe_add_reasoning_effort(
            {
                "model": DEFAULT_MODEL_ID,
                "stream": False,
                "temperature": 0.0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "raw_output": True,  # Include raw model output for debugging
            },
            "low",
        )
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=1.0,
    mode="pointwise",
)
def test_non_streaming_multiple_tools_with_reasoning(row: EvaluationRow) -> EvaluationRow:
    """Non-streaming version: Validate multiple tool calls with reasoning."""
    assistant_msg = row.last_assistant_message()
    finish_reason = row.execution_metadata.finish_reason

    content_str = _coerce_content_to_str(assistant_msg.content) if assistant_msg else ""
    reasoning_str = (assistant_msg.reasoning_content or "").strip() if assistant_msg else ""
    calls = _collect_tool_calls(assistant_msg.tool_calls if assistant_msg else [])

    reasoning_present = bool(reasoning_str)
    has_multiple_tools = len(calls) >= 3
    finish_reason_tool_calls = finish_reason == "tool_calls"

    # Check that all 3 cities are covered
    cities_covered = set()
    for name, args in calls:
        if name == "get_current_weather" and isinstance(args, dict):
            location = (args.get("location") or "").lower()
            if "boston" in location:
                cities_covered.add("boston")
            if "san francisco" in location or "sf" in location:
                cities_covered.add("san_francisco")
            if "seattle" in location:
                cities_covered.add("seattle")

    all_cities_covered = len(cities_covered) == 3

    metrics = {
        "has_multiple_tools": MetricResult(
            score=1.0 if has_multiple_tools else 0.0,
            is_score_valid=True,
            reason=f"{len(calls)} tool calls (expected 3+)" if has_multiple_tools else f"Only {len(calls)} tool calls",
            data={"tool_call_count": len(calls)},
        ),
        "all_cities_covered": MetricResult(
            score=1.0 if all_cities_covered else 0.0,
            is_score_valid=has_multiple_tools,
            reason="All 3 cities covered" if all_cities_covered else f"Only {len(cities_covered)} cities covered",
            data={"cities": list(cities_covered), "tool_calls": [{"name": n, "args": a} for n, a in calls]},
        ),
        "finish_reason_tool_calls": MetricResult(
            score=1.0 if finish_reason_tool_calls else 0.0,
            is_score_valid=True,
            reason="finish_reason is tool_calls" if finish_reason_tool_calls else f"Unexpected: {finish_reason}",
        ),
    }

    # Only add reasoning_present metric if model supports reasoning
    if SUPPORTS_REASONING:
        metrics["reasoning_present"] = MetricResult(
            score=1.0 if reasoning_present else 0.0,
            is_score_valid=True,
            reason="reasoning_content present" if reasoning_present else "reasoning_content missing",
            data={"reasoning_length": len(reasoning_str), "reasoning_preview": reasoning_str[:200]},
        )

    finish_reason_present, no_forbidden_tags, no_xml_tags, no_reasoning_leakage = _augment_metrics_with_common_checks(
        metrics, finish_reason, content_str, reasoning_str
    )

    # Build pass criteria - reasoning check is conditional
    all_checks_passed = (
        has_multiple_tools
        and all_cities_covered
        and finish_reason_tool_calls
        and finish_reason_present
        and no_forbidden_tags
        and no_xml_tags
        and no_reasoning_leakage
    )
    # Only require reasoning if model supports it
    if SUPPORTS_REASONING:
        all_checks_passed = all_checks_passed and reasoning_present

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_checks_passed else 0.0,
        is_score_valid=True,
        reason="Multiple tools + reasoning work together"
        if all_checks_passed
        else "Multiple tools or reasoning invalid",
        metrics=metrics,
    )
    return row
