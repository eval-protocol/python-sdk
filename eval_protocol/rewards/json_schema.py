import json
import re
from typing import Any, Dict, List, Optional, Union

from ..models import EvaluateResult, Message, MetricResult, ChatCompletionContentPartTextParam
from ..typed_interface import reward_function
from .function_calling import (
    calculate_jaccard_similarity,
    extract_schema_properties,
    normalize_schema,
)


@reward_function
def json_schema_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Union[List[Message], List[Dict[str, Any]]]] = None,
    json_content: Optional[Union[Dict[str, Any], str]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate JSON content against an expected schema using Jaccard similarity.
    The model's response (containing JSON) is assumed to be the last message in the `messages` list.

    This reward function compares the structure of JSON content against an
    expected schema and calculates a similarity score using Jaccard similarity.
    It repurposes the same approach used for function calling validation but for
    general JSON schema validation.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. Not directly used by this reward.
        json_content: The JSON content to evaluate (if not provided, extracts
                      from the last message).
        expected_schema: The expected schema for the JSON content.
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult with score and metrics
    """
    metrics = {}

    if json_content is None:
        if not messages:
            return EvaluateResult(
                score=0.0,
                reason="No messages provided to extract JSON content.",
                metrics={"error": MetricResult(score=0.0, reason="No messages provided", is_score_valid=False)},
            )

        last_message = messages[-1]
        content_text = ""

        if isinstance(last_message, Message):
            if last_message.role == "assistant" and last_message.content is not None:
                # Coerce to string if content is list parts
                if isinstance(last_message.content, str):
                    content_text = last_message.content
                else:
                    try:
                        parts: List[ChatCompletionContentPartTextParam] = last_message.content  # type: ignore[assignment]
                        content_text = "\n".join(getattr(p, "text", "") for p in parts)
                    except Exception:
                        content_text = ""
            else:
                return EvaluateResult(
                    score=0.0,
                    reason="Last message is not a valid assistant response to extract JSON from.",
                    metrics={
                        "error": MetricResult(
                            score=0.0,
                            reason="Invalid assistant message for JSON extraction.",
                            is_score_valid=False,
                        )
                    },
                )
        elif isinstance(last_message, dict):
            if last_message.get("role") == "assistant" and last_message.get("content") is not None:
                raw_content = last_message.get("content", "")
                content_text = raw_content if isinstance(raw_content, str) else ""
            else:
                return EvaluateResult(
                    score=0.0,
                    reason="Last message is not a valid assistant response (dict) to extract JSON from.",
                    metrics={
                        "error": MetricResult(
                            score=0.0,
                            reason="Invalid assistant message (dict) for JSON extraction.",
                            is_score_valid=False,
                        )
                    },
                )
        else:
            return EvaluateResult(
                score=0.0,
                reason=f"Unexpected type for last message: {type(last_message)}.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason="Invalid message type for JSON extraction.",
                        is_score_valid=False,
                    )
                },
            )

        extracted_json_str = None
        if content_text:
            try:
                pattern = r"```(?:json)?\s*([\s\S]*?)```"
                code_blocks = re.findall(pattern, content_text)
                if code_blocks:
                    extracted_json_str = code_blocks[0]
                else:
                    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content_text, re.DOTALL)
                    if json_match:
                        try:
                            json.loads(json_match.group(0))
                            extracted_json_str = json_match.group(0)
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass

        if extracted_json_str:
            json_content = extracted_json_str

        if not json_content:
            return EvaluateResult(
                score=0.0,
                reason="No JSON content found in messages.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason="No JSON content found in messages",
                        is_score_valid=False,
                    )
                },
            )

    if expected_schema is None:
        return EvaluateResult(
            score=0.0,
            reason="No expected schema provided for comparison.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="No expected schema provided",
                    is_score_valid=False,
                )
            },
        )

    expected_schema = normalize_schema(expected_schema)

    try:
        if isinstance(json_content, str):
            parsed_content = json.loads(json_content)
        else:
            parsed_content = json_content
    except json.JSONDecodeError:
        return EvaluateResult(
            score=0.0,
            reason=f"Invalid JSON content: {json_content}",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"Invalid JSON content: {json_content}",
                    is_score_valid=False,
                )
            },
        )

    # Function to recursively build a schema from content
    def build_schema_from_content(content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            schema: Dict[str, Any] = {"type": "object", "properties": {}}
            for key, value in content.items():
                if isinstance(schema["properties"], dict):  # Should always be true
                    schema["properties"][key] = build_schema_from_content(value)
            return schema
        elif isinstance(content, list):
            if content:
                return {
                    "type": "array",
                    "items": build_schema_from_content(content[0]),
                }
            return {"type": "array"}
        elif isinstance(content, str):
            return {"type": "string"}
        elif isinstance(content, bool):
            return {"type": "boolean"}
        elif isinstance(content, (int, float)):
            return {"type": "number"}
        elif content is None:
            return {"type": "null"}
        else:
            return {"type": "any"}

    content_schema = build_schema_from_content(parsed_content)
    expected_properties = extract_schema_properties(expected_schema)
    actual_properties = extract_schema_properties(content_schema)
    schema_similarity = calculate_jaccard_similarity(expected_properties, actual_properties)

    missing_props = expected_properties - actual_properties
    extra_props = actual_properties - expected_properties
    matching_props = expected_properties.intersection(actual_properties)

    comparison_details = []
    if matching_props:
        comparison_details.append(f"Matching properties ({len(matching_props)}):")
        for prop, prop_type in sorted(matching_props):
            comparison_details.append(f"  - {prop}: {prop_type}")
    if missing_props:
        comparison_details.append(f"Missing properties ({len(missing_props)}):")
        for prop, prop_type in sorted(missing_props):
            comparison_details.append(f"  - {prop}: {prop_type}")
    if extra_props:
        comparison_details.append(f"Extra properties ({len(extra_props)}):")
        for prop, prop_type in sorted(extra_props):
            comparison_details.append(f"  - {prop}: {prop_type}")

    schema_comparison_reason = "\n".join(comparison_details)

    metrics["schema_similarity"] = MetricResult(
        score=schema_similarity,
        reason=f"Schema similarity: {schema_similarity:.2f}\n{schema_comparison_reason}",
        is_score_valid=schema_similarity == 1.0,
    )

    final_score = schema_similarity
    final_reason = f"Final score based on schema similarity: {final_score:.2f}."

    return EvaluateResult(score=final_score, reason=final_reason, metrics=metrics)


def json_schema_reward_with_llm_judge(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Union[List[Message], List[Dict[str, Any]]]] = None,
    json_content: Optional[Union[Dict[str, Any], str]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    expected_behavior: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Combined reward function that evaluates JSON content using both schema
    validation and LLM judgment.

    Args:
        messages: The conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. Not directly used by this reward.
        json_content: The JSON content to evaluate (if not provided, extracts
                      from the last message).
        expected_schema: The expected schema for the JSON content.
        expected_behavior: Description of the expected behavior/content
        openai_api_key: OpenAI API key (if not provided, uses environment variable)
        model: Model to use for LLM evaluation (default: gpt-4o-mini)
        temperature: Temperature for the model generation (default: 0.0)
        weights: Dictionary of weights for each component
                (default: {"schema": 0.7, "llm": 0.3})
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with score and metrics
    """
    # Import OpenAI at call time to make this optional
    try:
        from openai import OpenAI
    except ImportError:
        return EvaluateResult(
            score=0.0,
            reason="OpenAI package not installed.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="OpenAI package not installed. Install it with: pip install openai",
                    is_score_valid=False,
                )
            },
        )

    if weights is None:
        weights = {"schema": 0.7, "llm": 0.3}

    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    schema_result = json_schema_reward(
        messages=messages,
        ground_truth=ground_truth,
        json_content=json_content,
        expected_schema=expected_schema,
        **kwargs,
    )

    llm_score = 0.0
    llm_reason = "Skipped: No expected behavior provided"
    if expected_behavior:
        if json_content is None:
            if "error" in schema_result.metrics:
                return schema_result
            last_message = messages[-1]
            assert last_message is not None, "Last message is None"
            # Support both dict-shaped messages and pydantic Message objects
            if isinstance(last_message, dict):
                content = last_message.get("content", "")
            else:
                try:
                    content = getattr(last_message, "content", "")
                except Exception:
                    content = ""
            json_str_from_msg = ""
            try:
                pattern = r"```(?:json)?\s*([\s\S]*?)```"
                code_blocks = re.findall(pattern, content)
                if code_blocks:
                    json_str_from_msg = code_blocks[0]
                else:
                    json_matches = re.findall(r"\{.*\}", content, re.DOTALL)
                    if json_matches:
                        json_str_from_msg = json_matches[0]
            except Exception:
                pass
            try:
                if json_str_from_msg:
                    json_content = json.loads(json_str_from_msg)
            except json.JSONDecodeError:
                json_content = json_str_from_msg

        if isinstance(json_content, dict):
            json_str_for_llm = json.dumps(json_content, indent=2)
        else:
            json_str_for_llm = str(json_content)

        expected_schema_str = json.dumps(expected_schema, indent=2) if expected_schema else "No schema provided"

        conversation_msg = "No conversation context provided"
        if messages:
            conversation_parts = []
            for msg in messages[:-1]:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content_part = msg.get("content", "")
                else:
                    # Fallback for Message objects
                    role = getattr(msg, "role", "")
                    content_part = getattr(msg, "content", "")
                if role and content_part:
                    conversation_parts.append(f"{role}: {content_part}")
            if conversation_parts:
                conversation_msg = "\n".join(conversation_parts)

        prompt = f"""You are evaluating the quality of JSON content provided by an AI assistant.
Your job is to assess whether the JSON structure and content is appropriate, correctly formatted,
and follows the expected schema and behavior.

CONVERSATION CONTEXT:
{conversation_msg}

JSON CONTENT:
{json_str_for_llm}

EXPECTED SCHEMA:
{expected_schema_str}

EXPECTED BEHAVIOR/CONTENT:
{expected_behavior}

Evaluate the JSON content and provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect)
2. A detailed explanation of your rating
3. Specific issues or strengths of the JSON content

Format your response as:
SCORE: [number between 0.0 and 1.0]
EXPLANATION: [your detailed explanation]
"""
        try:
            import os

            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            llm_response = response.choices[0].message.content or ""
            score_match = re.search(r"SCORE:\s*([\d.]+)", llm_response)
            explanation_match = re.search(r"EXPLANATION:\s*(.*)", llm_response, re.DOTALL)
            if score_match:
                try:
                    llm_score = float(score_match.group(1))
                    llm_score = max(0.0, min(llm_score, 1.0))
                except ValueError:
                    llm_score = 0.5
            else:
                llm_score = 0.5
            llm_reason = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        except Exception as e:
            llm_score = 0.0
            llm_reason = f"Error calling OpenAI API: {str(e)}"

    combined_metrics = {}
    for key, metric_val in schema_result.metrics.items():
        if key != "schema_similarity":
            combined_metrics[f"schema_{key}"] = metric_val
        else:
            combined_metrics[key] = metric_val

    combined_metrics["llm_judge"] = MetricResult(
        score=llm_score,
        reason=llm_reason,
        is_score_valid=llm_score >= 0.8,
    )
    combined_metrics["schema_score"] = MetricResult(
        score=schema_result.score,
        reason=f"Schema validation score: {schema_result.score:.2f}",
        is_score_valid=schema_result.score == 1.0,
    )
    combined_metrics["llm_score"] = MetricResult(
        score=llm_score,
        reason=f"LLM judge score: {llm_score:.2f}",
        is_score_valid=llm_score >= 0.8,
    )

    schema_weight = normalized_weights.get("schema", 0.7)
    llm_weight = normalized_weights.get("llm", 0.3)
    final_score = (schema_result.score * schema_weight) + (llm_score * llm_weight)
    final_reason = f"Composite score. Schema ({schema_result.score:.2f} * {schema_weight:.2f}) + LLM ({llm_score:.2f} * {llm_weight:.2f})."

    combined_metrics["weights"] = MetricResult(
        score=0.0,
        reason=f"Weights used - Schema: {schema_weight:.2f}, LLM: {llm_weight:.2f}",
        is_score_valid=True,
    )

    return EvaluateResult(score=final_score, reason=final_reason, metrics=combined_metrics)
