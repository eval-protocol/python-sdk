"""
LLM Judge Example

This example shows how to create a custom reward function that uses an LLM
to judge the quality of AI responses. It demonstrates using the Fireworks Build SDK
to call an LLM for evaluation purposes.
"""

import os
from typing import Any, Dict, List, Optional, Union

# Import the Fireworks Build SDK
from fireworks import LLM

from eval_protocol import (
    EvaluateResult,
    MetricResult,
    create_llm_resource,
    reward_function,
)
from eval_protocol.models import Message

# Initialize the judge LLM with on-demand deployment
judge_llm = LLM(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    deployment_type="on-demand",
)

# Create resource wrapper for the decorator
judge_llm_resource = create_llm_resource(judge_llm)


@reward_function(resources={"llms": [judge_llm_resource]})
def evaluate(
    messages: Union[List[Message], List[Dict[str, Any]]],
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate AI responses using an LLM judge.

    This function demonstrates how to use an external LLM to evaluate
    the quality of AI responses in a conversation.

    Args:
        messages: The conversation messages to evaluate
        **kwargs: Additional parameters including resources

    Returns:
        EvaluateResult with LLM judge score and reasoning
    """
    # Get the assistant's response (last message)
    assistant_message = messages[-1]
    if isinstance(assistant_message, dict):
        assistant_response = assistant_message.get("content", "")
    else:
        assistant_response = assistant_message.content or ""

    # Get the user query (second to last message, assuming user/assistant alternation)
    if len(messages) >= 2:
        user_message = messages[-2]
        if isinstance(user_message, dict):
            user_query = user_message.get("content", "")
        else:
            user_query = user_message.content or ""
    else:
        user_query = "No user query found"

    # Create evaluation prompt
    evaluation_prompt = f"""
Please evaluate the quality of the AI assistant's response to the user's query.

User Query: {user_query}

AI Response: {assistant_response}

Please rate the response on a scale of 0.0 to 1.0 based on:
- Helpfulness and relevance to the query
- Accuracy of information (if applicable)
- Clarity and coherence
- Completeness of the answer

Provide your rating as a decimal number (e.g., 0.8) followed by a brief explanation.
Format: SCORE: X.X
REASON: Your explanation here
"""

    try:
        # Get the LLM judge from resources
        judge_llm = kwargs["resources"]["llms"][0]

        # Call the LLM judge
        response = judge_llm.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator of AI responses. Provide accurate and fair evaluations.",
                },
                {"role": "user", "content": evaluation_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        judge_response = response.choices[0].message.content

        # Parse the LLM response to extract score and reason
        score = 0.5  # Default score
        reason = "Could not parse LLM judge response"

        lines = judge_response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_str = line.replace("SCORE:", "").strip()
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        # If we couldn't parse properly, try to extract from the full response
        if reason == "Could not parse LLM judge response":
            reason = f"LLM Judge: {judge_response[:200]}..."

    except Exception as e:
        # Fallback scoring based on simple heuristics
        score = 0.5
        if len(assistant_response) > 10:
            score += 0.2
        if len(assistant_response) > 50:
            score += 0.1
        if "?" in user_query and len(assistant_response) > 20:
            score += 0.1

        reason = f"LLM judge failed ({str(e)}), used heuristic scoring"

    # Create metrics for detailed analysis
    metrics = {
        "response_length": MetricResult(
            score=min(1.0, len(assistant_response) / 100.0),  # Normalize length
            reason=f"Response length: {len(assistant_response)} characters",
            is_score_valid=True,
        ),
        "has_content": MetricResult(
            score=1.0 if assistant_response.strip() else 0.0,
            reason=("Response is not empty" if assistant_response.strip() else "Response is empty"),
            is_score_valid=True,
        ),
    }

    return EvaluateResult(
        score=score,
        reason=reason,
        metrics=metrics,
    )
