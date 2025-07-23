from typing import Any, Dict, List  # Added Dict, Any for clarity

from haikus import haikus

from eval_protocol import EvaluateResult, Message, MetricResult, reward_function


# https://pypi.org/project/haikus/
@reward_function(
    id="word_count", requirements=["haikus==0.3.8", "dummy-pip-package==1.0.0"]
)
def evaluate(messages: List[Message], **kwargs) -> EvaluateResult:
    """
    Evaluate a sample entry by counting words in the response and analyzing for haikus.

    Args:
        messages: List of conversation messages, where the last message is the
                  assistant's response to be evaluated.
        **kwargs: Additional parameters

    Returns:
        EvaluateResult with score and metrics information
    """
    if not messages:
        return EvaluateResult(
            score=0.0, reason="No messages found", is_score_valid=False
        )

    last_message = messages[-1]
    content = last_message.content if last_message and last_message.content else ""

    # Original word count logic
    word_count = len(content.split())
    word_count_score = min(word_count / 100, 1.0)  # Cap at 1.0

    # Haiku analysis logic
    haiku_lines = content.splitlines()
    haiku_analysis_data: Dict[str, Any] = {}  # To store raw haiku lib output
    haiku_metric_score = 0.0
    haiku_metric_reason = (
        "Content not suitable for haiku analysis (e.g., not 3 or 5 lines)."
    )
    haiku_metric_valid = False

    if len(haiku_lines) in [3, 5]:  # Haikus are typically 3 or 5 phrases/lines
        try:
            analysis = haikus(haiku_lines)  # Call the haikus library
            haiku_analysis_data = analysis  # Store the raw analysis

            kigo = analysis.get("kigo", [])
            haiku_type = analysis.get("type", "unknown")

            # Basic scoring: 1.0 if kigo found, 0.5 if parsed as a known type, 0.0 otherwise
            if kigo:
                haiku_metric_score = 1.0
            elif haiku_type not in [
                "unknown",
                "error",
            ]:  # Assuming 'error' might be a type
                haiku_metric_score = 0.5

            haiku_metric_reason = f"Haiku analysis - Type: {haiku_type}, Kigo: {', '.join(kigo) if kigo else 'None'}"
            haiku_metric_valid = True
        except Exception as e:
            # This might happen if haikus() lib encounters an issue with unexpected input
            haiku_metric_reason = f"Haiku analysis failed: {str(e)}"
            haiku_metric_valid = False  # Or True, if we consider an attempt was made

    # Combine metrics
    metrics = {
        "word_count": MetricResult(
            score=word_count_score,  # Using the word_count_score calculated earlier
            is_score_valid=word_count > 0,
            reason=f"Word count: {word_count}",
        ),
        "haiku_analysis": MetricResult(
            score=haiku_metric_score,
            is_score_valid=haiku_metric_valid,
            reason=haiku_metric_reason,
            data=haiku_analysis_data,  # Optionally include raw data
        ),
    }

    # The overall score could be a combination, or just the word_count score
    # For now, let's keep the original score logic based on word count
    final_score = word_count_score

    return EvaluateResult(
        score=final_score,
        reason=f"Word count: {word_count}. {haiku_metric_reason}",  # Updated reason
        is_score_valid=True,  # Assuming overall evaluation is valid if processed
        metrics=metrics,
    )
