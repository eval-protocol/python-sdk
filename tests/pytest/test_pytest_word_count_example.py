from typing import List
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test
from eval_protocol.models import EvaluateResult, MetricResult, EvaluationRow
from tests.pytest.helper.word_count_to_evaluation_row import word_count_to_evaluation_row
from haikus import haikus


@evaluation_test(
    input_dataset=["development/gsm8k_sample.jsonl"],
    dataset_adapter=word_count_to_evaluation_row,
    model=["accounts/fireworks/models/kimi-k2-instruct"],
    input_params=[{"temperature": 0.0}],
    max_dataset_rows=5,
    threshold_of_success=0.3,  # Reasonable threshold for word count evaluation
    rollout_processor=default_single_turn_rollout_processor,
    mode="pointwise",  # Use pointwise mode for elegant row-by-row evaluation
)
def test_word_count_evaluate(row: EvaluationRow) -> EvaluationRow:
    """
    Pointwise word count evaluator - just the core evaluation logic.
    Everything else (models, datasets, thresholds) is parameterized in the decorator.
    """
    if not row.messages:
        return EvaluateResult(score=0.0, reason="No messages found", is_score_valid=False)

    last_message = row.messages[-1]
    content = last_message.content if last_message and last_message.content else ""

    # Word count logic
    word_count = len(content.split())
    word_count_score = min(word_count / 100, 1.0)

    # Haiku analysis logic
    haiku_lines = content.splitlines()
    haiku_analysis_data = {}
    haiku_metric_score = 0.0
    haiku_metric_reason = "Content not suitable for haiku analysis."
    haiku_metric_valid = False

    if len(haiku_lines) in [3, 5]:
        try:
            analysis = haikus(haiku_lines)
            haiku_analysis_data = analysis
            kigo = analysis.get("kigo", [])
            haiku_type = analysis.get("type", "unknown")

            if kigo:
                haiku_metric_score = 1.0
            elif haiku_type not in ["unknown", "error"]:
                haiku_metric_score = 0.5

            haiku_metric_reason = f"Haiku analysis - Type: {haiku_type}, Kigo: {', '.join(kigo) if kigo else 'None'}"
            haiku_metric_valid = True
        except Exception as e:
            haiku_metric_reason = f"Haiku analysis failed: {str(e)}"
            haiku_metric_valid = False

    # Combine metrics
    metrics = {
        "word_count": MetricResult(
            score=word_count_score,
            is_score_valid=word_count > 0,
            reason=f"Word count: {word_count}",
        ),
        "haiku_analysis": MetricResult(
            score=haiku_metric_score,
            is_score_valid=haiku_metric_valid,
            reason=haiku_metric_reason,
            data=haiku_analysis_data,
        ),
    }

    return EvaluateResult(
        score=word_count_score,
        reason=f"Word count: {word_count}. {haiku_metric_reason}",
        metrics=metrics,
    )
