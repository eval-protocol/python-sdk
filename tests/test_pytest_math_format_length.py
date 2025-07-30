from eval_protocol.pytest_utils import evaluate, evaluation_test
from examples.math_with_format_and_length.main import evaluate as math_fl_evaluate


@evaluation_test(
    input_dataset=["development/gsm8k_sample.jsonl"],
    model=["accounts/fireworks/models/kimi-k2-instruct"],
    input_params=[{"temperature": 0.0}],
    max_dataset_rows=5,
    threshold_of_success=0.0,
)
def test_math_format_length_dataset(evaluation_rows):
    """Run math with format and length evaluation on sample dataset."""
    return evaluate(evaluation_rows, math_fl_evaluate)
