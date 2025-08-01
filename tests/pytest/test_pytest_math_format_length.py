from eval_protocol.pytest import default_single_turn_rollout_processor, evaluate, evaluation_test
from examples.math_with_format_and_length.main import evaluate as math_fl_evaluate
from tests.pytest.helper.gsm8k_to_evaluation_row import gsm8k_to_evaluation_row


@evaluation_test(
    input_dataset=["development/gsm8k_sample.jsonl"],
    dataset_adapter=gsm8k_to_evaluation_row,
    model=["accounts/fireworks/models/kimi-k2-instruct"],
    input_params=[{"temperature": 0.0}],
    max_dataset_rows=5,
    threshold_of_success=0.0,
    rollout_processor=default_single_turn_rollout_processor,
)
def test_math_format_length_dataset(input_dataset, input_params, model):
    """Run math with format and length evaluation on sample dataset."""
    return evaluate(input_dataset, math_fl_evaluate)
