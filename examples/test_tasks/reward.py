"""
Reward function for the test tasks.
"""

from eval_protocol import reward_function
from eval_protocol.models import EvaluateResult, MetricResult


@reward_function
def evaluate(messages, *, state, ground_truth_comparable_state, **kwargs):
    """
    Evaluate the state against the ground truth.

    Args:
        messages: The conversation history
        state: The final state including the resource
        ground_truth_comparable_state: The expected final state

    Returns:
        EvaluateResult with the evaluation result
    """
    resource = state.get("resource")
    if not resource:
        return EvaluateResult(score=0.0, reason="Resource not available", metrics={})

    # Get the current state from the resource
    current_state = resource.get_state()

    # Compare with ground truth
    matches = True
    mismatches = []

    # Check each expected key in the ground truth
    for key, expected_value in ground_truth_comparable_state.items():
        if key not in current_state:
            matches = False
            mismatches.append(f"Missing key: {key}")
            continue

        actual_value = current_state[key]

        # Special handling for lists (order doesn't matter)
        if isinstance(expected_value, list) and isinstance(actual_value, list):
            expected_set = set(expected_value)
            actual_set = set(actual_value)
            if expected_set != actual_set:
                matches = False
                missing = expected_set - actual_set
                extra = actual_set - expected_set
                if missing:
                    mismatches.append(f"Missing items in {key}: {missing}")
                if extra:
                    mismatches.append(f"Extra items in {key}: {extra}")
        elif expected_value != actual_value:
            matches = False
            mismatches.append(f"Value mismatch for {key}: expected {expected_value}, got {actual_value}")

    # Check for extra keys in the current state (not required to match)

    # Construct the reward output
    score = 1.0 if matches else 0.0
    reason = "State matches ground truth" if matches else f"State differs from ground truth: {'; '.join(mismatches)}"

    metrics = {"state_match": MetricResult(score=float(matches), reason=reason)}

    return EvaluateResult(score=score, reason=reason, metrics=metrics)
