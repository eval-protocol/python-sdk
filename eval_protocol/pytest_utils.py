from typing import Any, Callable, Dict, List, Optional

import pytest
from openai import OpenAI

from .auth import get_fireworks_api_base, get_fireworks_api_key
from .common_utils import load_jsonl
from .models import EvaluateResult, EvaluationRow, Message


def default_rollout_processor(row: EvaluationRow, model: str, input_params: Dict[str, Any]) -> List[EvaluationRow]:
    """Generate a single response from a Fireworks model."""

    api_key = get_fireworks_api_key()
    api_base = get_fireworks_api_base()
    client = OpenAI(api_key=api_key, base_url=f"{api_base}/inference/v1")

    messages_payload = [{"role": m.role, "content": m.content} for m in row.messages]

    response = client.chat.completions.create(model=model, messages=messages_payload, **input_params)
    assistant_content = response.choices[0].message.content or ""
    messages = list(row.messages) + [Message(role="assistant", content=assistant_content)]
    processed = EvaluationRow(
        messages=messages,
        ground_truth=row.ground_truth,
        input_metadata={"model": model, "params": input_params},
    )
    return [processed]


def evaluate(
    rows: List[EvaluationRow], reward_fn: Callable[..., EvaluateResult], **kwargs: Any
) -> List[EvaluationRow]:
    """Apply a reward function to each row and attach the result."""
    evaluated: List[EvaluationRow] = []
    for row in rows:
        result = reward_fn(messages=row.messages, ground_truth=row.ground_truth, **kwargs)
        row.evaluation_result = result
        evaluated.append(row)
    return evaluated


def _aggregate(scores: List[float], method: str) -> float:
    if not scores:
        return 0.0
    if method == "mean":
        return sum(scores) / len(scores)
    if method == "max":
        return max(scores)
    if method == "min":
        return min(scores)
    raise ValueError(f"Unknown aggregation method: {method}")


def evaluation_test(
    *,
    input_dataset: List[str],
    model: List[str],
    input_params: List[Dict[str, Any]],
    rollout_processor: Callable[[EvaluationRow, str, Dict[str, Any]], List[EvaluationRow]] = default_rollout_processor,
    aggregation_method: str = "mean",
    threshold_of_success: Optional[float] = None,
    num_runs: int = 1,
    max_dataset_rows: Optional[int] = None,
) -> Callable[[Callable[[List[EvaluationRow]], List[EvaluationRow]]], Callable[..., None]]:
    """Decorator to create pytest-based evaluation tests.

    Args:
        input_dataset: Paths to JSONL datasets.
        model: Model identifiers to query.
        input_params: Generation parameters for the model.
        rollout_processor: Function used to perform the rollout.
        aggregation_method: How to aggregate scores across rows.
        threshold_of_success: If set, fail the test if the aggregated score is
            below this threshold.
        num_runs: Number of times to repeat the evaluation.
        max_dataset_rows: Limit dataset to the first N rows.
    """

    def decorator(test_func: Callable[[List[EvaluationRow]], List[EvaluationRow]]):
        params = []
        for ds in input_dataset:
            for m in model:
                for ip in input_params:
                    params.append((ds, m, ip))

        @pytest.mark.parametrize("dataset_path,model_name,in_params", params)
        def wrapper(dataset_path: str, model_name: str, in_params: Dict[str, Any]):
            data = load_jsonl(dataset_path)
            if max_dataset_rows is not None:
                data = data[:max_dataset_rows]
            rows: List[EvaluationRow] = []
            for entry in data:
                user_query = entry.get("user_query") or entry.get("prompt")
                if not user_query:
                    continue
                messages = [Message(role="user", content=user_query)]
                row = EvaluationRow(messages=messages, ground_truth=entry.get("ground_truth_for_eval"))
                processed = rollout_processor(row, model_name, in_params)
                rows.extend(processed)

            all_results: List[EvaluationRow] = []
            for _ in range(num_runs):
                # Each run reuses the same processed rows
                results = test_func(list(rows))
                all_results.extend(results)

            scores = [r.evaluation_result.score for r in all_results if r.evaluation_result]
            agg_score = _aggregate(scores, aggregation_method)
            if threshold_of_success is not None:
                assert (
                    agg_score >= threshold_of_success
                ), f"Aggregated score {agg_score:.3f} below threshold {threshold_of_success}"

        return wrapper

    return decorator
