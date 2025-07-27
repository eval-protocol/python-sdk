from typing import Any, Callable, Dict, List, Union

from eval_protocol.models import EvaluateResult, MetricResult
from eval_protocol.typed_interface import reward_function

__all__ = ["adapt"]


def _convert_result(res: Dict[str, Any]) -> EvaluateResult:
    score = float(res.get("score", 0.0))
    reason = res.get("comment")
    key = res.get("key", "openeval")
    metrics = {key: MetricResult(score=score, reason=reason or "", is_score_valid=True)}
    return EvaluateResult(score=score, reason=reason, metrics=metrics)


def adapt(openeval_fn: Callable[..., Union[Dict[str, Any], List[Dict[str, Any]]]]):
    """Adapt an OpenEvals evaluator into an Eval Protocol reward function."""

    @reward_function
    def wrapped(
        messages: List[Dict[str, str]],
        ground_truth: Union[str, List[Dict[str, str]], None] = None,
        **kwargs: Any,
    ) -> EvaluateResult:
        if not messages:
            return EvaluateResult(score=0.0, reason="No messages", metrics={})
        output = messages[-1].get("content", "")
        reference = None
        if isinstance(ground_truth, list):
            if ground_truth:
                reference = ground_truth[-1].get("content")
        else:
            reference = ground_truth
        res = openeval_fn(outputs=output, reference_outputs=reference, **kwargs)
        if isinstance(res, list):
            res = res[0]
        return _convert_result(res)

    return wrapped
