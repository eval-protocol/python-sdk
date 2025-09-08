from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluateResult, MetricResult
from eval_protocol.typed_interface import reward_function

__all__ = ["adapt_metric"]

try:
    from deepeval.metrics.base_metric import BaseConversationalMetric, BaseMetric
    from deepeval.test_case import ConversationalTestCase, LLMTestCase
except Exception:  # pragma: no cover - deepeval is optional
    BaseMetric = None
    BaseConversationalMetric = None
    LLMTestCase = None
    ConversationalTestCase = None


def _metric_name(metric: Any) -> str:
    name = getattr(metric, "__name__", None)
    if name and name not in {
        "Base Metric",
        "Base Conversational Metric",
        "Base Multimodal Metric",
    }:
        return str(name)
    name = getattr(metric, "name", None)
    if name:
        return str(name)
    return metric.__class__.__name__


def adapt_metric(metric: Any):
    """Adapt a deepeval metric object into an Eval Protocol reward function."""

    @reward_function
    def wrapped(
        messages: List[Dict[str, Any]],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluateResult:
        if BaseMetric is None or LLMTestCase is None:
            raise ImportError("deepeval must be installed to use this integration")
        if not messages:
            return EvaluateResult(score=0.0, reason="No messages", metrics={})

        output = messages[-1].get("content", "")
        input_msg = ""
        if len(messages) >= 2:
            input_msg = messages[-2].get("content", "")

        def _build_case_kwargs() -> Dict[str, Any]:
            case_kwargs: Dict[str, Any] = {}
            params = getattr(metric, "evaluation_params", None)
            if params:
                for param in params:
                    if param.value == "input":
                        case_kwargs["input"] = input_msg
                    elif param.value == "actual_output":
                        case_kwargs["actual_output"] = output
                    elif param.value == "expected_output":
                        case_kwargs["expected_output"] = ground_truth
                    elif param.value == "context":
                        case_kwargs["context"] = kwargs.get("context")
                    elif param.value == "retrieval_context":
                        case_kwargs["retrieval_context"] = kwargs.get("retrieval_context")
                    elif param.value == "tools_called":
                        case_kwargs["tools_called"] = kwargs.get("tools_called")
                    elif param.value == "expected_tools":
                        case_kwargs["expected_tools"] = kwargs.get("expected_tools")
            else:
                case_kwargs = {
                    "input": input_msg,
                    "actual_output": output,
                    "expected_output": ground_truth,
                }
            if "input" not in case_kwargs:
                case_kwargs["input"] = input_msg
            if "actual_output" not in case_kwargs:
                case_kwargs["actual_output"] = output
            return case_kwargs

        if BaseConversationalMetric is not None and isinstance(metric, BaseConversationalMetric):
            # Narrow types for optional imports to satisfy the type checker
            assert LLMTestCase is not None
            assert ConversationalTestCase is not None
            turns = []
            for i, msg in enumerate(messages):
                turn_input = messages[i - 1].get("content", "") if i > 0 else ""
                output_turn = msg.get("content", "")
                input_msg_backup = input_msg
                input_msg = turn_input
                output = output_turn
                turn_kwargs = _build_case_kwargs()
                turns.append(LLMTestCase(**turn_kwargs))
                input_msg = input_msg_backup
                output = messages[-1].get("content", "")
            test_case = ConversationalTestCase(turns=turns)
        else:
            # Narrow types for optional imports to satisfy the type checker
            assert LLMTestCase is not None
            case_kwargs = _build_case_kwargs()
            test_case = LLMTestCase(**case_kwargs)

        # Guard against metric.measure being None or non-callable
        measure_fn = getattr(metric, "measure", None)
        if not callable(measure_fn):
            raise TypeError("Provided metric does not have a callable 'measure' method")
        measure_fn(test_case, **kwargs)
        score = float(metric.score or 0.0)
        reason = getattr(metric, "reason", None)
        name = _metric_name(metric)
        metrics = {name: MetricResult(score=score, reason=reason or "", is_score_valid=True)}
        return EvaluateResult(score=score, reason=reason, metrics=metrics)

    return wrapped
