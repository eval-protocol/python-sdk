import types
from typing import Any, Dict, Callable

from eval_protocol.integrations.openai_rft import build_python_grader_from_evaluation_test
from eval_protocol.models import EvaluationRow


def _exec_and_get_grade(source: str) -> Callable[[Dict[str, Any], Dict[str, Any]], float]:
    """Execute generated grader source and return the grade(sample, item) function."""
    ns: Dict[str, Any] = {}
    exec(source, ns, ns)
    grade_obj = ns.get("grade")
    assert isinstance(grade_obj, types.FunctionType)
    return grade_obj


def test_build_python_grader_from_plain_eval_function():
    """Plain eval-style function should be converted into a working grade(sample, item)."""

    # Simulate an eval-style function with annotations
    def my_eval(row: EvaluationRow, **kwargs: Any) -> float:
        # Simple correctness check: 1.0 if ground_truth matches sample["output_text"], else 0.0
        ground_truth = getattr(row, "ground_truth", None)
        sample = getattr(row, "sample", None) or {}
        pred = sample.get("output_text")
        return 1.0 if ground_truth == pred else 0.0

    grader_spec = build_python_grader_from_evaluation_test(my_eval)
    assert grader_spec["type"] == "python"
    source = grader_spec["source"]

    # Basic structural sanity checks on the generated source
    assert '"EvaluationRow"' not in source
    assert "def _ep_eval" in source
    assert "def my_eval" not in source
    assert "@evaluation_test" not in source

    grade = _exec_and_get_grade(source)

    sample = {"output_text": "42"}
    item = {"reference_answer": "42"}
    score = grade(sample, item)
    assert isinstance(score, float)
    assert score == 1.0


def test_build_python_grader_from_wrapped_evaluation_test():
    """When the function is wrapped and carries _origin_func, we should use the origin."""

    def original_eval(row, **kwargs: Any) -> float:
        return 0.5

    def wrapper(*args: Any, **kwargs: Any) -> float:
        return original_eval(*args, **kwargs)

    # Simulate @evaluation_test attaching _origin_func
    setattr(wrapper, "_origin_func", original_eval)

    grader_spec = build_python_grader_from_evaluation_test(wrapper)
    assert grader_spec["type"] == "python"
    source = grader_spec["source"]

    grade = _exec_and_get_grade(source)
    score = grade({"output_text": "anything"}, {"reference_answer": "anything"})
    assert isinstance(score, float)
    assert score == 0.5
