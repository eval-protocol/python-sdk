"""
Integration helpers between Eval Protocol evaluations and OpenAI RFT graders.

Currently provides:
- build_python_grader_from_evaluation_test: turn an evaluation-style function into
  an OpenAI Python grader spec ({"type": "python", "source": ...}).
"""

import ast
import inspect
import textwrap


def build_python_grader_from_evaluation_test(test_fn) -> dict:
    """
    Return an OpenAI Python grader spec from an Eval Protocol-style evaluation function.

    Assumptions:
    - `test_fn` is either:
        * the core evaluation function, or
        * an @evaluation_test-decorated function that carries `_origin_func`.
      Its effective signature looks like:

          def my_eval(row, **kwargs) -> EvaluateResult | float | EvaluationRow

    - The function treats `row` as an `EvaluationRow` and only relies on attributes
      we provide in the duck-typed stand-in:
        * row.ground_truth
        * row.messages
        * row.item (raw item dict)
        * row.sample (raw sample dict)

    - We map OpenAI's (sample, item) into that duck-typed `EvaluationRow` as follows:
        * item["reference_answer"]      -> row.ground_truth
        * item["messages"] (if present) -> row.messages (normalized to Message-like objects)
        * sample["output_text"]         -> appended as the last assistant message in row.messages
        * the original dicts are also available via row.item / row.sample

    - The function returns either:
        * a numeric score, or
        * an object/dict with a `score` field, or
        * an EvaluationRow/EvaluateResult-like object with `.evaluation_result.score`.
    """

    # If the user passed an @evaluation_test wrapper, try to recover the original function
    origin = getattr(test_fn, "_origin_func", test_fn)

    # Get the source of the original function
    src = inspect.getsource(origin)
    src = textwrap.dedent(src)

    # Parse into AST so we can safely strip decorators and type annotations
    tree = ast.parse(src)

    class _StripAnnotationsAndDecorators(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            # Drop all decorators (e.g., @evaluation_test)
            node.decorator_list = []
            # Remove return type annotation
            node.returns = None
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            node.decorator_list = []
            node.returns = None
            self.generic_visit(node)
            return node

        def visit_arg(self, node: ast.arg) -> ast.AST:
            # Remove all parameter annotations (e.g., row: EvaluationRow)
            node.annotation = None
            return node

    transformer = _StripAnnotationsAndDecorators()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Find the first function definition and rename it to _ep_eval
    func_node: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break

    if func_node is None:
        raise ValueError("Expected a function definition in test_fn source.")

    func_node.name = "_ep_eval"

    # Turn the modified AST back into source
    src = ast.unparse(tree)

    # Helper code that will live *inside* the grader source
    helper = """
from typing import Any, Dict
from types import SimpleNamespace


class EvaluationRow(SimpleNamespace):
    \"\"\"Minimal duck-typed stand-in for an evaluation row.

    Extend this with whatever attributes your eval logic uses.
    \"\"\"
    pass


class EvaluateResult(SimpleNamespace):
    \"\"\"Simple stand-in for Eval Protocol's EvaluateResult.

    This lets evaluation-style functions that construct EvaluateResult(score=...)
    run inside the Python grader sandbox without importing eval_protocol.
    \"\"\"

    def __init__(self, score: float, **kwargs: Any) -> None:
        super().__init__(score=score, **kwargs)


class Message(SimpleNamespace):
    \"\"\"Duck-typed stand-in for eval_protocol.models.Message (role/content).\"\"\"
    pass


def _build_row(sample: Dict[str, Any], item: Dict[str, Any]) -> EvaluationRow:
    # Start from any item-provided messages (EP-style), defaulting to [].
    raw_messages = item.get("messages") or []
    normalized_messages = []
    for m in raw_messages:
        if isinstance(m, dict):
            normalized_messages.append(
                Message(
                    role=m.get("role"),
                    content=m.get("content"),
                )
            )
        else:
            # Already Message-like; rely on duck typing (must have role/content)
            normalized_messages.append(m)

    reference = item.get("reference_answer")
    prediction = sample.get("output_text")

    # EP-style: ensure the model prediction is present as the last assistant message
    if prediction is not None:
        normalized_messages = list(normalized_messages)  # shallow copy
        normalized_messages.append(Message(role="assistant", content=prediction))

    return EvaluationRow(
        ground_truth=reference,
        messages=normalized_messages,
        item=item,
        sample=sample,
    )


def grade(sample: Dict[str, Any], item: Dict[str, Any]) -> float:
    row = _build_row(sample, item)
    result = _ep_eval(row=row)

    # Try to normalize different result shapes into a float score
    try:
        from collections.abc import Mapping

        if isinstance(result, (int, float)):
            return float(result)

        # EvaluateResult-like object with .score
        if hasattr(result, "score"):
            return float(result.score)

        # EvaluationRow-like object with .evaluation_result.score
        eval_res = getattr(result, "evaluation_result", None)
        if eval_res is not None:
            if isinstance(eval_res, Mapping):
                if "score" in eval_res:
                    return float(eval_res["score"])
            elif hasattr(eval_res, "score"):
                return float(eval_res.score)

        # Dict-like with score
        if isinstance(result, Mapping) and "score" in result:
            return float(result["score"])
    except Exception:
        pass

    return 0.0
"""

    full_source = src + "\n\n" + textwrap.dedent(helper)
    return {"type": "python", "source": full_source}
