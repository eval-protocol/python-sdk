from inspect import Signature

from eval_protocol.models import CompletionParams, EvaluationRow
from eval_protocol.pytest.types import EvaluationTestMode


def validate_signature(
    signature: Signature, mode: EvaluationTestMode, completion_params: list[CompletionParams | None] | None
) -> None:
    # For pointwise/groupwise mode, we expect a different signature
    # we expect single row to be passed in as the original row
    if mode == "pointwise":
        # Pointwise mode: function should accept messages and other row-level params
        if "row" not in signature.parameters:
            raise ValueError("In pointwise mode, your eval function must have a parameter named 'row'")

        # validate that "Row" is of type EvaluationRow
        if signature.parameters["row"].annotation is not EvaluationRow:  # pyright: ignore[reportAny]
            raise ValueError("In pointwise mode, the 'row' parameter must be of type EvaluationRow")

        # validate that the function has a return type of EvaluationRow
        if signature.return_annotation is not EvaluationRow:  # pyright: ignore[reportAny]
            raise ValueError("In pointwise mode, your eval function must return an EvaluationRow instance")

        # additional check for groupwise evaluation
    elif mode == "groupwise":
        if "rows" not in signature.parameters:
            raise ValueError("In groupwise mode, your eval function must have a parameter named 'rows'")

        # validate that "Rows" is of type List[EvaluationRow]
        if signature.parameters["rows"].annotation is not list[EvaluationRow]:  # pyright: ignore[reportAny]
            raise ValueError("In groupwise mode, the 'rows' parameter must be of type List[EvaluationRow")

        # validate that the function has a return type of List[EvaluationRow]
        if signature.return_annotation is not list[EvaluationRow]:  # pyright: ignore[reportAny]
            raise ValueError("In groupwise mode, your eval function must return a list of EvaluationRow instances")
        if completion_params is not None and len(completion_params) < 2:
            raise ValueError("In groupwise mode, you must provide at least 2 completion parameters")
    else:
        # all mode: function should accept input_dataset and model
        if "rows" not in signature.parameters:
            raise ValueError("In all mode, your eval function must have a parameter named 'rows'")

        # validate that "Rows" is of type List[EvaluationRow]
        if signature.parameters["rows"].annotation is not list[EvaluationRow]:  # pyright: ignore[reportAny]
            raise ValueError("In all mode, the 'rows' parameter must be of type List[EvaluationRow")

        # validate that the function has a return type of List[EvaluationRow]
        if signature.return_annotation is not list[EvaluationRow]:  # pyright: ignore[reportAny]
            raise ValueError("In all mode, your eval function must return a list of EvaluationRow instances")
