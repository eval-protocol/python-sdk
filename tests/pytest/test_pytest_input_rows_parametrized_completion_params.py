from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test


@evaluation_test(
    completion_params=[{"model": "gpt-4"}, {"model": "gpt-4o"}],
    input_rows=[[EvaluationRow(messages=[Message(role="user", content="Hello, how are you?")])]],
    evaluation_test_kwargs=[{"seen_models": set()}],
)
def test_pytest_input_rows_parametrized_completion_params(row: EvaluationRow, **kwargs) -> EvaluationRow:
    """Tests that parametrized completion params are working correctly for input_rows"""
    seen_models = kwargs["seen_models"]
    model = row.input_metadata.completion_params["model"]
    if len(seen_models) == 1:
        # assert that the other model was seen
        if model == "gpt-4":
            assert "gpt-4o" in seen_models
        else:
            assert "gpt-4" in seen_models
    seen_models.add(model)
    return row
