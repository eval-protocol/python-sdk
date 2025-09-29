"""
Test cases for pytest.mark.parametrize validation functionality.
"""

import ast
import pytest
from eval_protocol.pytest.parameterize import _is_pytest_parametrize_with_completion_params


def create_parametrize_decorator(argnames, argvalues, use_keyword=False):
    """Create a pytest.mark.parametrize decorator AST node."""
    pytest_name = ast.Name(id="pytest", ctx=ast.Load())
    mark_attr = ast.Attribute(value=pytest_name, attr="mark", ctx=ast.Load())
    parametrize_attr = ast.Attribute(value=mark_attr, attr="parametrize", ctx=ast.Load())

    if use_keyword:
        call = ast.Call(
            func=parametrize_attr,
            args=[],
            keywords=[
                ast.keyword(arg="argnames", value=ast.Constant(value=argnames)),
                ast.keyword(arg="argvalues", value=argvalues),
            ],
        )
    else:
        call = ast.Call(func=parametrize_attr, args=[ast.Constant(value=argnames), argvalues], keywords=[])

    return call


class TestParametrizeValidation:
    """Test cases for pytest.mark.parametrize validation."""

    def test_invalid_dict_argvalues_positional(self):
        """Test that a dict as positional argvalues throws an error."""
        decorator = create_parametrize_decorator(
            "completion_params", ast.Dict(keys=[ast.Constant(value="model")], values=[ast.Constant(value="gpt-4")])
        )

        with pytest.raises(ValueError) as exc_info:
            _is_pytest_parametrize_with_completion_params(decorator)

        error_msg = str(exc_info.value)
        assert (
            "For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict"
            in error_msg
        )
        assert "Use [{'model': 'gpt-4'}] instead of {'model': 'gpt-4'}" in error_msg

    def test_valid_list_argvalues_positional(self):
        """Test that a list as positional argvalues works correctly."""
        dict_value = ast.Dict(keys=[ast.Constant(value="model")], values=[ast.Constant(value="gpt-4")])
        list_value = ast.List(elts=[dict_value], ctx=ast.Load())

        decorator = create_parametrize_decorator("completion_params", list_value)

        result = _is_pytest_parametrize_with_completion_params(decorator)
        assert result is True

    def test_invalid_dict_argvalues_keyword(self):
        """Test that a dict as keyword argvalues throws an error."""
        decorator = create_parametrize_decorator(
            "completion_params",
            ast.Dict(keys=[ast.Constant(value="model")], values=[ast.Constant(value="gpt-4")]),
            use_keyword=True,
        )

        with pytest.raises(ValueError) as exc_info:
            _is_pytest_parametrize_with_completion_params(decorator)

        error_msg = str(exc_info.value)
        assert (
            "For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict"
            in error_msg
        )
        assert "Use [{'model': 'gpt-4'}] instead of {'model': 'gpt-4'}" in error_msg

    def test_valid_list_argvalues_keyword(self):
        """Test that a list as keyword argvalues works correctly."""
        dict_value = ast.Dict(keys=[ast.Constant(value="model")], values=[ast.Constant(value="gpt-4")])
        list_value = ast.List(elts=[dict_value], ctx=ast.Load())

        decorator = create_parametrize_decorator("completion_params", list_value, use_keyword=True)

        result = _is_pytest_parametrize_with_completion_params(decorator)
        assert result is True

    def test_dynamic_error_simple_dict(self):
        """Test dynamic error message with a simple dict."""
        decorator = create_parametrize_decorator(
            "completion_params", ast.Dict(keys=[ast.Constant(value="model")], values=[ast.Constant(value="gpt-4")])
        )

        with pytest.raises(ValueError) as exc_info:
            _is_pytest_parametrize_with_completion_params(decorator)

        error_msg = str(exc_info.value)
        assert (
            "For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict"
            in error_msg
        )
        assert "Use [{'model': 'gpt-4'}] instead of {'model': 'gpt-4'}" in error_msg

    def test_dynamic_error_complex_dict(self):
        """Test dynamic error message with a complex dict."""
        decorator = create_parametrize_decorator(
            "completion_params",
            ast.Dict(
                keys=[
                    ast.Constant(value="model"),
                    ast.Constant(value="temperature"),
                    ast.Constant(value="max_tokens"),
                ],
                values=[
                    ast.Constant(value="accounts/fireworks/models/gpt-oss-120b"),
                    ast.Constant(value=0.7),
                    ast.Constant(value=1000),
                ],
            ),
        )

        with pytest.raises(ValueError) as exc_info:
            _is_pytest_parametrize_with_completion_params(decorator)

        error_msg = str(exc_info.value)
        assert (
            "For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict"
            in error_msg
        )
        assert "Use [{" in error_msg
        assert "}] instead of {" in error_msg
        assert "gpt-oss-120b" in error_msg
        assert "0.7" in error_msg
        assert "1000" in error_msg

    def test_dynamic_error_nested_dict(self):
        """Test dynamic error message with nested structures."""
        # Create a dict with nested dict
        nested_dict = ast.Dict(
            keys=[ast.Constant(value="config")],
            values=[
                ast.Dict(
                    keys=[ast.Constant(value="model"), ast.Constant(value="api_key")],
                    values=[ast.Constant(value="gpt-4"), ast.Constant(value="sk-123")],
                )
            ],
        )

        decorator = create_parametrize_decorator("completion_params", nested_dict)

        with pytest.raises(ValueError) as exc_info:
            _is_pytest_parametrize_with_completion_params(decorator)

        error_msg = str(exc_info.value)
        assert (
            "For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict"
            in error_msg
        )
        assert "Use [{" in error_msg
        assert "}] instead of {" in error_msg

    def test_dynamic_error_boolean_values(self):
        """Test dynamic error message with boolean values."""
        decorator = create_parametrize_decorator(
            "completion_params",
            ast.Dict(
                keys=[ast.Constant(value="stream"), ast.Constant(value="echo")],
                values=[ast.Constant(value=True), ast.Constant(value=False)],
            ),
        )

        with pytest.raises(ValueError) as exc_info:
            _is_pytest_parametrize_with_completion_params(decorator)

        error_msg = str(exc_info.value)
        assert (
            "For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict"
            in error_msg
        )
        assert "True" in error_msg
        assert "False" in error_msg

    def test_dynamic_error_empty_dict(self):
        """Test dynamic error message with empty dict."""
        decorator = create_parametrize_decorator("completion_params", ast.Dict(keys=[], values=[]))

        with pytest.raises(ValueError) as exc_info:
            _is_pytest_parametrize_with_completion_params(decorator)

        error_msg = str(exc_info.value)
        assert (
            "For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict"
            in error_msg
        )
        assert "Use [{}] instead of {}" in error_msg

    def test_valid_tuple_argvalues(self):
        """Test that a tuple as argvalues works correctly."""
        dict_value = ast.Dict(keys=[ast.Constant(value="model")], values=[ast.Constant(value="gpt-4")])
        tuple_value = ast.Tuple(elts=[dict_value], ctx=ast.Load())

        decorator = create_parametrize_decorator("completion_params", tuple_value)

        result = _is_pytest_parametrize_with_completion_params(decorator)
        assert result is True

    def test_non_parametrize_decorator(self):
        """Test that non-parametrize decorators are ignored."""
        # Create a different decorator
        pytest_name = ast.Name(id="pytest", ctx=ast.Load())
        mark_attr = ast.Attribute(value=pytest_name, attr="mark", ctx=ast.Load())
        skipif_attr = ast.Attribute(value=mark_attr, attr="skipif", ctx=ast.Load())

        decorator = ast.Call(func=skipif_attr, args=[ast.Constant(value=True)], keywords=[])

        result = _is_pytest_parametrize_with_completion_params(decorator)
        assert result is False

    def test_parametrize_without_completion_params(self):
        """Test that parametrize without completion_params is ignored."""
        decorator = create_parametrize_decorator(
            "other_param", ast.List(elts=[ast.Constant(value="value")], ctx=ast.Load())
        )

        result = _is_pytest_parametrize_with_completion_params(decorator)
        assert result is False
