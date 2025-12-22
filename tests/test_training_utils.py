import pytest

from eval_protocol.models import EPParameters
from eval_protocol.training.utils import build_ep_parameters_from_test


def test_build_ep_parameters_from_test_returns_attached_model():
    """build_ep_parameters_from_test should return the EPParameters attached to the test function."""

    def dummy_test() -> None:
        pass

    params = EPParameters(num_runs=3, completion_params={"model": "gpt-4"})
    setattr(dummy_test, "__ep_params__", params)

    result = build_ep_parameters_from_test(dummy_test)

    assert result is params
    assert result.num_runs == 3
    assert result.completion_params == {"model": "gpt-4"}


def test_build_ep_parameters_from_test_missing_attr_raises():
    """build_ep_parameters_from_test should raise when __ep_params__ is missing."""

    def dummy_test_no_attr() -> None:
        pass

    with pytest.raises(ValueError) as exc_info:
        build_ep_parameters_from_test(dummy_test_no_attr)

    assert "__ep_params__" in str(exc_info.value)
