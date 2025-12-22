from typing import Any

from eval_protocol.models import EPParameters


def build_ep_parameters_from_test(test_fn: Any) -> EPParameters:
    """
    Build an `EPParameters` instance from an `@evaluation_test`-decorated function.

    The decorator is responsible for attaching a `__ep_params__` attribute that
    contains all effective evaluation parameters after parsing/env overrides.
    """
    if not hasattr(test_fn, "__ep_params__"):
        raise ValueError(
            "The provided test function does not have `__ep_params__` attached. "
            "Ensure it is decorated with `@evaluation_test` from eval_protocol.pytest."
        )

    return getattr(test_fn, "__ep_params__")
