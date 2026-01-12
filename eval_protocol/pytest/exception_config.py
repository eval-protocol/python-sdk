"""
Exception handling configuration for rollout processors with backoff retry logic.

This module intentionally avoids importing heavy deps (litellm/requests/httpx)
at module import time to keep `@evaluation_test` import fast.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Set, Type, Union

import backoff
import eval_protocol.exceptions

# Cache for the default retryable exceptions (populated on first access)
_default_retryable_exceptions: Set[Type[Exception]] | None = None


def get_default_retryable_exceptions() -> Set[Type[Exception]]:
    """Compute the default set of retryable exceptions (lazy heavy imports)."""
    global _default_retryable_exceptions
    if _default_retryable_exceptions is not None:
        return _default_retryable_exceptions

    # Lazy imports (these are expensive)
    import httpx
    import litellm
    import requests

    _default_retryable_exceptions = {
        # Standard library exceptions
        ConnectionError,  # type: ignore[assignment]
        TimeoutError,  # type: ignore[assignment]
        OSError,  # type: ignore[assignment]  # Covers network-related OS errors
        # Requests library exceptions
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
        requests.exceptions.RequestException,
        # HTTPX library exceptions
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
        # LiteLLM library exceptions
        litellm.exceptions.RateLimitError,
        litellm.exceptions.InternalServerError,
        litellm.exceptions.Timeout,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.APIError,
        litellm.exceptions.BadRequestError,
        # Eval Protocol exceptions
        eval_protocol.exceptions.UnknownError,
        eval_protocol.exceptions.DeadlineExceededError,
        eval_protocol.exceptions.NotFoundError,
        eval_protocol.exceptions.PermissionDeniedError,
        eval_protocol.exceptions.UnavailableError,
        eval_protocol.exceptions.UnauthenticatedError,
        eval_protocol.exceptions.ResourceExhaustedError,
        eval_protocol.exceptions.ResponseQualityError,
    }

    return _default_retryable_exceptions


class _LazyDefaultRetryableExceptions(Set[Type[Exception]]):
    """Set-like view that materializes the default exception set on first use."""

    def __iter__(self):
        return iter(get_default_retryable_exceptions())

    def __len__(self) -> int:
        return len(get_default_retryable_exceptions())

    def __contains__(self, x: object) -> bool:
        return x in get_default_retryable_exceptions()

    def copy(self) -> Set[Type[Exception]]:
        return set(get_default_retryable_exceptions())


# Backwards compatible name: behaves like a set but doesn't import heavy deps until used
DEFAULT_RETRYABLE_EXCEPTIONS: Set[Type[Exception]] = _LazyDefaultRetryableExceptions()


@dataclass
class BackoffConfig:
    """Configuration for backoff behavior."""

    # Backoff strategy: 'expo' for exponential, 'constant' for constant delay
    strategy: str = "expo"

    # Base delay in seconds
    base_delay: float = 1.0

    # Maximum delay in seconds
    max_delay: float = 60.0

    # Maximum number of retry attempts
    max_tries: int = 3

    # Jitter: adds randomness to backoff delays (None = no jitter for predictable timing)
    # Backoff's jitter expects a function like `lambda value: float`
    jitter: Union[None, Callable[[float], float]] = None

    # Factor for exponential backoff (only used if strategy == 'expo')
    factor: float = 2.0

    # Whether to raise the exception when giving up (instead of returning it)
    raise_on_giveup: bool = True

    # Optional custom giveup function - if provided, overrides the default exception handling logic
    giveup_func: Callable[[Exception], bool] = lambda e: False

    def get_backoff_decorator(self, exceptions: Set[Type[Exception]]):
        """Get the appropriate backoff decorator based on configuration.

        Args:
            exceptions: Set of exception types to retry
        """
        if not exceptions:
            # If no exceptions specified, return a no-op decorator
            def no_op_decorator(func):
                return func

            return no_op_decorator

        if self.strategy == "expo":
            return backoff.on_exception(
                backoff.expo,
                tuple(exceptions),
                max_tries=self.max_tries,
                base=self.base_delay,
                max_value=self.max_delay,
                factor=self.factor,
                jitter=self.jitter,
                giveup=self.giveup_func,
                raise_on_giveup=self.raise_on_giveup,
            )
        elif self.strategy == "constant":
            return backoff.on_exception(
                backoff.constant,
                tuple(exceptions),
                max_tries=self.max_tries,
                interval=self.base_delay,
                jitter=self.jitter,
                giveup=self.giveup_func,
                raise_on_giveup=self.raise_on_giveup,
            )
        else:
            raise ValueError(f"Unknown backoff strategy: {self.strategy}")


@dataclass
class ExceptionHandlerConfig:
    """Configuration for exception handling in rollout processors."""

    # Exceptions that should be retried using backoff
    # Use field with default_factory to lazily get the exceptions
    retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: set(get_default_retryable_exceptions()))

    # Backoff configuration
    backoff_config: BackoffConfig = field(default_factory=BackoffConfig)

    def __post_init__(self):
        """Automatically apply environment variable overrides after initialization."""
        # Override backoff settings from environment variables
        if "EP_MAX_RETRY" in os.environ:
            max_retry = int(os.environ["EP_MAX_RETRY"])
            self.backoff_config.max_tries = max_retry

        if "EP_FAIL_ON_MAX_RETRY" in os.environ:
            fail_on_max_retry = os.environ["EP_FAIL_ON_MAX_RETRY"].lower()
            self.backoff_config.raise_on_giveup = fail_on_max_retry != "false"

    def get_backoff_decorator(self):
        """Get the backoff decorator configured for this exception handler."""
        return self.backoff_config.get_backoff_decorator(self.retryable_exceptions)


def get_default_exception_handler_config() -> ExceptionHandlerConfig:
    """Get a fresh default exception handler configuration."""
    return ExceptionHandlerConfig()
