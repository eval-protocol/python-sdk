"""
Exception handling configuration for rollout processors with backoff retry logic.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Set, Type, Union

import backoff

import litellm
import requests
import httpx

import eval_protocol.exceptions


# Default exceptions that should be retried with backoff
DEFAULT_RETRYABLE_EXCEPTIONS: Set[Type[Exception]] = {
    # Standard library exceptions
    ConnectionError,
    TimeoutError,
    OSError,  # Covers network-related OS errors
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
}


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
    jitter: Union[None, Callable] = None

    # Factor for exponential backoff (only used if strategy == 'expo')
    factor: float = 2.0

    # Whether to raise the exception when giving up (instead of returning it)
    raise_on_giveup: bool = True

    # Optional custom giveup function - if provided, overrides the default exception handling logic
    giveup_func: Callable[[Exception], bool] = lambda e: False

    def get_backoff_decorator(self, exceptions: Set[Type[Exception]]):
        """Get the appropriate backoff decorator based on configuration."""
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
    retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: DEFAULT_RETRYABLE_EXCEPTIONS.copy())

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
