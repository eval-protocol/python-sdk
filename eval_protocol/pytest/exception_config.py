"""
Exception handling configuration for rollout processors with backoff retry logic.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Set, Type, Union

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
    eval_protocol.exceptions.ResponseQualityError,
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

    def get_backoff_decorator(self, exceptions: Set[Type[Exception]], exception_backoff_overrides: Dict[Type[Exception], "BackoffConfig"] | None = None):
        """Get the appropriate backoff decorator based on configuration.
        
        Args:
            exceptions: Set of exception types to retry
            exception_backoff_overrides: Optional mapping of exception types to custom backoff configs.
                If an exception type has an override, that config will be used instead of this one.
        """
        if not exceptions:
            # If no exceptions specified, return a no-op decorator
            def no_op_decorator(func):
                return func

            return no_op_decorator

        # If no overrides, use simple decorator for all exceptions
        if not exception_backoff_overrides:
            return self._create_single_decorator(exceptions, self)
        
        # Group exceptions by their backoff config to avoid double backoff
        # Each exception type gets exactly one decorator based on its config
        # Use a tuple of config attributes as the key since BackoffConfig is not hashable
        config_to_exceptions: Dict[tuple, tuple[Set[Type[Exception]], "BackoffConfig"]] = {}
        
        for exc_type in exceptions:
            if exc_type in exception_backoff_overrides:
                override_config = exception_backoff_overrides[exc_type]
            else:
                override_config = self
            
            # Create a hashable key from config attributes
            # Note: jitter and giveup_func are callable, which are hashable in Python
            config_key = (
                override_config.strategy,
                override_config.base_delay,
                override_config.max_delay,
                override_config.max_tries,
                override_config.factor,
                id(override_config.jitter) if override_config.jitter is not None else None,
                id(override_config.giveup_func) if override_config.giveup_func is not None else None,
                override_config.raise_on_giveup,
            )
            
            if config_key not in config_to_exceptions:
                config_to_exceptions[config_key] = (set(), override_config)
            exc_set, _ = config_to_exceptions[config_key]
            exc_set.add(exc_type)
        
        # If all exceptions use the same config, use a single decorator
        if len(config_to_exceptions) == 1:
            exc_set, config = next(iter(config_to_exceptions.values()))
            return self._create_single_decorator(exc_set, config)
        
        # Create separate decorators for each config group
        # Each exception type gets exactly one decorator, preventing double backoff
        decorators_by_config: list[tuple[Set[Type[Exception]], Callable]] = []
        
        for exc_set, config in config_to_exceptions.values():
            decorator = self._create_single_decorator(exc_set, config)
            if decorator:
                decorators_by_config.append((exc_set, decorator))
        
        # Create a combined decorator that applies all decorators
        # Each decorator only catches exceptions in its exception set, so no double backoff
        def combined_decorator(func):
            decorated_func = func
            
            # Apply each decorator in order (inner to outer)
            # Each decorator only catches exceptions in its specific exception set
            # Since exception sets are disjoint (grouped by config), no double backoff
            for exc_set, decorator in decorators_by_config:
                decorated_func = decorator(decorated_func)
            
            return decorated_func
        
        return combined_decorator
    
    def _create_single_decorator(self, exc_set: Set[Type[Exception]], config: "BackoffConfig"):
        """Create a single backoff decorator for a set of exceptions."""
        if not exc_set:
            return None
        
        if config.strategy == "expo":
            return backoff.on_exception(
                backoff.expo,
                tuple(exc_set),
                max_tries=config.max_tries,
                base=config.base_delay,
                max_value=config.max_delay,
                factor=config.factor,
                jitter=config.jitter,
                giveup=config.giveup_func,
                raise_on_giveup=config.raise_on_giveup,
            )
        elif config.strategy == "constant":
            return backoff.on_exception(
                backoff.constant,
                tuple(exc_set),
                max_tries=config.max_tries,
                interval=config.base_delay,
                jitter=config.jitter,
                giveup=config.giveup_func,
                raise_on_giveup=config.raise_on_giveup,
            )
        else:
            raise ValueError(f"Unknown backoff strategy: {config.strategy}")


@dataclass
class ExceptionHandlerConfig:
    """Configuration for exception handling in rollout processors."""

    # Exceptions that should be retried using backoff
    retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: DEFAULT_RETRYABLE_EXCEPTIONS.copy())

    # Backoff configuration
    backoff_config: BackoffConfig = field(default_factory=BackoffConfig)

    # Per-exception backoff overrides - allows custom backoff config for specific exception types
    # For example, ResponseQualityError can use no backoff (base_delay=0, max_delay=0)
    exception_backoff_overrides: Dict[Type[Exception], BackoffConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Automatically apply environment variable overrides after initialization."""
        # Override backoff settings from environment variables
        if "EP_MAX_RETRY" in os.environ:
            max_retry = int(os.environ["EP_MAX_RETRY"])
            self.backoff_config.max_tries = max_retry

        if "EP_FAIL_ON_MAX_RETRY" in os.environ:
            fail_on_max_retry = os.environ["EP_FAIL_ON_MAX_RETRY"].lower()
            self.backoff_config.raise_on_giveup = fail_on_max_retry != "false"
        
        # Set default no-backoff config for ResponseQualityError if not already set
        if eval_protocol.exceptions.ResponseQualityError not in self.exception_backoff_overrides:
            # Default: no backoff for ResponseQualityError (immediate retry)
            self.exception_backoff_overrides[eval_protocol.exceptions.ResponseQualityError] = BackoffConfig(
                strategy="constant",
                base_delay=0.0,
                max_delay=0.0,
                max_tries=self.backoff_config.max_tries,
            )

    def get_backoff_decorator(self):
        """Get the backoff decorator configured for this exception handler."""
        return self.backoff_config.get_backoff_decorator(
            self.retryable_exceptions,
            self.exception_backoff_overrides if self.exception_backoff_overrides else None
        )


def get_default_exception_handler_config() -> ExceptionHandlerConfig:
    """Get a fresh default exception handler configuration."""
    return ExceptionHandlerConfig()
