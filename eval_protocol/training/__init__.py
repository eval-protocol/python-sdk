from typing import TYPE_CHECKING

# GEPA/DSPy-related imports are optional - only available when dspy extra is installed
# Use: pip install eval-protocol[dspy]
_DSPY_AVAILABLE = False
try:
    import dspy  # noqa: F401

    _DSPY_AVAILABLE = True
except ImportError:
    pass


def _raise_dspy_import_error(name: str):
    """Raise a helpful error when dspy is not installed."""
    raise ImportError(f"'{name}' requires the 'dspy' extra. Install it with: pip install eval-protocol[dspy]")


if TYPE_CHECKING or _DSPY_AVAILABLE:
    from .gepa_trainer import GEPATrainer
    from .gepa_utils import (
        DSPyModuleType,
        DSPyModuleFactory,
        create_single_turn_program,
        create_signature,
        build_reflection_lm,
    )

__all__ = [
    "GEPATrainer",
    # DSPy module creation utilities
    "DSPyModuleType",
    "DSPyModuleFactory",
    "create_single_turn_program",
    "create_signature",
    # Reflection LM helpers
    "build_reflection_lm",
]


def __getattr__(name: str):
    """Lazy loading for dspy-dependent exports."""
    if name in __all__ and not _DSPY_AVAILABLE:
        _raise_dspy_import_error(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
