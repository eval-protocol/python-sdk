"""Utility functions for model name handling."""

import re

from eval_protocol.models import CompletionParams


def normalize_fireworks_model_for_litellm(completion_params: CompletionParams | None) -> CompletionParams | None:
    """Fireworks model names like 'accounts/<org>/models/<model>' or 'accounts/<org>/deployments/<model>'
    need the fireworks_ai/ prefix when routing through LiteLLM. This function adds the prefix if missing.
    """
    if completion_params is None:
        return None

    model = completion_params.get("model")
    if (
        model
        and isinstance(model, str)
        and not model.startswith("fireworks_ai/")
        and re.match(r"^accounts/[^/]+/(models|deployments)/.+", model)
    ):
        completion_params = completion_params.copy()
        completion_params["model"] = f"fireworks_ai/{model}"
    return completion_params
