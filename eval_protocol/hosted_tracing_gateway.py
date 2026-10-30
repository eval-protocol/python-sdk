"""Deprecation notice for features built on the hosted Fireworks tracing gateway."""


def hosted_gateway_deprecation_message(feature: str, configured_via: str) -> str:
    """Build the deprecation warning for a feature that depends on the hosted tracing gateway.

    Args:
        feature: User-facing name of the deprecated feature, e.g. "RemoteRolloutProcessor".
        configured_via: Parameters or environment variables that point the feature at a self-hosted gateway.
    """
    return (
        "The hosted Fireworks tracing gateway (https://tracing.fireworks.ai) was shut down on October 28, 2026. "
        f"{feature} depends on it and is deprecated; it will be removed in a future release. "
        "Until then, it only works with a self-hosted gateway that implements the same API, "
        f"set via {configured_via}."
    )
