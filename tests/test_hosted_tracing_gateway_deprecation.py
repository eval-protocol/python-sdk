"""Deprecation of the features built on the hosted Fireworks tracing gateway."""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Callable, List

import pytest

import eval_protocol.cli_commands.logs as logs_cli
import eval_protocol.log_utils.init as log_init
from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter
from eval_protocol.hosted_tracing_gateway import hosted_gateway_deprecation_message
from eval_protocol.log_utils.fireworks_tracing_http_handler import FireworksTracingHttpHandler
from eval_protocol.pytest.github_action_rollout_processor import GithubActionRolloutProcessor
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor

SHUTDOWN_NOTICE = (
    "The hosted Fireworks tracing gateway (https://tracing.fireworks.ai) was shut down on October 28, 2026."
)
SELF_HOSTED_GATEWAY_URL = "http://self-hosted-gateway.test"


@pytest.fixture(autouse=True)
def clear_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FIREWORKS_API_KEY", "FW_TRACING_GATEWAY_BASE_URL", "GATEWAY_URL"):
        monkeypatch.delenv(name, raising=False)


def _gateway_deprecations(caught: List[warnings.WarningMessage]) -> List[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, DeprecationWarning) and SHUTDOWN_NOTICE in str(w.message)]


def _attach_external_logging() -> List[logging.Handler]:
    """Run init_external_logging_from_env and detach whatever it attached to the root logger."""
    root = logging.getLogger()
    before = list(root.handlers)
    log_init.init_external_logging_from_env()
    added = [handler for handler in root.handlers if handler not in before]
    for handler in added:
        root.removeHandler(handler)
        handler.close()
    return added


def test_deprecation_message_wording() -> None:
    assert hosted_gateway_deprecation_message("Feature", "feature_url") == (
        "The hosted Fireworks tracing gateway (https://tracing.fireworks.ai) was shut down on October 28, 2026. "
        "Feature depends on it and is deprecated; it will be removed in a future release. "
        "Until then, it only works with a self-hosted gateway that implements the same API, set via feature_url."
    )


@pytest.mark.parametrize(
    ("construct", "feature"),
    [
        (lambda: RemoteRolloutProcessor(remote_base_url="http://127.0.0.1:1"), "RemoteRolloutProcessor"),
        (
            lambda: GithubActionRolloutProcessor(owner="owner", repo="repo", workflow_id="rollout.yml"),
            "GithubActionRolloutProcessor",
        ),
        (lambda: FireworksTracingAdapter(), "FireworksTracingAdapter"),
        (lambda: FireworksTracingHttpHandler(), "FireworksTracingHttpHandler"),
    ],
)
def test_entry_point_warns_at_caller(construct: Callable[[], object], feature: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        construct()

    deprecations = [w for w in _gateway_deprecations(caught) if f"{feature} depends on it" in str(w.message)]
    assert len(deprecations) == 1
    assert Path(deprecations[0].filename) == Path(__file__)


def test_init_does_not_attach_fireworks_handler_without_gateway_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(log_init, "_INITIALIZED", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        added = _attach_external_logging()

    assert added == []
    assert _gateway_deprecations(caught) == []


@pytest.mark.parametrize("gateway_env_var", ["FW_TRACING_GATEWAY_BASE_URL", "GATEWAY_URL"])
def test_init_attaches_fireworks_handler_for_explicit_gateway_url(
    monkeypatch: pytest.MonkeyPatch, gateway_env_var: str
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv(gateway_env_var, SELF_HOSTED_GATEWAY_URL)
    monkeypatch.setattr(log_init, "_INITIALIZED", False)

    with pytest.warns(DeprecationWarning, match="FireworksTracingHttpHandler depends on it"):
        added = _attach_external_logging()

    assert len(added) == 1
    assert isinstance(added[0], log_init.FireworksTracingHttpHandler)
    assert added[0].gateway_base_url == SELF_HOSTED_GATEWAY_URL


def _logs_args() -> argparse.Namespace:
    return argparse.Namespace(port=8000, debug=False, use_fireworks=False, use_elasticsearch=False)


def test_ep_logs_warns_when_using_fireworks_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(logs_cli, "serve_logs", lambda **kwargs: None)

    with pytest.warns(DeprecationWarning, match="The Fireworks backend of 'ep logs' depends on it"):
        assert logs_cli.logs_command(_logs_args()) == 0


def test_ep_logs_does_not_warn_for_elasticsearch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logs_cli, "serve_logs", lambda **kwargs: None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert logs_cli.logs_command(_logs_args()) == 0

    assert _gateway_deprecations(caught) == []
