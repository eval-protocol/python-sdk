import os
import pytest


MODEL_ID_OPT = None
CONCURRENCY_OPT = None
MODEL_KWARGS_OPT = None


def pytest_addoption(parser):
    parser.addoption("--model-id", action="store", default=None, help="Fireworks model ID")
    parser.addoption("--concurrent-workers", action="store", type=int, default=None, help="Max concurrent rollouts")
    parser.addoption("--temperature", action="store", type=float, default=None, help="Model temperature")
    parser.addoption("--max-tokens", action="store", type=int, default=None, help="Max tokens")
    parser.addoption(
        "--reasoning", action="store", choices=["low", "medium", "high"], default=None, help="Reasoning effort"
    )


def pytest_configure(config):
    global MODEL_ID_OPT, CONCURRENCY_OPT, MODEL_KWARGS_OPT
    MODEL_ID_OPT = config.getoption("--model-id")
    CONCURRENCY_OPT = config.getoption("--concurrent-workers")
    temp = config.getoption("--temperature")
    mtok = config.getoption("--max-tokens")
    reas = config.getoption("--reasoning")
    mk = {}
    if temp is not None:
        mk["temperature"] = float(temp)
    if mtok is not None:
        mk["max_tokens"] = int(mtok)
    if reas is not None:
        mk["reasoning"] = reas
    MODEL_KWARGS_OPT = mk or None
