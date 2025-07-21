import importlib.util
import os

import pytest

from eval_protocol.models import Message


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_example_module():
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "braintrust_example",
        "main.py",
    )
    return load_module_from_path("braintrust_example_main_test", example_path)


def test_evaluate_match():
    module = get_example_module()
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    ground_truth = [Message(role="assistant", content="hello")]
    result = module.evaluate(messages=messages, ground_truth=ground_truth)
    assert result.score == 1.0
    assert result.is_score_valid is True


def test_evaluate_mismatch():
    module = get_example_module()
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="goodbye"),
    ]
    ground_truth = [Message(role="assistant", content="hello")]
    result = module.evaluate(messages=messages, ground_truth=ground_truth)
    assert result.score == 0.0
    assert result.is_score_valid is True
