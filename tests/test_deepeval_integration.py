import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union
from unittest.mock import patch

# --- DEEPEVAL_AVAILABLE and type imports ---
DEEPEVAL_AVAILABLE = False
# Initialize all potentially imported names to None
BaseMetric, GEval, GPTModel, DeepEvalBaseLLM = None, None, None, None
LLMTestCase, LLMTestCaseParams = None, None
valid_gpt_models, model_pricing, parse_model_name = None, None, None
# OpenAI Pydantic types
(
    ChatCompletion,
    OpenAIChoice,
    ChatCompletionMessage,
    ChatCompletionTokenLogprob,
    OpenAIChoiceLogprobs,
    CompletionUsage,
) = (None, None, None, None, None, None)
httpx_module = None

try:
    from deepeval.metrics import GEval
    from deepeval.metrics.base_metric import BaseMetric
    from deepeval.models import DeepEvalBaseLLM, GPTModel
    from deepeval.models.llms.openai_model import model_pricing, valid_gpt_models
    from deepeval.models.utils import parse_model_name
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    DEEPEVAL_AVAILABLE = True

    try:
        import httpx as httpx_module
        from openai.types.chat.chat_completion import ChatCompletion
        from openai.types.chat.chat_completion_choice import Choice as OpenAIChoice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage
        from openai.types.chat.chat_completion_token_logprob import (
            ChatCompletionTokenLogprob,
        )
        from openai.types.chat.choice_logprobs import ChoiceLogprobs as OpenAIChoiceLogprobs
        from openai.types.completion_usage import CompletionUsage
    except ImportError:
        print("Warning: Failed to import some openai.types. Mock will use dicts instead of Pydantic models.")
        # DEEPEVAL_AVAILABLE remains True; mock will adapt.
        pass

except ImportError:
    DEEPEVAL_AVAILABLE = False

    # Fallback dummy definitions if core deepeval failed
    class BaseMetric:
        pass  # type: ignore

    class DeepEvalBaseLLM:
        pass  # type: ignore

    class GEval:
        pass  # type: ignore

    class GPTModel:
        pass  # type: ignore

    class LLMTestCase:  # type: ignore
        def __init__(self, input: str = "", actual_output: str = "", expected_output: str = "") -> None:
            pass

    class LLMTestCaseParams:  # type: ignore
        INPUT = "input"
        ACTUAL_OUTPUT = "actual_output"
        EXPECTED_OUTPUT = "expected_output"

    valid_gpt_models = []  # type: ignore
    model_pricing = {}  # type: ignore

    def parse_model_name(name):
        return name  # type: ignore


# Fallback for OpenAI types if they were not imported (ChatCompletion will be None, etc.)
# The mock will handle this by returning SimpleNamespace if Pydantic types are None.

from eval_protocol.integrations.deepeval import adapt_metric
from eval_protocol.models import EvaluateResult


class DummyMetric(BaseMetric):  # type: ignore
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # type: ignore
        self.score = 1.0 if test_case.actual_output == test_case.expected_output else 0.0  # type: ignore
        self.reason = "match" if self.score == 1.0 else "mismatch"
        self.success = self.score >= self.threshold
        return self.score  # type: ignore

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # type: ignore
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success is True


class DummyGEval(BaseMetric):  # type: ignore
    evaluation_params = (
        [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
        if DEEPEVAL_AVAILABLE and hasattr(LLMTestCaseParams, "INPUT")
        else []
    )  # type: ignore

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.last_case = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # type: ignore
        self.last_case = test_case
        self.score = 1.0 if test_case.actual_output == test_case.expected_output else 0.0  # type: ignore
        self.reason = "match" if self.score == 1.0 else "mismatch"
        self.success = self.score >= self.threshold
        return self.score  # type: ignore

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # type: ignore
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success is True


class TestDeepevalIntegration(unittest.TestCase):
    original_openai_api_key_env = None
    original_openai_base_url_env = None
    original_valid_models_list_content_backup = None
    original_model_pricing_dict_content_backup = None

    def setUp(self):
        if not DEEPEVAL_AVAILABLE:
            self.skipTest("Core deepeval modules not available.")
        self.original_valid_models_list_content_backup = list(valid_gpt_models)  # type: ignore
        self.original_model_pricing_dict_content_backup = model_pricing.copy()  # type: ignore
        self.original_openai_api_key_env = os.environ.get("OPENAI_API_KEY")
        self.original_openai_base_url_env = os.environ.get("OPENAI_BASE_URL")

    def tearDown(self):
        if not DEEPEVAL_AVAILABLE:
            return
        if self.original_valid_models_list_content_backup is not None:
            valid_gpt_models.clear()
            valid_gpt_models.extend(self.original_valid_models_list_content_backup)  # type: ignore
        if self.original_model_pricing_dict_content_backup is not None:
            model_pricing.clear()
            model_pricing.update(self.original_model_pricing_dict_content_backup)  # type: ignore
        if self.original_openai_api_key_env is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self.original_openai_api_key_env
        if self.original_openai_base_url_env is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = self.original_openai_base_url_env

    @unittest.skipUnless(DEEPEVAL_AVAILABLE, "deepeval package is required")
    def test_dummy_metric_wrapper(self) -> None:
        metric = DummyMetric()
        wrapped = adapt_metric(metric)
        messages = [{"role": "assistant", "content": "hi"}]
        result = wrapped(messages=messages, ground_truth="hi")
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics[metric.__class__.__name__].reason, "match")

    @unittest.skipUnless(DEEPEVAL_AVAILABLE, "deepeval package is required")
    def test_dummy_geval_wrapper(self) -> None:
        metric = DummyGEval()
        wrapped = adapt_metric(metric)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi"},
        ]
        result = wrapped(messages=messages, ground_truth="hi")
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(metric.last_case.input, "hi")  # type: ignore

    @unittest.skipUnless(DEEPEVAL_AVAILABLE, "deepeval package is required")
    def test_fireworks_geval_integration_with_mock(self) -> None:
        fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
        if not fireworks_api_key:
            self.skipTest("FIREWORKS_API_KEY environment variable not set.")
        os.environ["OPENAI_API_KEY"] = fireworks_api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.fireworks.ai/inference/v1"

        fireworks_model_name_for_api = "accounts/fireworks/models/deepseek-v3-0324"
        parsed_fireworks_model_name = parse_model_name(fireworks_model_name_for_api)  # type: ignore
        if parsed_fireworks_model_name not in valid_gpt_models:
            valid_gpt_models.append(parsed_fireworks_model_name)  # type: ignore

        actual_fireworks_model_for_geval = GPTModel(
            model=fireworks_model_name_for_api, _openai_api_key=fireworks_api_key
        )  # type: ignore
        actual_fireworks_model_for_geval.model_name = fireworks_model_name_for_api  # type: ignore

        if fireworks_model_name_for_api not in model_pricing:
            model_pricing[fireworks_model_name_for_api] = {"input": 0.0, "output": 0.0}  # type: ignore
        if (
            parsed_fireworks_model_name != fireworks_model_name_for_api
            and parsed_fireworks_model_name not in model_pricing
        ):
            model_pricing[parsed_fireworks_model_name] = {"input": 0.0, "output": 0.0}  # type: ignore

        mock_call_count = 0
        openai_types_available = all(
            [
                OpenAIChoice,
                ChatCompletionMessage,
                ChatCompletion,
                CompletionUsage,
                ChatCompletionTokenLogprob,
                OpenAIChoiceLogprobs,
            ]
        )

        async def mock_chat_completions_create(*args, messages: List[dict], model: str, **kwargs):  # type: ignore
            nonlocal mock_call_count
            mock_call_count += 1

            usage_dict = {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            }
            if openai_types_available:  # Construct Pydantic models if types were imported
                usage_obj = CompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
                if mock_call_count == 1:
                    response_content_json = json.dumps({"steps": ["Mocked: Response is relevant."]})
                    choice = OpenAIChoice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=response_content_json,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                        logprobs=None,
                    )
                    return ChatCompletion(
                        id="chatcmpl-mock-steps",
                        object="chat.completion",
                        created=123,
                        model=model,
                        choices=[choice],
                        usage=usage_obj,
                    )
                elif mock_call_count == 2:
                    # Provide logprobs with high probability for high score
                    mock_logprobs = OpenAIChoiceLogprobs(
                        content=[ChatCompletionTokenLogprob(token="10", logprob=0.0, bytes=None, top_logprobs=[])]
                    )
                    mock_score_reason_json = json.dumps(
                        {
                            "reason": "Mocked reason: Score derived from logprobs.",
                            "score": 10,
                        }
                    )
                    choice = OpenAIChoice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=mock_score_reason_json,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                        logprobs=mock_logprobs,
                    )
                    return ChatCompletion(
                        id="chatcmpl-mock-score",
                        object="chat.completion",
                        created=124,
                        model=model,
                        choices=[choice],
                        usage=usage_obj,
                    )
            else:  # Fallback to SimpleNamespace if Pydantic types are not available
                if mock_call_count == 1:
                    response_content_json = json.dumps({"steps": ["Mocked: Response is relevant."]})
                    return SimpleNamespace(
                        id="chatcmpl-mock-steps",
                        object="chat.completion",
                        created=123,
                        model=model,
                        choices=[
                            SimpleNamespace(
                                index=0,
                                message=SimpleNamespace(
                                    role="assistant",
                                    content=response_content_json,
                                    tool_calls=None,
                                ),
                                finish_reason="stop",
                                logprobs=None,
                            )
                        ],
                        usage=SimpleNamespace(**usage_dict),
                    )
                elif mock_call_count == 2:
                    # Provide logprobs with high probability for high score
                    mock_logprobs = SimpleNamespace(
                        content=[SimpleNamespace(token="10", logprob=0.0, bytes=None, top_logprobs=[])]
                    )
                    mock_score_reason_json = json.dumps(
                        {
                            "reason": "Mocked reason: Score derived from logprobs.",
                            "score": 10,
                        }
                    )
                    return SimpleNamespace(
                        id="chatcmpl-mock-score",
                        object="chat.completion",
                        created=124,
                        model=model,
                        choices=[
                            SimpleNamespace(
                                index=0,
                                message=SimpleNamespace(
                                    role="assistant",
                                    content=mock_score_reason_json,
                                    tool_calls=None,
                                ),
                                finish_reason="stop",
                                logprobs=mock_logprobs,
                            )
                        ],
                        usage=SimpleNamespace(**usage_dict),
                    )
            raise AssertionError(f"Unexpected call count: {mock_call_count}")

        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create",
            new=mock_chat_completions_create,
        ):
            geval_metric = GEval(
                name="Fireworks GEval Mocked",
                criteria="Evaluate the helpfulness and relevance of the actual output based on the input.",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                model=actual_fireworks_model_for_geval,
                strict_mode=False,
            )  # type: ignore
            wrapped_metric = adapt_metric(geval_metric)
            messages_data = [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]
            result = wrapped_metric(messages=messages_data)

        self.assertEqual(mock_call_count, 2, "Expected two calls to mocked LLM endpoint.")
        self.assertIsInstance(result, EvaluateResult)
        self.assertIsNotNone(result.score, "GEval score should not be None")
        self.assertEqual(result.score, 1.0, f"GEval score {result.score} was not 1.0 with mock.")
        expected_metric_key = f"{geval_metric.name} ({geval_metric.__class__.__name__})"  # type: ignore
        self.assertIn(
            expected_metric_key,
            result.metrics,
            f"Constructed metric key '{expected_metric_key}' not found. Keys: {list(result.metrics.keys())}",
        )  # type: ignore
        self.assertIsNotNone(result.metrics[expected_metric_key].reason)  # type: ignore


if __name__ == "__main__":
    unittest.main()
