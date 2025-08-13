import importlib
from unittest.mock import patch

import pytest

# Module to be tested
from eval_protocol import generic_server
from eval_protocol.generic_server import load_reward_function

# Dummy functions for testing from our dummy module
from tests.dummy_module_for_server_test import (
    dummy_accepts_args_returns_string,
    dummy_reward_func,
    not_a_reward_function,
)


# Reset the global loaded function in generic_server before each test in this module
@pytest.fixture(autouse=True)
def reset_generic_server_state():
    generic_server._LOADED_REWARD_FUNCTION = None
    generic_server._REWARD_FUNCTION_NAME = "N/A"
    yield  # Test runs here
    generic_server._LOADED_REWARD_FUNCTION = None
    generic_server._REWARD_FUNCTION_NAME = "N/A"


class TestLoadRewardFunction:
    def test_load_successful(self):
        """Test successful loading of a valid reward function."""
        import_string = "tests.dummy_module_for_server_test.dummy_reward_func"
        load_reward_function(import_string)
        assert generic_server._LOADED_REWARD_FUNCTION is not None
        assert generic_server._LOADED_REWARD_FUNCTION == dummy_reward_func
        assert generic_server._REWARD_FUNCTION_NAME == import_string

    def test_load_module_not_found(self):
        """Test loading from a non-existent module."""
        import_string = "non_existent_module.some_function"
        with pytest.raises(ImportError):  # Or ModuleNotFoundError for Python 3.6+
            load_reward_function(import_string)
        assert generic_server._LOADED_REWARD_FUNCTION is None
        assert generic_server._REWARD_FUNCTION_NAME == "Error loading"  # As set in except block

    def test_load_function_not_found(self):
        """Test loading a non-existent function from an existing module."""
        import_string = "tests.dummy_module_for_server_test.non_existent_function"
        with pytest.raises(AttributeError):
            load_reward_function(import_string)
        assert generic_server._LOADED_REWARD_FUNCTION is None
        assert generic_server._REWARD_FUNCTION_NAME == "Error loading"

    def test_load_module_object_successfully(self):
        """Test loading a module object itself using an import string like 'package.module'."""
        # import_string "tests.dummy_module_for_server_test" means:
        # module_path = "tests"
        # attribute_name = "dummy_module_for_server_test"
        # This will load the 'tests' package, then get the 'dummy_module_for_server_test'
        # attribute, which is the module object itself. load_reward_function should allow this.
        import_string = "tests.dummy_module_for_server_test"
        load_reward_function(import_string)  # Should not raise an error

        from tests import dummy_module_for_server_test as expected_module_object

        assert generic_server._LOADED_REWARD_FUNCTION is expected_module_object
        assert generic_server._REWARD_FUNCTION_NAME == import_string

    def test_load_invalid_import_string_format_no_dot(self):
        """Test import string without a dot, causing rsplit to fail."""
        import_string = "nodothere"
        with pytest.raises(ValueError, match="not enough values to unpack"):
            load_reward_function(import_string)
        assert generic_server._LOADED_REWARD_FUNCTION is None
        assert generic_server._REWARD_FUNCTION_NAME == "Error loading"

    def test_load_function_is_not_callable_if_not_reward_function_type(self):
        import_string = "tests.dummy_module_for_server_test.not_a_reward_function"
        load_reward_function(import_string)
        assert generic_server._LOADED_REWARD_FUNCTION is not None
        assert generic_server._LOADED_REWARD_FUNCTION == not_a_reward_function
        assert generic_server._REWARD_FUNCTION_NAME == import_string

    def test_global_state_update_on_load(self):
        import_string = "tests.dummy_module_for_server_test.dummy_reward_func"
        load_reward_function(import_string)
        assert generic_server._LOADED_REWARD_FUNCTION == dummy_reward_func
        assert generic_server._REWARD_FUNCTION_NAME == import_string

    def test_load_failure_resets_globals(self):
        success_import_string = "tests.dummy_module_for_server_test.dummy_reward_func"
        load_reward_function(success_import_string)
        assert generic_server._LOADED_REWARD_FUNCTION == dummy_reward_func

        fail_import_string = "non_existent_module.some_function"
        with pytest.raises(ImportError):
            load_reward_function(fail_import_string)

        assert generic_server._LOADED_REWARD_FUNCTION is None
        assert generic_server._REWARD_FUNCTION_NAME == "Error loading"


# --- Tests for FastAPI app endpoints ---
from fastapi.testclient import TestClient

from eval_protocol.generic_server import EvaluationRequest, app as generic_fastapi_app
from eval_protocol.models import EvaluateResult, Message, MetricResult


class TestServerEndpoints:
    client = TestClient(generic_fastapi_app)

    def test_health_check_no_function_loaded(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "error",
            "reason": "Reward function not loaded",
        }

    def test_health_check_function_loaded(self):
        load_reward_function("tests.dummy_module_for_server_test.dummy_reward_func")
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "reward_function": "tests.dummy_module_for_server_test.dummy_reward_func",
        }

    def test_evaluate_endpoint_no_function_loaded(self):
        request_payload = EvaluationRequest(messages=[{"role": "user", "content": "test"}])
        response = self.client.post("/evaluate", json=request_payload.model_dump())
        assert response.status_code == 500
        assert response.json() == {"detail": "Reward function not loaded."}

    def test_evaluate_endpoint_success(self):
        load_reward_function("tests.dummy_module_for_server_test.dummy_reward_func")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        request_payload = EvaluationRequest(messages=messages, ground_truth="success", kwargs={"extra_param": 123})
        response = self.client.post("/evaluate", json=request_payload.model_dump())
        assert response.status_code == 200
        result_data = response.json()

        assert "score" in result_data
        assert result_data["score"] == 1.0
        assert "reason" in result_data
        assert "With kwargs: {'extra_param': 123}" in result_data["reason"]
        assert "metrics" in result_data
        assert "dummy_metric" in result_data["metrics"]
        assert result_data["metrics"]["dummy_metric"]["score"] == 0.75
        parsed_result = EvaluateResult(**result_data)
        assert parsed_result.score == 1.0

    def test_evaluate_endpoint_invalid_request_payload(self):
        load_reward_function("tests.dummy_module_for_server_test.dummy_reward_func")
        invalid_payload = {"ground_truth": "test"}
        response = self.client.post("/evaluate", json=invalid_payload)
        assert response.status_code == 422
        assert "detail" in response.json()
        assert isinstance(response.json()["detail"], list)
        assert response.json()["detail"][0]["type"] == "missing"
        assert "messages" in response.json()["detail"][0]["loc"]

    def test_evaluate_endpoint_reward_function_raises_error(self):
        load_reward_function("tests.dummy_module_for_server_test.dummy_reward_func_error")
        request_payload = EvaluationRequest(messages=[{"role": "user", "content": "test"}])
        response = self.client.post("/evaluate", json=request_payload.model_dump())
        assert response.status_code == 500
        assert "Intentional error in dummy_reward_func_error" in response.json()["detail"]

    def test_evaluate_endpoint_function_returns_invalid_type(self):
        """
        Tests the server's fallback when a loaded function is callable with standard args
        but returns a type that is not EvaluateResult.
        This uses a function NOT decorated with @reward_function.
        """
        load_reward_function("tests.dummy_module_for_server_test.dummy_accepts_args_returns_string")
        request_payload = EvaluationRequest(messages=[{"role": "user", "content": "test"}])
        response = self.client.post("/evaluate", json=request_payload.model_dump())

        assert response.status_code == 200
        result_data = response.json()
        assert result_data["score"] == 0.0
        assert result_data["reason"] == "Invalid return type from reward function, check server logs."
        assert result_data["is_score_valid"] is False
        assert result_data["metrics"] == {}

    def test_evaluate_endpoint_decorated_function_returns_coercible_dict(self):
        """
        Tests a @reward_function decorated function that returns a dict which
        the decorator should coerce into an EvaluateResult.
        """
        load_reward_function("tests.dummy_module_for_server_test.dummy_reward_func_invalid_return")
        request_payload = EvaluationRequest(messages=[{"role": "user", "content": "test"}])
        response = self.client.post("/evaluate", json=request_payload.model_dump())

        assert response.status_code == 200
        result_data = response.json()
        # The @reward_function decorator should have coerced the dict.
        # The dummy_reward_func_invalid_return returns:
        # {"score": 0.1, "reason": "This is a dict, not EvaluateResult", "is_score_valid": True, "metrics": {}}
        assert result_data["score"] == 0.1
        assert result_data["reason"] == "This is a dict, not EvaluateResult"
        assert result_data["is_score_valid"] is True
        assert result_data["metrics"] == {}
