import os
from unittest.mock import patch

import pytest

from eval_protocol.auth import (
    get_fireworks_account_id,
    get_fireworks_api_key,
    verify_api_key_and_get_account_id,
)


TEST_ENV_API_KEY = "test_env_api_key_123"


@pytest.fixture(autouse=True)
def clear_env_vars_fixture():
    env_vars_to_clear = ["FIREWORKS_API_KEY", "FIREWORKS_API_BASE"]
    original_values = {var: os.environ.get(var) for var in env_vars_to_clear}
    for var in env_vars_to_clear:
        os.environ.pop(var, None)
    yield
    for var, value in original_values.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


def test_get_api_key_from_env():
    os.environ["FIREWORKS_API_KEY"] = TEST_ENV_API_KEY
    assert get_fireworks_api_key() == TEST_ENV_API_KEY


def test_get_api_key_not_found():
    assert get_fireworks_api_key() is None


def test_get_account_id_not_found():
    assert get_fireworks_account_id() is None


def test_verify_api_key_and_get_account_id_success_from_header():
    os.environ["FIREWORKS_API_KEY"] = TEST_ENV_API_KEY

    class _Resp:
        status_code = 200
        headers = {"x-fireworks-account-id": "acct_123"}

    with patch("eval_protocol.auth.requests.get", return_value=_Resp()):
        assert verify_api_key_and_get_account_id() == "acct_123"


def test_verify_api_key_and_get_account_id_non_200_returns_none():
    os.environ["FIREWORKS_API_KEY"] = TEST_ENV_API_KEY

    class _Resp:
        status_code = 403
        headers = {"x-fireworks-account-id": "acct_789"}

    with patch("eval_protocol.auth.requests.get", return_value=_Resp()):
        assert verify_api_key_and_get_account_id() is None


def test_get_account_id_resolves_via_verify_when_api_key_present():
    os.environ["FIREWORKS_API_KEY"] = TEST_ENV_API_KEY

    class _Resp:
        status_code = 200
        headers = {"X-Fireworks-Account-Id": "acct_456"}

    with patch("eval_protocol.auth.requests.get", return_value=_Resp()):
        assert get_fireworks_account_id() == "acct_456"
