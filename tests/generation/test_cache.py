import json
import os
import shutil
from unittest.mock import patch  # Added import for patch

import pytest
from omegaconf import DictConfig

from eval_protocol.generation.cache import ResponseCache


@pytest.fixture(scope="function")
def cache_manager(tmp_path):
    """Fixture to set up and tear down the cache directory for each test."""
    # Use pytest's tmp_path for unique temporary directories per test
    test_cache_dir = tmp_path / "test_cache"
    test_cache_dir.mkdir(exist_ok=True)

    cache_config_enabled = DictConfig(
        {
            "cache_dir": str(test_cache_dir),
            # enabled is not a direct param of ResponseCache, but used by pipeline.
            # ResponseCache itself is always "enabled" if cache_dir is valid.
        }
    )
    cache = ResponseCache(cache_config_enabled)
    yield cache  # Provide the ResponseCache instance

    # Cleanup is handled automatically by pytest's tmp_path


def test_generate_key_includes_all_params(cache_manager: ResponseCache):
    key1 = cache_manager._generate_key(
        sample_id="s1",
        system_prompt="sys",
        user_query="uq1",
        model_name="m1",
        temperature=0.0,
        top_p=0.9,
        top_k=10,
        min_p=0.1,
        max_tokens=100,
        reasoning_effort="low",
    )
    key2 = cache_manager._generate_key(
        sample_id="s1",
        system_prompt="sys",
        user_query="uq1",
        model_name="m1",
        temperature=0.0,
        top_p=0.9,
        top_k=10,
        min_p=0.1,
        max_tokens=100,
        reasoning_effort="high",  # Different reasoning_effort
    )
    key3 = cache_manager._generate_key(
        sample_id="s1",
        system_prompt="sys",
        user_query="uq1",
        model_name="m1",
        temperature=0.0,
        top_p=0.9,
        top_k=10,
        min_p=0.1,
        max_tokens=100,
        reasoning_effort=None,  # None reasoning_effort
    )
    key4 = cache_manager._generate_key(
        sample_id="s2",
        system_prompt="sys",
        user_query="uq1",
        model_name="m1",  # Different sample_id
        temperature=0.0,
        top_p=0.9,
        top_k=10,
        min_p=0.1,
        max_tokens=100,
        reasoning_effort="low",
    )

    assert key1 != key2, "reasoning_effort should change the key"
    assert key1 != key3, "None reasoning_effort should change the key"
    assert key2 != key3
    assert key1 != key4, "sample_id should change the key"


def test_put_and_get_with_reasoning_effort(cache_manager: ResponseCache):
    params1 = {
        "sample_id": "sample1",
        "system_prompt": "System",
        "user_query": "Query",
        "model_name": "modelX",
        "temperature": 0.0,
        "response": "Response RE low",
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "max_tokens": 200,
        "reasoning_effort": "low",
    }
    params2 = {**params1, "reasoning_effort": "high", "response": "Response RE high"}
    params3 = {**params1, "reasoning_effort": None, "response": "Response RE none"}

    cache_manager.put(**params1)
    cache_manager.put(**params2)
    cache_manager.put(**params3)

    # Test get for params1
    retrieved1 = cache_manager.get(
        sample_id=params1["sample_id"],
        system_prompt=params1["system_prompt"],
        user_query=params1["user_query"],
        model_name=params1["model_name"],
        temperature=params1["temperature"],
        top_p=params1["top_p"],
        top_k=params1["top_k"],
        min_p=params1["min_p"],
        max_tokens=params1["max_tokens"],
        reasoning_effort=params1["reasoning_effort"],
    )
    assert retrieved1 == "Response RE low"

    # Test get for params2
    retrieved2 = cache_manager.get(
        sample_id=params2["sample_id"],
        system_prompt=params2["system_prompt"],
        user_query=params2["user_query"],
        model_name=params2["model_name"],
        temperature=params2["temperature"],
        top_p=params2["top_p"],
        top_k=params2["top_k"],
        min_p=params2["min_p"],
        max_tokens=params2["max_tokens"],
        reasoning_effort=params2["reasoning_effort"],
    )
    assert retrieved2 == "Response RE high"

    # Test get for params3
    retrieved3 = cache_manager.get(
        sample_id=params3["sample_id"],
        system_prompt=params3["system_prompt"],
        user_query=params3["user_query"],
        model_name=params3["model_name"],
        temperature=params3["temperature"],
        top_p=params3["top_p"],
        top_k=params3["top_k"],
        min_p=params3["min_p"],
        max_tokens=params3["max_tokens"],
        reasoning_effort=params3["reasoning_effort"],
    )
    assert retrieved3 == "Response RE none"


def test_cache_non_zero_temperature(cache_manager: ResponseCache):
    params = {
        "sample_id": "temp_test",
        "system_prompt": "Sys",
        "user_query": "Q",
        "model_name": "m_temp",
        "temperature": 0.7,
        "response": "Non-deterministic",
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "max_tokens": 100,
        "reasoning_effort": "low",
    }
    cache_manager.put(**params)

    retrieved = cache_manager.get(
        sample_id=params["sample_id"],
        system_prompt=params["system_prompt"],
        user_query=params["user_query"],
        model_name=params["model_name"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        min_p=params["min_p"],
        max_tokens=params["max_tokens"],
        reasoning_effort=params["reasoning_effort"],
    )
    assert (
        retrieved is None
    ), "Should not cache or retrieve for non-zero temperature by default"


def test_cache_disabled_if_dir_creation_fails(tmp_path):
    # This test needs to mock os.makedirs to raise an OSError
    unique_dir_1 = tmp_path / "some_uncreatable_dir"
    unique_dir_2 = tmp_path / "another_uncreatable_dir"

    with patch("os.makedirs", side_effect=OSError("Test OS Error")):
        with patch(
            "logging.Logger.error"
        ) as mock_log_error:  # Check that error is logged
            cache_config_fail = DictConfig({"cache_dir": str(unique_dir_1)})
            cache = ResponseCache(cache_config_fail)
            assert cache.cache_dir is None  # Caching should be disabled
            mock_log_error.assert_called_once()  # Ensure the error was logged

    # Double check that put/get do nothing if cache_dir is None
    cache_config_fail = DictConfig({"cache_dir": str(unique_dir_2)})
    with patch("os.makedirs", side_effect=OSError("Test OS Error")):
        cache = ResponseCache(cache_config_fail)  # cache.cache_dir will be None

    params = {
        "sample_id": "fail_test",
        "system_prompt": "S",
        "user_query": "Q",
        "model_name": "m_fail",
        "temperature": 0.0,
        "response": "R",
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "max_tokens": 50,
        "reasoning_effort": "none",
    }
    cache.put(**params)  # Should not raise error, should do nothing

    # Check that no file was created (indirectly, by trying to get)
    # This is a bit of a weak check, but direct file system check is harder with mocks
    retrieved = cache.get(
        sample_id=params["sample_id"],
        system_prompt=params["system_prompt"],
        user_query=params["user_query"],
        model_name=params["model_name"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        min_p=params["min_p"],
        max_tokens=params["max_tokens"],
        reasoning_effort=params["reasoning_effort"],
    )
    assert retrieved is None
