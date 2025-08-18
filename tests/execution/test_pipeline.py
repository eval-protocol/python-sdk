import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from omegaconf import DictConfig, OmegaConf

from eval_protocol.execution.pipeline import EvaluationPipeline
from eval_protocol.generation.cache import ResponseCache
from eval_protocol.generation.clients import (  # For type hinting and mocking
    FireworksModelClient,
    GenerationResult,  # Import GenerationResult
)
from eval_protocol.models import EvaluateResult, Message, MetricResult


# Minimal valid config for pipeline initialization
@pytest.fixture
def minimal_pipeline_cfg():
    return OmegaConf.create(
        {
            "generation": {
                "enabled": True,
                "model_name": "test-model",
                "temperature": 0.0,
                "max_tokens": 50,
                "top_p": 0.9,
                "top_k": 10,
                "min_p": 0.1,
                "reasoning_effort": "test_re",  # Include reasoning_effort
                "cache": {
                    "enabled": True,  # Add the missing 'enabled' key
                    "cache_dir": ".test_pipeline_cache",
                },
                "api_params": {"max_retries": 1, "max_concurrent_requests": 1},
            },
            "dataset": {  # Required for instantiation, even if not directly used in some tests
                "_target_": "eval_protocol.datasets.loader.load_and_process_dataset",  # Dummy target
                "source_type": "jsonl",
                "path_or_name": "dummy.jsonl",
            },
            "reward": {
                "function_path": "tests.mocks.mock_reward_function.mock_reward_func",  # Path to a mock
            },
            "evaluation_params": {"limit_samples": 1},
            "logging_params": {"batch_log_interval": 1},
            "output": {"results_file": "test_results.jsonl"},
            "hydra_output_dir": ".test_pipeline_outputs",  # For resolving relative output paths
        }
    )


@pytest.fixture
def mock_model_client():
    client = AsyncMock(spec=FireworksModelClient)
    client.model_name = "test-model"
    client.temperature = 0.0
    client.top_p = 0.9
    client.top_k = 10
    client.min_p = 0.1
    client.max_tokens = 50
    client.reasoning_effort = "test_re"  # Ensure this attribute exists
    client.generate = AsyncMock(return_value="Generated response")
    return client


@pytest.fixture
def mock_cache():
    cache = MagicMock(spec=ResponseCache)
    cache.get = MagicMock(return_value=None)  # Default to cache miss
    cache.put = MagicMock()
    return cache


@pytest.fixture
def mock_reward_function():
    # Returns a MetricResult with score 1.0
    return MagicMock(
        return_value=EvaluateResult(
            score=1.0,
            reason="Mock success",
            metrics={"test_metric": MetricResult(score=1.0, reason="test")},
        )
    )


@pytest.fixture
def mock_dataset():
    # Mock the dataset returned by hydra.utils.instantiate
    dataset_mock = MagicMock()
    dataset_mock.__len__.return_value = 1
    dataset_mock.__getitem__.return_value = {
        "id": "sample0",
        "user_query": "Test query",
        "ground_truth_for_eval": "Test GT",
    }
    return dataset_mock


@pytest.mark.asyncio
@patch("eval_protocol.execution.pipeline.FireworksModelClient", autospec=True)  # Mock at source of import
@patch("eval_protocol.execution.pipeline.ResponseCache", autospec=True)
@patch("eval_protocol.execution.pipeline.load_reward_function")
@patch("hydra.utils.instantiate")  # Mock hydra's instantiate for dataset loading
async def test_pipeline_passes_reasoning_effort_to_cache(
    mock_instantiate: MagicMock,
    mock_load_reward: MagicMock,
    MockResponseCache: MagicMock,
    MockFireworksModelClient: MagicMock,
    minimal_pipeline_cfg: DictConfig,
    mock_dataset: MagicMock,
    mock_reward_function: MagicMock,
):
    # Setup mocks
    mock_instantiate.return_value = mock_dataset  # For dataset loading
    mock_load_reward.return_value = mock_reward_function

    mock_fireworks_instance = MockFireworksModelClient.return_value
    mock_fireworks_instance.model_name = "test-model"
    mock_fireworks_instance.temperature = 0.0
    mock_fireworks_instance.top_p = 0.9
    mock_fireworks_instance.top_k = 10
    mock_fireworks_instance.min_p = 0.1
    mock_fireworks_instance.max_tokens = 50
    mock_fireworks_instance.reasoning_effort = "test_re"  # Critical: ensure mock client has this
    mock_fireworks_instance.generate = AsyncMock(
        return_value=GenerationResult(content="Generated via mock client")  # Return GenerationResult
    )

    mock_cache_instance = MockResponseCache.return_value
    mock_cache_instance.get = MagicMock(return_value=None)  # Ensure cache miss to trigger put
    mock_cache_instance.put = MagicMock()

    # Patch get_fireworks_api_key as it's called in pipeline init
    with patch(
        "eval_protocol.execution.pipeline.get_fireworks_api_key",
        return_value="fake_key",
    ):
        pipeline = EvaluationPipeline(minimal_pipeline_cfg)

    # Manually assign the mocked client and cache if constructor doesn't use the class mocks directly
    # This is often needed if the instances are created deep inside.
    # However, our patches should ensure the constructor gets the mocked classes.
    # If issues, uncomment and adjust:
    # pipeline.model_client = mock_fireworks_instance
    # pipeline.cache = mock_cache_instance
    # pipeline.reward_function = mock_reward_function

    # Run the pipeline (or a relevant part like _process_single_sample if easier)
    # For a full run, we also need aiohttp.ClientSession
    async with aiohttp.ClientSession() as session:
        # We'll test _process_single_sample directly to isolate cache interaction
        sample_data = mock_dataset[0]
        await pipeline._process_single_sample(sample_data, session)

    # Assertions for cache.put
    # cache.get is not called in this path because generation is enabled
    # and cache.get in the pipeline is for when generation is disabled.
    # The model client itself might use a cache.get, but that's not what this mock_cache_instance is for.
    mock_cache_instance.put.assert_called_once()
    put_args, put_kwargs = mock_cache_instance.put.call_args
    assert put_kwargs.get("reasoning_effort") == "test_re"
    assert put_kwargs.get("model_name") == "test-model"  # sanity check

    # Ensure the mock client's generate was called
    mock_fireworks_instance.generate.assert_called_once()
