"""Klavis judging and authorization checks that do not require live credentials."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval_protocol.mcp.mcp_multi_client import MCPMultiClient
from eval_protocol.models import EvaluationRow, Message
from tests import klavis_helpers as klavis
from tests.klavis_helpers import klavis_gmail_sandbox_dataset_adapter


@pytest.fixture
def judge_client(monkeypatch):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())))
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(klavis, "AsyncOpenAI", lambda **kwargs: context)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    return client.chat.completions.create


@pytest.mark.parametrize("response,score", [('{"score":1}', 1.0), ('{"score":0}', 0.0), ("bad json", 0.0)])
async def test_klavis_judge(judge_client, response, score):
    judge_client.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
    row = EvaluationRow(messages=[Message(role="assistant", content="five emails")], ground_truth="five emails")
    result = await klavis.judge_klavis_response(row)
    assert result.evaluation_result.score == score
    params = judge_client.call_args.kwargs
    assert params["model"] == klavis.MODEL
    assert "five emails" in params["messages"][-1]["content"]
    assert params["response_format"]["json_schema"]["schema"] == klavis.ResponseFormat.model_json_schema()


async def test_klavis_judge_service_error_fails(judge_client):
    judge_client.side_effect = RuntimeError("model unavailable")
    row = EvaluationRow(messages=[Message(role="assistant", content="answer")])
    with pytest.raises(RuntimeError, match="model unavailable"):
        await klavis.judge_klavis_response(row)


def test_klavis_authorization_configuration(monkeypatch):
    client = MCPMultiClient(config_path="tests/pytest/mcp_configurations/klavis_strata_mcp.json")
    config = client.config.mcpServers["klavis-strata"]
    assert config.url == "https://strata.klavis.ai/mcp/"
    monkeypatch.setenv("KLAVIS_AUTH_TOKEN", "test-token")
    assert client._process_headers({"Authorization": config.authorization}) == {"Authorization": "Bearer test-token"}
    monkeypatch.delenv("KLAVIS_AUTH_TOKEN")
    with pytest.raises(ValueError, match="KLAVIS_AUTH_TOKEN"):
        client._process_headers({"Authorization": config.authorization})


def test_klavis_gmail_dataset_adapter():
    initial = {"emails": []}
    expected = {"emails": [{"subject": "Hello"}]}
    rows = klavis_gmail_sandbox_dataset_adapter(
        [{"initialize_data": initial, "messages": "Send hello", "ground_truth": expected}]
    )
    assert len(rows) == 1
    assert rows[0].messages[-1].content == "Send hello"
    assert rows[0].ground_truth == expected
    assert rows[0].input_metadata.session_data == {"initialize_data": initial, "task": "Send hello"}
