import asyncio
from typing import List, Optional

from mcp.types import TextContent
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessageToolCall,
    FunctionCall,
)

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.default_agent_rollout_processor import Agent


class NoOpLogger:
    def log(self, row: EvaluationRow) -> None:
        return None

    def read(self, row_id: Optional[str] = None) -> List[EvaluationRow]:
        return []


def test_tool_result_single_text_becomes_string():
    # Prepare a minimal evaluation row and agent
    row = EvaluationRow(messages=[Message(role="user", content="use the tool")])
    agent = Agent(model="dummy", row=row, config_path="", logger=NoOpLogger())

    # Single text content becomes a plain string
    single = [TextContent(type="text", text="single result")]
    formatted = agent._format_tool_message_content(single)
    assert isinstance(formatted, str)
    assert formatted == "single result"

    # Multiple text contents become a list of text parts
    multiple = [
        TextContent(type="text", text="first"),
        TextContent(type="text", text="second"),
    ]
    formatted_multi = agent._format_tool_message_content(multiple)
    assert isinstance(formatted_multi, list)
    assert [part["text"] for part in formatted_multi] == ["first", "second"]
