import json
import os
from typing import Any, List, Optional

from mcp.types import CallToolResult
from openai.types.chat import ChatCompletionMessage, ChatCompletionToolParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.mcp.mcp_multi_client import MCPMultiClient
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import RolloutProcessorConfig


class Agent:
    """
    A really simple agent that calls the model until no more tool calls are needed.
    """

    def __init__(self, model: str, initial_messages: list[Message], config_path: str):
        self.model = model
        self.messages: list[Message] = initial_messages
        self._policy = LiteLLMPolicy(model_id=model)
        self.mcp_client = MCPMultiClient(config_path=config_path) if config_path else None

    async def setup(self):
        if self.mcp_client:
            await self.mcp_client.connect_to_servers()

    async def call_agent(self) -> str:
        """
        Call the assistant with the user query.
        """
        tools = await self.mcp_client.get_available_tools() if self.mcp_client else None

        message = await self._call_model(self.messages, tools)
        self.messages.append(message)
        if message["tool_calls"]:
            for tool_call in message["tool_calls"]:
                tool_call_id = tool_call["id"]
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                tool_args_dict = json.loads(tool_args)
                tool_result = await self.mcp_client.call_tool(tool_name, tool_args_dict)
                content = self._get_content_from_tool_result(tool_result)
                self.messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call_id,
                    }
                )
        return message["content"]

    async def _call_model(
        self, messages: list[Message], tools: Optional[list[ChatCompletionToolParam]]
    ) -> ChatCompletionMessage:
        messages = [message.model_dump() if hasattr(message, "model_dump") else message for message in messages]
        response = await self._policy._make_llm_call(
            messages=messages,
            tools=tools,
        )
        return response["choices"][0]["message"]

    def _get_content_from_tool_result(self, tool_result: CallToolResult) -> str:
        if tool_result.structuredContent:
            return json.dumps(tool_result.structuredContent)
        if len(tool_result.content) > 1:
            raise NotImplementedError("Multiple content is not supported yet")
        first_content = tool_result.content[0]
        if first_content.type != "text":
            raise NotImplementedError("Non-text content is not supported yet")
        return first_content.text


async def default_agent_rollout_processor(row: EvaluationRow, config: RolloutProcessorConfig) -> List[EvaluationRow]:
    agent = Agent(model=config.model, initial_messages=config.initial_messages, config_path=config.mcp_config_path)
    await agent.setup()
    await agent.call_agent()
    return [EvaluationRow(messages=agent.messages)]
