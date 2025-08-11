import asyncio
import json
import os
from typing import Any, List, Optional, Union

from mcp.types import CallToolResult, TextContent
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionMessage, ChatCompletionToolParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.mcp.mcp_multi_client import MCPMultiClient
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import Dataset, RolloutProcessorConfig


class Agent:
    """
    A really simple agent that calls the model until no more tool calls are needed.
    """

    def __init__(self, model: str, row: EvaluationRow, config_path: str, logger: DatasetLogger):
        self.model = model
        self.evaluation_row: EvaluationRow = row
        self._policy = LiteLLMPolicy(model_id=model)
        self.mcp_client = MCPMultiClient(config_path=config_path) if config_path else None
        self.tools: Union[List[ChatCompletionToolParam], NotGiven] = NOT_GIVEN
        self.logger: DatasetLogger = logger

    async def setup(self):
        if self.mcp_client:
            await self.mcp_client.connect_to_servers()

    async def _get_tools(self) -> Optional[List[ChatCompletionToolParam]]:
        if self.tools is NOT_GIVEN:
            self.tools = await self.mcp_client.get_available_tools() if self.mcp_client else None
        return self.tools

    @property
    def messages(self) -> list[Message]:
        return self.evaluation_row.messages

    def append_message_and_log(self, message: Message):
        self.messages.append(message)
        self.logger.log(self.evaluation_row)

    async def call_agent(self) -> str:
        """
        Call the assistant with the user query.
        """
        tools = await self._get_tools() if self.mcp_client else None

        message = await self._call_model(self.messages, tools)
        self.append_message_and_log(message)
        if message.tool_calls:
            # Create tasks for all tool calls to run them in parallel
            tool_tasks = []
            for tool_call in message.tool_calls:
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                tool_args_dict = json.loads(tool_args)

                # Create a task for each tool call
                task = self._execute_tool_call(tool_call_id, tool_name, tool_args_dict)
                tool_tasks.append(task)

            # Execute all tool calls in parallel
            tool_results: List[tuple[str, List[TextContent]]] = await asyncio.gather(*tool_tasks)

            # Add all tool results to messages (they will be in the same order as tool_calls)
            for tool_call, (tool_call_id, content) in zip(message.tool_calls, tool_results):
                self.append_message_and_log(
                    Message(
                        role="tool",
                        content=[
                            ChatCompletionContentPartTextParam(text=content.text, type="text") for content in content
                        ],
                        tool_call_id=tool_call_id,
                    )
                )
            return await self.call_agent()
        return message.content

    async def _call_model(self, messages: list[Message], tools: Optional[list[ChatCompletionToolParam]]) -> Message:
        messages = [message.model_dump() if hasattr(message, "model_dump") else message for message in messages]
        tools = [{"function": tool["function"].model_dump(), "type": "function"} for tool in tools] if tools else []
        response = await self._policy._make_llm_call(messages=messages, tools=tools)
        return Message(
            role=response["choices"][0]["message"]["role"],
            content=response["choices"][0]["message"]["content"],
            tool_calls=response["choices"][0]["message"]["tool_calls"],
        )

    async def _execute_tool_call(
        self, tool_call_id: str, tool_name: str, tool_args_dict: dict
    ) -> tuple[str, List[TextContent]]:
        """
        Execute a single tool call and return the tool_call_id and content.
        This method is designed to be used with asyncio.gather() for parallel execution.
        """
        tool_result = await self.mcp_client.call_tool(tool_name, tool_args_dict)
        content = self._get_content_from_tool_result(tool_result)
        return tool_call_id, content

    def _get_content_from_tool_result(self, tool_result: CallToolResult) -> List[TextContent]:
        if tool_result.structuredContent:
            return [TextContent(text=json.dumps(tool_result.structuredContent), type="text")]
        if not all(isinstance(content, TextContent) for content in tool_result.content):
            raise NotImplementedError("Non-text content is not supported yet")
        return tool_result.content


async def default_agent_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> List[EvaluationRow]:
    dataset: Dataset = []
    for row in rows:
        agent = Agent(model=config.model, row=row, config_path=config.mcp_config_path, logger=config.logger)
        await agent.setup()
        await agent.call_agent()
        dataset.append(agent.evaluation_row)
        if agent.mcp_client:
            await agent.mcp_client.cleanup()
    return dataset
