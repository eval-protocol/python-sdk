import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, List, Optional, Union

from mcp.types import CallToolResult, TextContent
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionMessage, ChatCompletionToolParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.mcp.mcp_multi_client import MCPMultiClient
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import Dataset, RolloutProcessorConfig

logger = logging.getLogger(__name__)


class Agent:
    """
    A really simple agent that calls the model until no more tool calls are needed.
    """

    def __init__(self, model: str, row: EvaluationRow, config_path: str, logger: DatasetLogger):
        self.model = model
        self.evaluation_row: EvaluationRow = row
        self._policy = LiteLLMPolicy(model_id=model)
        self.mcp_client = MCPMultiClient(config_path=config_path) if config_path else None
        self.logger: DatasetLogger = logger

    async def setup(self):
        if self.mcp_client:
            await self.mcp_client.connect_to_servers()

    async def _get_tools(self) -> Optional[List[ChatCompletionToolParam]]:
        if self.evaluation_row.tools is None:
            self.evaluation_row.tools = await self.mcp_client.get_available_tools() if self.mcp_client else None
        return self.evaluation_row.tools

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
            tool_tasks: List[asyncio.Task[tuple[str, List[TextContent]]]] = []
            for tool_call in message.tool_calls:
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                tool_args_dict = json.loads(tool_args)

                # Create a task for each tool call
                task = self._execute_tool_call(tool_call_id, tool_name, tool_args_dict)
                tool_tasks.append(task)

            # Execute all tool calls in parallel
            tool_results = await asyncio.gather(*tool_tasks)

            # Add all tool results to messages (they will be in the same order as tool_calls)
            for tool_call, (tool_call_id, content) in zip(message.tool_calls, tool_results):
                tool_message_content = self._format_tool_message_content(content)
                self.append_message_and_log(
                    Message(role="tool", content=tool_message_content, tool_call_id=tool_call_id)
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

    def _format_tool_message_content(
        self, content: List[TextContent]
    ) -> Union[str, List[ChatCompletionContentPartTextParam]]:
        """Format tool result content for inclusion in a tool message.

        - If a single text item, return plain string per OpenAI semantics.
        - If multiple items, return a list of text parts.
        """
        if len(content) == 1 and isinstance(content[0], TextContent):
            return content[0].text
        return [ChatCompletionContentPartTextParam(text=c.text, type="text") for c in content]


class AgentRolloutProcessor(RolloutProcessor):
    """Agent rollout processor for tool-calling agents."""

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Create agent rollout tasks and return them for external handling."""

        max_concurrent = getattr(config, "max_concurrent_rollouts", 8) or 8
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row with agent rollout."""
            agent = Agent(
                model=row.input_metadata.completion_params["model"],
                row=row,
                config_path=config.mcp_config_path,
                logger=config.logger,
            )
            try:
                await agent.setup()
                await agent.call_agent()
                return agent.evaluation_row
            finally:
                if agent.mcp_client:
                    await agent.mcp_client.cleanup()

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await process_row(r)
                return result

        # Create and return tasks for external handling
        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks
