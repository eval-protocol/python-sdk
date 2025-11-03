import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, List, Optional, Union, Dict

from mcp.types import CallToolResult, TextContent
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionContentPartTextParam as OpenAIChatContentPart
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.mcp.mcp_multi_client import MCPMultiClient
from eval_protocol.models import EvaluationRow, Message, ChatCompletionContentPartTextParam
from openai.types import CompletionUsage
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import Dataset, RolloutProcessorConfig
from pydantic import BaseModel
from typing import Optional


class FunctionLike(BaseModel):
    name: Optional[str] = None
    parameters: Any = None


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
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    async def setup(self):
        if self.mcp_client:
            await self.mcp_client.connect_to_servers()

    async def _get_tools(self) -> Optional[List[dict[str, Any]]]:
        if self.evaluation_row.tools is None:
            if self.mcp_client:
                raw_tools = await self.mcp_client.get_available_tools()
                tools_dicts: List[dict[str, Any]] = []
                for t in raw_tools or []:
                    # Normalize any tool to dict shape expected by tests
                    tool_type = getattr(t, "type", None)
                    func = getattr(t, "function", None)
                    if isinstance(t, dict):
                        # Ensure function is dict-like; if it has .name/.parameters convert
                        f = t.get("function")
                        if f is not None and not isinstance(f, dict):
                            f_name = getattr(f, "name", None)
                            f_params = getattr(f, "parameters", None)
                            if f_params is not None and hasattr(f_params, "model_dump"):
                                f_params = f_params.model_dump()
                            func_obj = FunctionLike(name=f_name, parameters=f_params)
                            t = {"type": t.get("type", "function"), "function": func_obj}
                        elif isinstance(f, dict):
                            func_obj = FunctionLike(name=f.get("name"), parameters=f.get("parameters"))
                            t = {"type": t.get("type", "function"), "function": func_obj}
                        tools_dicts.append(t)
                        continue
                    # Construct a dict from object-like tool
                    name = getattr(func, "name", None)
                    params = getattr(func, "parameters", None)
                    if params is not None and hasattr(params, "model_dump"):
                        params_payload = params.model_dump()
                    elif isinstance(params, dict):
                        params_payload = params
                    else:
                        params_payload = {}
                    func_obj = FunctionLike(name=name, parameters=params_payload)
                    tools_dicts.append({"type": tool_type or "function", "function": func_obj})
                self.evaluation_row.tools = tools_dicts
            else:
                self.evaluation_row.tools = None
        return self.evaluation_row.tools

    @property
    def messages(self) -> list[Message]:
        return self.evaluation_row.messages

    def append_message_and_log(self, message: Message):
        self.messages.append(message)
        self.logger.log(self.evaluation_row)

    async def call_agent(self) -> Optional[Union[str, List[ChatCompletionContentPartTextParam]]]:
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
                task = asyncio.create_task(self._execute_tool_call(tool_call_id, tool_name, tool_args_dict))
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

    async def _call_model(self, messages: list[Message], tools: Optional[List[dict[str, Any]]]) -> Message:
        # Convert Message models to plain dicts for LLM call
        # Filter out fields that are not supported by OpenAI/LiteLLM APIs (e.g., weight, control_plane_step, reasoning_content)
        messages_payload: List[Dict[str, Any]] = [
            message.dump_mdoel_for_chat_completion_request()
            if hasattr(message, "dump_mdoel_for_chat_completion_request")
            else (message.model_dump() if hasattr(message, "model_dump") else message)  # type: ignore[misc]
            for message in messages
        ]
        # Normalize tool definitions into OpenAI-compatible dicts
        payload_tools: List[Dict[str, Any]] = []
        for tool in tools or []:
            if isinstance(tool, dict):
                fn = tool.get("function")
                if fn is not None and hasattr(fn, "model_dump"):
                    fn_payload = fn.model_dump()
                elif isinstance(fn, dict):
                    fn_payload = fn
                else:
                    # Best effort fallback
                    name = getattr(fn, "name", None)
                    params = getattr(fn, "parameters", None)
                    if params is not None and hasattr(params, "model_dump"):
                        params_payload = params.model_dump()
                    elif isinstance(params, dict):
                        params_payload = params
                    else:
                        params_payload = {}
                    fn_payload = {"name": name, "parameters": params_payload}
                payload_tools.append({"type": tool.get("type", "function"), "function": fn_payload})
            else:
                # Attribute-based fallback
                tool_type = getattr(tool, "type", "function")
                func = getattr(tool, "function", None)
                name = getattr(func, "name", None)
                params = getattr(func, "parameters", None)
                if params is not None and hasattr(params, "model_dump"):
                    params_payload = params.model_dump()
                elif isinstance(params, dict):
                    params_payload = params
                else:
                    params_payload = {}
                payload_tools.append({"type": tool_type, "function": {"name": name, "parameters": params_payload}})

        response = await self._policy._make_llm_call(messages=messages_payload, tools=payload_tools)

        self.usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self.usage["completion_tokens"] += response["usage"]["completion_tokens"]
        self.usage["total_tokens"] += response["usage"]["total_tokens"]

        # Coerce content to a string to align with our Message model type expectations
        raw_content = response["choices"][0]["message"].get("content")
        if isinstance(raw_content, list):
            content_for_model = "".join([getattr(p, "text", str(p)) for p in raw_content])
        else:
            content_for_model = raw_content
        return Message(
            role=response["choices"][0]["message"]["role"],
            content=content_for_model,
            tool_calls=response["choices"][0]["message"].get("tool_calls"),
        )

    async def _execute_tool_call(
        self, tool_call_id: str, tool_name: str, tool_args_dict: dict
    ) -> tuple[str, List[TextContent]]:
        """
        Execute a single tool call and return the tool_call_id and content.
        This method is designed to be used with asyncio.gather() for parallel execution.
        """
        assert self.mcp_client is not None, "MCP client is not initialized"
        tool_result = await self.mcp_client.call_tool(tool_name, tool_args_dict)
        # Accept string errors from client and normalize to text content
        content = self._get_content_from_tool_result(tool_result)  # type: ignore[arg-type]
        return tool_call_id, content

    def _get_content_from_tool_result(self, tool_result: CallToolResult | str) -> List[TextContent]:
        if isinstance(tool_result, str):
            return [TextContent(text=tool_result, type="text")]
        if getattr(tool_result, "structuredContent", None):
            return [TextContent(text=json.dumps(tool_result.structuredContent), type="text")]
        normalized: List[TextContent] = []
        for content in getattr(tool_result, "content", []) or []:
            if isinstance(content, TextContent):
                normalized.append(content)
            else:
                text_val = getattr(content, "text", str(content))
                normalized.append(TextContent(text=str(text_val), type="text"))
        return normalized

    def _format_tool_message_content(
        self, content: List[TextContent]
    ) -> Union[str, List[ChatCompletionContentPartTextParam]]:
        """Format tool result content for inclusion in a tool message.

        - If a single text item, return plain string per OpenAI semantics.
        - If multiple items, return a list of text parts.
        """
        if len(content) == 1 and isinstance(content[0], TextContent):
            return content[0].text
        # Build our SDK's ChatCompletionContentPartTextParam instances, not OpenAI types
        return [ChatCompletionContentPartTextParam(text=c.text, type="text") for c in content]


class AgentRolloutProcessor(RolloutProcessor):
    """Agent rollout processor for tool-calling agents."""

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Create agent rollout tasks and return them for external handling."""

        semaphore = config.semaphore

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row with agent rollout."""
            start_time = time.perf_counter()

            agent = Agent(
                model=row.input_metadata.completion_params["model"],
                row=row,
                config_path=config.mcp_config_path,
                logger=config.logger,
            )
            try:
                await agent.setup()
                await agent.call_agent()

                agent.evaluation_row.execution_metadata.usage = CompletionUsage(
                    prompt_tokens=agent.usage["prompt_tokens"],
                    completion_tokens=agent.usage["completion_tokens"],
                    total_tokens=agent.usage["total_tokens"],
                )

                agent.evaluation_row.execution_metadata.duration_seconds = time.perf_counter() - start_time

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
