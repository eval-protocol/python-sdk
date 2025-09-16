# pyright: reportPrivateUsage=false

import asyncio
from collections.abc import Callable
import logging
import time
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits
from typing_extensions import override
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from pydantic import TypeAdapter
from pydantic_ai import Agent, ModelSettings
from pydantic_ai._utils import generate_tool_call_id
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import (
    ModelRequest,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


class PydanticAgentRolloutProcessor(RolloutProcessor):
    """Rollout processor for Pydantic AI agents. Mainly converts
    EvaluationRow.messages to and from Pydantic AI ModelMessage format."""

    def __init__(
        self,
        agent_factory: Callable[[RolloutProcessorConfig], Agent],
        usage_limits: UsageLimits | None = None,
    ):
        # dummy model used for its helper functions for processing messages
        self._util: OpenAIChatModel = OpenAIChatModel("dummy-model", provider=OpenAIProvider(api_key="dummy"))
        self._setup_agent = agent_factory

    @override
    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        """Create agent rollout tasks and return them for external handling."""

        semaphore = config.semaphore
        agent = self._setup_agent(config)

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row with agent rollout."""
            start_time = time.perf_counter()

            tools = []
            for toolset in agent.toolsets:
                if isinstance(toolset, FunctionToolset):
                    for _, tool in toolset.tools.items():
                        tool_dict = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "parameters": tool.function_schema.json_schema,
                            },
                        }
                        if tool.description:
                            tool_dict["function"]["description"] = tool.description
                        tools.append(tool_dict)
            row.tools = tools

            model_messages = [self.convert_ep_message_to_pyd_message(m, row) for m in row.messages]
            settings = self.construct_model_settings(agent, row)
            response = await agent.run(
                message_history=model_messages, usage_limits=config.kwargs.get("usage_limits"), model_settings=settings
            )
            row.messages = await self.convert_pyd_message_to_ep_message(response.all_messages())

            # TODO: pydantic ai accumulates usage info across all models in multi-agent setup, so this simple tracking doesn't work for cost. to discuss with @dphuang2 when he's back.
            # usage_info = response.usage()
            # row.execution_metadata.usage = CompletionUsage(
            #     prompt_tokens=usage_info.request_tokens or 0,
            #     completion_tokens=usage_info.response_tokens or 0,
            #     total_tokens=usage_info.total_tokens or 0,
            # )

            row.execution_metadata.duration_seconds = time.perf_counter() - start_time

            return row

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await process_row(r)
                return result

        # Create and return tasks for external handling
        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks

    async def convert_pyd_message_to_ep_message(self, messages: list[ModelMessage]) -> list[Message]:
        oai_messages: list[ChatCompletionMessageParam] = await self._util._map_messages(messages)
        return [Message(**m) for m in oai_messages]  # pyright: ignore[reportArgumentType]

    def construct_model_settings(self, agent: Agent, row: EvaluationRow) -> ModelSettings:
        model = agent.model
        settings = None
        if model and not isinstance(model, str) and model.settings:
            # We must copy model settings to avoid concurrency issues by modifying the same object in-place
            settings = model.settings.copy()
        if settings is None:
            settings = ModelSettings()
        settings["extra_body"] = settings.get("extra_body", {})
        extra_body = settings["extra_body"]

        # Only store metadata for ResponsesModel, not for ChatModel
        if isinstance(extra_body, dict) and isinstance(model, OpenAIResponsesModel):
            extra_body["metadata"] = settings.get("metadata", {})
            extra_body["metadata"]["row_id"] = row.input_metadata.row_id
            extra_body["metadata"]["invocation_id"] = row.execution_metadata.invocation_id
            extra_body["metadata"]["rollout_id"] = row.execution_metadata.rollout_id
            extra_body["metadata"]["run_id"] = row.execution_metadata.run_id
            extra_body["metadata"]["experiment_id"] = row.execution_metadata.experiment_id

        return settings

    def convert_ep_message_to_pyd_message(self, message: Message, row: EvaluationRow) -> ModelMessage:
        if message.role == "assistant":
            type_adapter = TypeAdapter(ChatCompletionMessage)
            oai_message = type_adapter.validate_python(message)
            # Fix: Provide required finish_reason and index, and ensure created is int (timestamp)
            return self._util._process_response(
                ChatCompletion(
                    choices=[ChatCompletionChoice(message=oai_message, finish_reason="stop", index=0)],
                    object="chat.completion",
                    model="",
                    id="",
                    created=int(row.created_at.timestamp()),
                )
            )
        elif message.role == "user":
            if isinstance(message.content, str):
                return ModelRequest(parts=[UserPromptPart(content=message.content)])
            elif isinstance(message.content, list):
                return ModelRequest(parts=[UserPromptPart(content=message.content[0].text)])
            else:
                raise ValueError(f"Unsupported content type for user message: {type(message.content)}")
        elif message.role == "system":
            if isinstance(message.content, str):
                return ModelRequest(parts=[SystemPromptPart(content=message.content)])
            elif isinstance(message.content, list):
                return ModelRequest(parts=[SystemPromptPart(content=message.content[0].text)])
            else:
                raise ValueError(f"Unsupported content type for system message: {type(message.content)}")
        elif message.role == "tool":
            return ModelRequest(
                parts=[
                    ToolReturnPart(
                        content=message.content,
                        tool_name="",
                        tool_call_id=message.tool_call_id or generate_tool_call_id(),
                    )
                ]
            )
        raise ValueError(f"Unknown role: {message.role}")
