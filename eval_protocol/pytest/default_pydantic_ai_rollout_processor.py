import asyncio
import logging
import types
from typing import List, Literal

from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.google import GoogleModel
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage
from pydantic_ai._utils import generate_tool_call_id
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import (
    ModelRequest,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.providers.openai import OpenAIProvider
from typing import Callable, Union

logger = logging.getLogger(__name__)


class PydanticAgentRolloutProcessor(RolloutProcessor):
    """Rollout processor for Pydantic AI agents. Mainly converts
    EvaluationRow.messages to and from Pydantic AI ModelMessage format."""

    def __init__(self, setup_agent: Union[Callable[..., Agent], Agent], usage_limits: UsageLimits = None):
        # dummy model used for its helper functions for processing messages
        self.util = OpenAIModel("dummy-model", provider=OpenAIProvider(api_key="dummy"))
        self.setup_agent = setup_agent
        self.usage_limits = usage_limits

    def _map_litellm_to_pydantic_ai(
        self, model_name: str
    ) -> Literal[
        "openai",
        "deepseek",
        "azure",
        "openrouter",
        "grok",
        "fireworks",
        "together",
    ]:
        mapping = {
            "fireworks_ai": "fireworks",
            "together_ai": "together",
            "xai": "grok",
            "azure_ai": "azure",
        }
        provider = model_name.split("/")[0]
        model_name = model_name.removeprefix(f"{provider}/")
        if provider in mapping:
            provider = mapping[provider]
        return provider, model_name

    def _map_litellm_to_pydantic_ai_model(self, model_name: str) -> Union[OpenAIModel, GoogleModel, AnthropicModel]:
        if model_name.startswith("anthropic/"):
            return AnthropicModel(
                model_name.removeprefix("anthropic/"),
            )
        elif model_name.startswith("google/"):
            return GoogleModel(
                model_name.removeprefix("google/"),
            )
        elif model_name.startswith("gemini/"):
            return GoogleModel(
                model_name.removeprefix("gemini/"),
            )
        provider, model_name = self._map_litellm_to_pydantic_ai(model_name)
        return OpenAIModel(
            model_name,
            provider=provider,
        )

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Create agent rollout tasks and return them for external handling."""

        max_concurrent = getattr(config, "max_concurrent_rollouts", 8) or 8
        semaphore = asyncio.Semaphore(max_concurrent)

        if isinstance(self.setup_agent, types.FunctionType):
            kwargs: dict[str, OpenAIModel | GoogleModel | AnthropicModel] = {}
            for agent, model_config in config.completion_params["model"].items():
                if "model" not in model_config:
                    raise ValueError(f"model_config for agent {agent} must contain a 'model' key")
                kwargs[agent] = self._map_litellm_to_pydantic_ai_model(model_config["model"])
            agent = self.setup_agent(**kwargs)
            model = None
        else:
            agent = self.setup_agent
            model = self._map_litellm_to_pydantic_ai_model(config.completion_params["model"])

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            """Process a single row with agent rollout."""
            model_messages = [self.convert_ep_message_to_pyd_message(m, row) for m in row.messages]
            response = await agent.run(
                message_history=model_messages, model=model, usage_limits=config.kwargs.get("usage_limits")
            )
            row.messages = await self.convert_pyd_message_to_ep_message(response.all_messages())
            return row

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                result = await process_row(r)
                return result

        # Create and return tasks for external handling
        tasks = [asyncio.create_task(_sem_wrapper(row)) for row in rows]
        return tasks

    async def convert_pyd_message_to_ep_message(self, messages: list[ModelMessage]) -> list[Message]:
        oai_messages: list[ChatCompletionMessageParam] = await self.util._map_messages(messages)
        return [Message(**m) for m in oai_messages]

    def convert_ep_message_to_pyd_message(self, message: Message, row: EvaluationRow) -> ModelMessage:
        if message.role == "assistant":
            type_adapter = TypeAdapter(ChatCompletionAssistantMessageParam)
            oai_message = type_adapter.validate_python(message)
            # Fix: Provide required finish_reason and index, and ensure created is int (timestamp)
            return self.util._process_response(
                ChatCompletion(
                    choices=[ChatCompletionChoice(message=oai_message, finish_reason="stop", index=0)],
                    object="chat.completion",
                    model="",
                    id="",
                    created=(
                        int(row.created_at.timestamp())
                        if hasattr(row.created_at, "timestamp")
                        else int(row.created_at)
                    ),
                )
            )
        elif message.role == "user":
            if isinstance(message.content, str):
                return ModelRequest(parts=[UserPromptPart(content=message.content)])
            elif isinstance(message.content, list):
                return ModelRequest(parts=[UserPromptPart(content=message.content[0].text)])
        elif message.role == "system":
            if isinstance(message.content, str):
                return ModelRequest(parts=[SystemPromptPart(content=message.content)])
            elif isinstance(message.content, list):
                return ModelRequest(parts=[SystemPromptPart(content=message.content[0].text)])
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
