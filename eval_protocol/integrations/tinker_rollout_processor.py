import asyncio
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Union

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

try:
    import tinker
    from tinker_cookbook import renderers, tokenizer_utils

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

logger = logging.getLogger(__name__)


class TinkerRolloutProcessor(RolloutProcessor):
    """
    Rollout processor that uses a Tinker SamplingClient to generate responses.
    """

    def __init__(
        self,
        sampling_client: Optional[Any] = None,
        model_name: Optional[str] = None,
        renderer_name: str = "llama3",
    ) -> None:
        """
        Args:
            sampling_client: Pre-initialized tinker.SamplingClient. If None, one will be created using model_name.
            model_name: Name of the model to use (if sampling_client is None).
            renderer_name: Name of the renderer to use for formatting messages.
        """
        if not TINKER_AVAILABLE:
            raise ImportError("tinker-cookbook is required to use TinkerRolloutProcessor")

        self.sampling_client = sampling_client
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.renderer = None
        self.tokenizer = None

    def setup(self) -> None:
        """Setup resources."""
        if self.sampling_client is None:
            if self.model_name is None:
                raise ValueError("Either sampling_client or model_name must be provided")

            # Initialize Tinker service client
            # This assumes TINKER_API_KEY is set in env
            service_client = tinker.ServiceClient()
            self.sampling_client = service_client.create_sampling_client(base_model=self.model_name)

        # Initialize tokenizer and renderer
        # We need the model name to get the correct tokenizer.
        # If sampling_client was provided without model_name, we might need to infer it or require it.
        if self.model_name:
            self.tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        else:
            # Fallback or try to get from client if possible?
            # For now, require model_name even if client is passed, or use a default
            # But usually we want the renderer to match the model.
            # Let's assume Llama-3 tokenizer if not specified for now or raise error
            raise ValueError("model_name is required to initialize tokenizer/renderer")

        self.renderer = renderers.get_renderer(self.renderer_name, tokenizer=self.tokenizer)

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Generate rollout tasks using Tinker."""

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            if not row.messages:
                raise ValueError("Messages is empty")

            # Prepare prompt using renderer
            # Convert messages to Tinker ModelInput
            # We need to convert EvaluationRow messages (standard format) to the renderer's expected input
            # The renderer expects a list of dicts or objects with role/content
            # eval_protocol Message objects have role/content attributes, which should work if renderer supports objects
            # checking renderer code... it typically iterates and accesses keys or attributes.
            # Let's convert to dicts to be safe.

            convo = [
                {"role": m.role, "content": m.content}
                for m in row.messages
                if m.role in ["system", "user", "assistant"]
            ]

            prompt = self.renderer.build_generation_prompt(convo)

            # Prepare sampling params
            # Map config.completion_params to Tinker SamplingParams
            # Default values matching standard configs
            max_tokens = config.completion_params.get("max_tokens", 512)
            temperature = config.completion_params.get("temperature", 1.0)
            top_p = config.completion_params.get("top_p", 1.0)
            top_k = config.completion_params.get("top_k", -1)

            # Get stop sequences from renderer
            stop_sequences = self.renderer.get_stop_sequences()
            # Ensure stop_sequences is a list
            if stop_sequences is None:
                stop_sequences = []

            sampling_params = tinker.SamplingParams(
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                stop=stop_sequences,
            )

            # Call Tinker API
            try:
                sample_result = await self.sampling_client.sample_async(
                    prompt=prompt, num_samples=1, sampling_params=sampling_params
                )

                # Parse response
                # renderer.parse_response returns (Message, bool)
                sampled_tokens = sample_result.sequences[0].tokens
                message, parse_success = self.renderer.parse_response(sampled_tokens)

                if message:
                    assistant_content = message["content"]
                else:
                    assistant_content = ""

            except Exception as e:
                # Try to extract more info if '0' is not helpful
                error_details = str(e)
                if error_details == "0":
                    try:
                        error_details = f"Code: {e.code}, Message: {getattr(e, 'message', 'unknown')}"
                    except Exception as e2:
                        pass
                # Log full traceback for debugging
                tb_str = traceback.format_exc()
                logger.error(f"Tinker sampling failed: {error_details}\nTraceback:\n{tb_str}")
                assistant_content = ""  # Or handle error more gracefully
                # Could set status on row

            # Update row
            new_messages = list(row.messages) + [Message(role="assistant", content=assistant_content)]
            row.messages = new_messages
            row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time

            # Log usage (approximate since Tinker might not return usage stats in same format)
            # We can count tokens ourselves
            row.execution_metadata.usage = None  # Placeholder

            default_logger.log(row)
            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await process_row(r)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]
