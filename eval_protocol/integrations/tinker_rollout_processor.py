import asyncio
import logging
import time
import traceback
from typing import Any, List, Optional

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow
from eval_protocol.integrations.tinker_utils import (
    TINKER_AVAILABLE,
    build_tinker_renderer,
    normalize_eval_protocol_messages,
    tinker_message_to_eval_protocol_message,
)
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

try:
    import tinker
except ImportError:
    tinker = None  # type: ignore[assignment]

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

        if self.model_name:
            self.tokenizer, self.renderer = build_tinker_renderer(
                model_name=self.model_name,
                renderer_name=self.renderer_name,
            )
        else:
            raise ValueError("model_name is required to initialize tokenizer/renderer")

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Generate rollout tasks using Tinker."""

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            if not row.messages:
                raise ValueError("Messages is empty")

            convo = normalize_eval_protocol_messages(row.messages)
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

                assistant_message = (
                    tinker_message_to_eval_protocol_message(message)
                    if message
                    else tinker_message_to_eval_protocol_message({"role": "assistant", "content": ""})
                )

            except Exception as e:
                error_details = str(e)
                if error_details == "0":
                    try:
                        error_details = f"Code: {e.code}, Message: {getattr(e, 'message', 'unknown')}"
                    except Exception:
                        pass
                tb_str = traceback.format_exc()
                logger.error(f"Tinker sampling failed: {error_details}\nTraceback:\n{tb_str}")
                assistant_message = tinker_message_to_eval_protocol_message(
                    {"role": "assistant", "content": ""}
                )

            new_messages = list(row.messages) + [assistant_message]
            row.messages = new_messages
            row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time

            row.execution_metadata.usage = None  # Placeholder

            default_logger.log(row)
            return row

        semaphore = config.semaphore

        async def _sem_wrapper(r: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await process_row(r)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]
