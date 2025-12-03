import logging
import math
import asyncio
import inspect
from typing import Any, Callable, Literal, Optional, Sequence, List

try:
    import chz
    from tinker_cookbook import renderers, tokenizer_utils
    from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
    from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
    from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
    import tinker

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False
    # Dummy classes to avoid NameError when defining the class if imports fail
    # but we should probably raise an error if these are instantiated without dependencies
    RLDataset = object
    RLDatasetBuilder = object
    ProblemGroupBuilder = object
    SamplingClientEvaluator = object

from eval_protocol.adapters.base import BaseAdapter
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig

logger = logging.getLogger(__name__)


class EvalProtocolRLDataset(RLDataset):
    def __init__(
        self,
        adapter: BaseAdapter,
        row_converter: Callable[[Any, int], Optional[ProblemGroupBuilder]],
        batch_size: int,
        group_size: int,
        split: str = "train",
        limit: Optional[int] = None,
    ):
        if not TINKER_AVAILABLE:
            raise ImportError("tinker-cookbook is required to use EvalProtocolRLDataset")

        self.adapter = adapter
        self.row_converter = row_converter
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1

        logger.info(f"Fetching {limit if limit else 'all'} rows from adapter for split {split}...")
        self.rows = list(self.adapter.get_evaluation_rows(split=split, limit=limit))
        logger.info(f"Loaded {len(self.rows)} rows.")

    def get_batch(self, index: int) -> Sequence[ProblemGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.rows))

        batch_builders = []
        for i in range(batch_start, batch_end):
            row = self.rows[i]
            # row_converter should take the row and group_size and return a ProblemGroupBuilder
            builder = self.row_converter(row, self.group_size)
            if builder is not None:
                batch_builders.append(builder)

        return batch_builders

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)


if TINKER_AVAILABLE:

    class EvalProtocolEvaluator(SamplingClientEvaluator):
        def __init__(
            self,
            rows: List[EvaluationRow],
            eval_func: Callable[[EvaluationRow], EvaluationRow],
            rollout_processor_cls: Any,
            model_name: str,
            renderer_name: str,
            max_tokens: int = 512,
            temperature: float = 0.0,
        ):
            self.rows = rows

            # If the function is a dual_mode_wrapper (from @evaluation_test), unwrap it to get the raw function logic.
            # This avoids the overhead of the wrapper which is designed for pytest execution.
            if hasattr(eval_func, "_origin_func"):
                self.eval_func = eval_func._origin_func
            else:
                self.eval_func = eval_func

            self.rollout_processor_cls = rollout_processor_cls
            self.model_name = model_name
            self.renderer_name = renderer_name
            self.max_tokens = max_tokens
            self.temperature = temperature

        async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
            processor = self.rollout_processor_cls(
                sampling_client=sampling_client, model_name=self.model_name, renderer_name=self.renderer_name
            )
            processor.setup()

            # Config for rollout
            config = RolloutProcessorConfig(
                completion_params={
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                semaphore=asyncio.Semaphore(10),  # Concurrency limit
                mcp_config_path="",  # Not used
                steps=1,
                logger=None,  # Optional logger
                kwargs={},
            )

            # Run rollouts
            tasks = processor(self.rows, config)
            processed_rows = await asyncio.gather(*tasks)

            # Score
            scores = []
            for row in processed_rows:
                # Call the function logic (sync or async)
                res = self.eval_func(row)

                if inspect.isawaitable(res):
                    scored_row = await res
                else:
                    scored_row = res

                if scored_row.evaluation_result and scored_row.evaluation_result.score is not None:
                    scores.append(scored_row.evaluation_result.score)

            mean_score = sum(scores) / len(scores) if scores else 0.0
            return {"accuracy": mean_score}


def create_eval_protocol_dataset_builder(
    adapter_factory: Callable[[], BaseAdapter],
    row_converter: Callable[[Any, int, Any, Any], Optional[ProblemGroupBuilder]],
    convo_prefix_factory: Optional[Callable[[], list]] = None,
    train_limit: int = 1000,
    test_limit: int = 100,
) -> type:
    """
    Factory to create a specific RLDatasetBuilder class for a given adapter.
    """
    if not TINKER_AVAILABLE:
        return object

    @chz.chz
    class CustomBuilder(RLDatasetBuilder):
        batch_size: int
        model_name_for_tokenizer: str
        renderer_name: str
        group_size: int
        seed: int = 0

        async def __call__(self) -> tuple[RLDataset, RLDataset]:
            tokenizer = tokenizer_utils.get_tokenizer(self.model_name_for_tokenizer)
            renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

            # Create adapter
            adapter = adapter_factory()

            # Get convo prefix if needed
            convo_prefix = convo_prefix_factory() if convo_prefix_factory else None

            # Bind renderer and prefix to row converter if needed
            # We'll wrap the row_converter to inject renderer and prefix
            def bound_row_converter(row, g_size):
                return row_converter(row, g_size, renderer, convo_prefix)

            train_ds = EvalProtocolRLDataset(
                adapter=adapter,
                row_converter=bound_row_converter,
                batch_size=self.batch_size,
                group_size=self.group_size,
                split="train",
                limit=train_limit,
            )

            test_ds = EvalProtocolRLDataset(
                adapter=adapter,
                row_converter=bound_row_converter,
                batch_size=self.batch_size,
                group_size=self.group_size,
                split="test",
                limit=test_limit,
            )

            return (train_ds, test_ds)

    return CustomBuilder
