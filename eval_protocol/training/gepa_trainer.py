import asyncio
from typing import Any, Dict, List, Literal

import dspy
from dspy.clients.lm import LM
from dspy.primitives import Module, Example
from dspy.teleprompt.gepa.gepa import GEPA
from gepa.core.adapter import ProposalFn
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector

from eval_protocol.models import EPParameters, EvaluationRow, Message
from eval_protocol.pytest.types import TestFunction, RolloutProcessorConfig
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.pytest.execution import execute_pytest
from eval_protocol.dataset_logger import default_logger
from eval_protocol.training.trainer import Trainer
from eval_protocol.training.utils import build_ep_parameters_from_test
from eval_protocol.training.gepa_utils import (
    ep_test_to_gepa_metric,
    create_single_turn_program,
    configure_dspy_lm,
    extract_system_prompt_from_rows,
    evaluation_rows_to_dspy_examples,
    train_val_test_split,
    DSPyModuleType,
    DSPyModuleFactory,
)


class GEPATrainer(Trainer):
    """
    High-level entrypoint for running GEPA-style training against an existing
    `@evaluation_test`-decorated function.

    This trainer:
    1. Extracts configuration from the @evaluation_test decorator
    2. Creates a DSPy ChainOfThought program (mirrors SingleTurnRolloutProcessor)
    3. Converts the EP dataset to DSPy format
    4. Uses EP's test function as the GEPA metric
    5. Runs GEPA optimization to find the best system prompt

    The optimized system prompt can then be used with EP's rollout processor
    for final evaluation.
    """

    def __init__(
        self,
        test_fn: TestFunction,
        *,
        # Dataset splitting
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        # DSPy signature configuration
        input_field: str = "problem",
        output_field: str = "answer",
        input_desc: str | None = None,
        output_desc: str | None = None,
        # DSPy module configuration
        module_type: DSPyModuleType | str = DSPyModuleType.CHAIN_OF_THOUGHT,
        module_factory: DSPyModuleFactory | None = None,
        # Custom program (overrides automatic creation)
        program: Module | None = None,
    ) -> None:
        """
        Args:
            test_fn: The `@evaluation_test`-decorated function defining the eval.
            train_ratio: Proportion of data for training (default 0.8)
            val_ratio: Proportion of data for validation (default 0.1)
            seed: Random seed for dataset splitting
            input_field: Name of the input field in DSPy signature (default: "problem")
            output_field: Name of the output field in DSPy signature (default: "answer")
            input_desc: Optional description for the input field
            output_desc: Optional description for the output field
            module_type: Which DSPy module to use:
                - PREDICT: Simple input → output
                - CHAIN_OF_THOUGHT: Adds reasoning (default, good for complex tasks)
                - PROGRAM_OF_THOUGHT: Generates code to solve problems
            module_factory: Custom factory to create DSPy module. Overrides module_type.
            program: Pre-built DSPy module. If provided, skips automatic creation.

        Examples:
            # Default: ChainOfThought for math
            trainer = GEPATrainer(test_fn)

            # Simple classification
            trainer = GEPATrainer(
                test_fn,
                input_field="text",
                output_field="label",
                module_type=DSPyModuleType.PREDICT,
            )

            # Custom DSPy module
            my_program = dspy.ChainOfThought(MySignature)
            trainer = GEPATrainer(test_fn, program=my_program)
        """
        super().__init__(test_fn)
        self.ep_params: EPParameters = build_ep_parameters_from_test(test_fn)

        # Store configuration
        self._input_field = input_field
        self._output_field = output_field
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed

        # Configure DSPy to use the same LLM as EP
        configure_dspy_lm(self.ep_params)

        # Wrap the EP test function as a GEPA metric (with configured field names)
        self.metric = ep_test_to_gepa_metric(test_fn, input_field, output_field)

        # Load and split the dataset
        self._rows: List[EvaluationRow] = self._load_dataset()
        train_rows, val_rows, test_rows = train_val_test_split(
            self._rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        # Store original EvaluationRow objects for later use in evaluate_with_ep
        self._train_rows: List[EvaluationRow] = train_rows
        self._val_rows: List[EvaluationRow] = val_rows
        self._test_rows: List[EvaluationRow] = test_rows

        # Extract the system prompt from the dataset (this is what GEPA will optimize!)
        self._initial_system_prompt = extract_system_prompt_from_rows(self._rows)

        # Create or use provided DSPy program
        if program is not None:
            # Use the provided program directly
            self.program: Module = program
        else:
            # Create DSPy program (mirrors SingleTurnRolloutProcessor)
            # - system_prompt → signature.instructions (GEPA optimizes this!)
            # - user message → input field
            # - assistant response → output field
            self.program = create_single_turn_program(
                system_prompt=self._initial_system_prompt,
                input_field=input_field,
                output_field=output_field,
                module_type=module_type,
                input_desc=input_desc,
                output_desc=output_desc,
                module_factory=module_factory,
            )

        # Convert EP rows to DSPy Examples
        self.train_set: List[Example] = evaluation_rows_to_dspy_examples(train_rows, input_field, output_field)
        self.val_set: List[Example] = evaluation_rows_to_dspy_examples(val_rows, input_field, output_field)
        self.test_set: List[Example] = evaluation_rows_to_dspy_examples(test_rows, input_field, output_field)

    def _load_dataset(self) -> List[EvaluationRow]:
        """
        Load the dataset from ep_params.

        Supports:
        - input_rows: Pre-constructed EvaluationRow objects
            - Can be List[EvaluationRow] (direct usage)
            - Or Sequence[list[EvaluationRow]] (parameterized usage)
        - input_dataset: Paths to JSONL files (requires dataset_adapter)
        - input_messages: Raw message lists
        - data_loaders: EvaluationDataLoader instances
        """
        ep = self.ep_params

        # Case 1: Pre-constructed rows
        # Handle both direct List[EvaluationRow] and parameterized Sequence[list[EvaluationRow]]
        if ep.input_rows:
            rows_input = ep.input_rows
            # Check if it's a list of EvaluationRows (direct) or list of lists (parameterized)
            if rows_input and isinstance(rows_input[0], EvaluationRow):
                # Direct usage: List[EvaluationRow]
                return list(rows_input)
            else:
                # Parameterized usage: Sequence[list[EvaluationRow]]
                all_rows: List[EvaluationRow] = []
                for rows_list in rows_input:
                    if rows_list is not None:
                        all_rows.extend(rows_list)
                return all_rows

        # Case 2: Dataset paths with adapter
        if ep.input_dataset and ep.dataset_adapter:
            from eval_protocol.common_utils import load_jsonl

            all_data: List[Dict[str, Any]] = []
            dataset_paths = ep.input_dataset if isinstance(ep.input_dataset, list) else [ep.input_dataset]

            for path in dataset_paths:
                all_data.extend(load_jsonl(path))

            # Apply max_dataset_rows limit
            if ep.max_dataset_rows:
                all_data = all_data[: ep.max_dataset_rows]

            return ep.dataset_adapter(all_data)

        # Case 3: Input messages (convert to rows)
        # Handle both direct List[List[Message]] and parameterized Sequence[list[list[Message]] | None]
        if ep.input_messages:
            rows: List[EvaluationRow] = []
            messages_input = ep.input_messages

            # Check if first element is a Message (direct list of conversations) or a list (parameterized)
            if messages_input and messages_input[0]:
                first_elem = messages_input[0]
                # Check if it's List[Message] (a single conversation) or List[List[Message]]
                if hasattr(first_elem, "role"):
                    # It's a Message - so input is a single conversation List[Message]
                    rows.append(EvaluationRow(messages=list(messages_input)))
                elif first_elem and hasattr(first_elem[0], "role"):
                    # It's List[List[Message]] - direct usage with multiple conversations
                    for messages in messages_input:
                        if messages:
                            rows.append(EvaluationRow(messages=messages))
                else:
                    # Parameterized usage: Sequence[list[list[Message]] | None]
                    for messages_list in messages_input:
                        if messages_list is not None:
                            for messages in messages_list:
                                rows.append(EvaluationRow(messages=messages))
            return rows

        # Case 4: Data loaders
        if ep.data_loaders:
            from eval_protocol.data_loader.models import EvaluationDataLoader

            rows = []
            data_loaders = ep.data_loaders
            data_loaders_list = (
                [data_loaders] if isinstance(data_loaders, EvaluationDataLoader) else list(data_loaders)
            )
            for data_loader in data_loaders_list:
                results = data_loader.load()
                for result in results:
                    rows.extend(result.rows)

            # Apply max_dataset_rows limit
            if ep.max_dataset_rows:
                rows = rows[: ep.max_dataset_rows]

            return rows

        raise ValueError(
            "No dataset found in ep_params. "
            "Provide input_rows, input_dataset (with dataset_adapter), input_messages, or data_loaders."
        )

    @property
    def initial_system_prompt(self) -> str | None:
        """The original system prompt extracted from the dataset."""
        return self._initial_system_prompt

    def get_optimized_system_prompt(self, optimized_program: Module) -> str:
        """
        Extract the optimized system prompt from a GEPA-optimized program.

        This can be used with EP's rollout processor via system_prompt_override.
        """
        # GEPA stores optimized instructions in the signature
        # Handle both PREDICT (has .signature directly) and ChainOfThought (has .predict.signature)
        if hasattr(optimized_program, "signature"):
            return optimized_program.signature.instructions  # pyright: ignore[reportAttributeAccessIssue]
        elif hasattr(optimized_program, "predict") and hasattr(optimized_program.predict, "signature"):  # pyright: ignore[reportAttributeAccessIssue]
            return optimized_program.predict.signature.instructions  # pyright: ignore[reportAttributeAccessIssue]
        else:
            raise ValueError("Could not find signature.instructions on the optimized program")

    def train(
        self,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        max_full_evals: int | None = None,
        max_metric_calls: int | None = None,
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
        reflection_lm: LM | None = None,
        skip_perfect_score: bool = True,
        add_format_failure_as_feedback: bool = False,
        instruction_proposer: ProposalFn | None = None,
        component_selector: ReflectionComponentSelector | str = "round_robin",
        use_merge: bool = True,
        max_merge_invocations: int | None = 5,
        num_threads: int | None = None,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        log_dir: str | None = None,
        track_stats: bool = False,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        track_best_outputs: bool = False,
        warn_on_score_mismatch: bool = True,
        use_mlflow: bool = False,
        seed: int | None = 0,
        gepa_kwargs: dict | None = None,
    ) -> Module:
        """
        Run GEPA to optimize over candidates.
        """
        gepa_args: dict[str, Any] = {
            "auto": auto,
            "max_full_evals": max_full_evals,
            "max_metric_calls": max_metric_calls,
            "reflection_minibatch_size": reflection_minibatch_size,
            "candidate_selection_strategy": candidate_selection_strategy,
            "reflection_lm": reflection_lm,
            "skip_perfect_score": skip_perfect_score,
            "add_format_failure_as_feedback": add_format_failure_as_feedback,
            "instruction_proposer": instruction_proposer,
            "component_selector": component_selector,
            "use_merge": use_merge,
            "max_merge_invocations": max_merge_invocations,
            "num_threads": num_threads,
            "failure_score": failure_score,
            "perfect_score": perfect_score,
            "log_dir": log_dir,
            "track_stats": track_stats,
            "use_wandb": use_wandb,
            "wandb_api_key": wandb_api_key,
            "wandb_init_kwargs": wandb_init_kwargs,
            "track_best_outputs": track_best_outputs,
            "warn_on_score_mismatch": warn_on_score_mismatch,
            "use_mlflow": use_mlflow,
            "seed": seed,
        }
        gepa_args.update(gepa_kwargs or {})

        optimizer = GEPA(
            metric=self.metric,
            **gepa_args,
        )

        optimized_program = optimizer.compile(
            self.program,
            trainset=self.train_set,
            valset=self.val_set,
        )

        return optimized_program

    def evaluate(
        self,
        optimized_program: Module,
        num_threads: int = 32,
        display_table: bool = True,
        display_progress: bool = True,
    ) -> Any:  # Returns dspy.evaluate.EvaluationResult
        """
        Evaluate the optimized program on the test set using DSPy's Evaluate.

        Args:
            optimized_program: The GEPA-optimized program
            num_threads: Number of parallel threads for evaluation
            display_table: Whether to display results table
            display_progress: Whether to show progress bar

        Returns:
            DSPy EvaluationResult with score and per-example results
        """
        evaluator = dspy.Evaluate(
            devset=self.test_set,
            metric=self.metric,
            num_threads=num_threads,
            display_table=display_table,
            display_progress=display_progress,
        )

        return evaluator(optimized_program)

    def evaluate_baseline(
        self,
        num_threads: int = 32,
        display_table: bool = True,
        display_progress: bool = True,
    ) -> Any:  # Returns dspy.evaluate.EvaluationResult
        """
        Evaluate the unoptimized baseline program on the test set.

        Useful for comparing before/after GEPA optimization.
        """
        return self.evaluate(
            self.program,
            num_threads=num_threads,
            display_table=display_table,
            display_progress=display_progress,
        )

    def _inject_system_prompt(self, rows: List[EvaluationRow], new_system_prompt: str) -> List[EvaluationRow]:
        """
        Create copies of rows with the system prompt replaced.
        """
        modified_rows = []
        for row in rows:
            new_row = row.model_copy(deep=True)
            new_messages = []
            system_found = False
            for msg in new_row.messages:
                if msg.role == "system" and not system_found:
                    # Replace the first system message
                    new_messages.append(Message(role="system", content=new_system_prompt))
                    system_found = True
                else:
                    new_messages.append(msg)
            # If no system message found, prepend one
            if not system_found:
                new_messages.insert(0, Message(role="system", content=new_system_prompt))
            new_row.messages = new_messages
            modified_rows.append(new_row)
        return modified_rows

    async def evaluate_with_ep(
        self,
        optimized_program: Module,
        *,
        use_test_set: bool = True,
        max_concurrent_rollouts: int = 8,
    ) -> Dict[str, Any]:
        """
        Run final evaluation through the normal EP infrastructure.

        This uses the same LLM proxy (EP_LLM_API_BASE) and tracing as a normal
        @evaluation_test job.

        Args:
            optimized_program: The GEPA-optimized program
            use_test_set: If True, evaluate on test set. If False, use full dataset.
            max_concurrent_rollouts: Maximum concurrent LLM calls

        Returns:
            Dict with evaluation results:
            - 'rows': List of evaluated EvaluationRow objects
            - 'score': Aggregate score
            - 'optimized_prompt': The prompt used for evaluation
        """
        # Get optimized system prompt
        optimized_prompt = self.get_optimized_system_prompt(optimized_program)

        # Get rows to evaluate
        if use_test_set:
            # Use stored test rows (same split from __init__)
            rows_to_eval = self._test_rows
        else:
            rows_to_eval = self._rows

        # Inject optimized system prompt into rows
        modified_rows = self._inject_system_prompt(rows_to_eval, optimized_prompt)

        # Set up rollout processor config
        completion_params = self.ep_params.completion_params
        if isinstance(completion_params, list):
            completion_params = completion_params[0] if completion_params else {}
        completion_params = completion_params or {}

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent_rollouts)

        config = RolloutProcessorConfig(
            completion_params=completion_params,
            mcp_config_path="",
            server_script_path=None,
            steps=30,
            logger=default_logger,
            semaphore=semaphore,
            kwargs={},
            exception_handler_config=None,
        )

        # Run rollouts through EP infrastructure (uses EP_LLM_API_BASE)
        rollout_processor = SingleTurnRolloutProcessor()
        rollout_processor.setup()

        try:
            # Execute rollouts
            tasks = rollout_processor(modified_rows, config)
            rolled_out_rows = await asyncio.gather(*tasks)

            # Run evaluation function on each row
            evaluated_rows = []
            scores = []

            for row in rolled_out_rows:
                # Call the original test function for evaluation
                evaluated_row = await execute_pytest(
                    self.test_fn,
                    processed_row=row,  # pyright: ignore[reportArgumentType]
                )
                evaluated_rows.append(evaluated_row)

                # Extract score - evaluated_row is EvaluationRow from execute_pytest
                if hasattr(evaluated_row, "evaluation_result") and evaluated_row.evaluation_result:  # pyright: ignore[reportAttributeAccessIssue]
                    scores.append(evaluated_row.evaluation_result.score)  # pyright: ignore[reportAttributeAccessIssue]

            # Calculate aggregate score
            avg_score = sum(scores) / len(scores) if scores else 0.0

            return {
                "rows": evaluated_rows,
                "score": avg_score,
                "scores": scores,
                "optimized_prompt": optimized_prompt,
            }

        finally:
            # Prefer async cleanup if available, fall back to sync
            if hasattr(rollout_processor, "aclose"):
                await getattr(rollout_processor, "aclose")()
            else:
                rollout_processor.cleanup()

    def run_ep_evaluation(
        self,
        optimized_program: Module,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_with_ep.

        Example:
            trainer = GEPATrainer(test_fn)
            optimized = trainer.train()
            results = trainer.run_ep_evaluation(optimized)
        """
        return asyncio.run(self.evaluate_with_ep(optimized_program, **kwargs))
