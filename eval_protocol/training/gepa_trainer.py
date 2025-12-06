from typing import Any, Dict, Literal

import dspy
from dspy.clients.lm import LM
from dspy.primitives import Module
from dspy.teleprompt.gepa.gepa import GEPA
from gepa.core.adapter import ProposalFn
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector

from eval_protocol.models import EPParameters, EvaluationRow
from eval_protocol.pytest.types import TestFunction
from eval_protocol.training.trainer import Trainer
from eval_protocol.training.utils import build_ep_parameters_from_test


class GEPATrainer(Trainer):
    """
    High-level entrypoint for running GEPA-style training against an existing
    `@evaluation_test`-decorated function.

    This class is intentionally minimal for now:
    - It captures `EPParameters` from the provided test function via
      `build_ep_parameters_from_test`.
    - It stores any GEPA-related configuration kwargs for future use.
    - The actual GEPA optimization loop is left as a TODO.
    """

    def __init__(self, test_fn: TestFunction) -> None:
        """
        Args:
            test_fn: The `@evaluation_test`-decorated function defining the eval.
        """
        super().__init__(test_fn)
        self.ep_params: EPParameters = build_ep_parameters_from_test(test_fn)

        self.metric = (
            test_fn  # TODO: need to convert our ep test_fn to a GEPA metric. also need to inject the feedback text.
        )

        self.program = ...  # TODO: converting between a program (dspy.Module) and an @evaluation_test is a bit tricky.

        self.train_set, self.val_set, self.test_set = (
            ...,
            ...,
            ...,
        )  # TODO: need to convert our input_dataset to a train set

    def train(
        self,
        auto: Literal["light", "medium", "heavy"] | None = None,
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
        enable_tool_optimization: bool = False,
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
            "enable_tool_optimization": enable_tool_optimization,
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

    def evaluate(self, optimized_program: Module) -> list[EvaluationRow]:
        # convert back to EP

        # and then just run our evaluation_test function on the optimized program.

        # OR we can evaluate using dspy.Evaluate

        # evaluate = dspy.Evaluate(
        #     devset=self.test_set,
        #     metric=self.metric,
        #     num_threads=32,
        #     display_table=True,
        #     display_progress=True
        # )

        # return evaluate(self.optimized_program)
        ...
