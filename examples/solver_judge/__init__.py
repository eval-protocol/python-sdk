"""
Solver-Judge Evaluation Example for Countdown Task

This module demonstrates a two-phase evaluation workflow:
1. Solver Phase: Generate multiple candidate solutions using an LLM
2. Judge Phase: Use an LLM judge to select the best solution

The example uses the Countdown task where you must use given numbers
to create an arithmetic expression that equals a target number.

See README.md for usage instructions.
"""

from .main import (
    # Core functions
    run_solver_judge_workflow,
    run_judge,
    solver_judge_reward,
    countdown_reward,
    # Countdown utilities
    extract_solution,
    validate_equation,
    evaluate_equation,
    compute_score,
    get_countdown_prompt,
    create_judge_prompt,
    parse_judge_selection,
    # Configuration
    JUDGE_CONFIG,
    N_SOLUTIONS,
    # Demo data
    DEMO_ROWS,
)

__all__ = [
    "run_solver_judge_workflow",
    "run_judge",
    "solver_judge_reward",
    "countdown_reward",
    "extract_solution",
    "validate_equation",
    "evaluate_equation",
    "compute_score",
    "get_countdown_prompt",
    "create_judge_prompt",
    "parse_judge_selection",
    "JUDGE_CONFIG",
    "N_SOLUTIONS",
    "DEMO_ROWS",
]
