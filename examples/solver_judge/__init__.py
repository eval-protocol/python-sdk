"""
Solver-Judge Evaluation Example

This module demonstrates a two-phase evaluation workflow:
1. Solver Phase: Generate multiple candidate solutions using an LLM
2. Judge Phase: Use an LLM judge to select the best solution

See README.md for usage instructions.
"""

from .main import (
    # Core functions
    run_solver_judge_workflow,
    run_judge,
    solver_judge_reward,
    # Utilities
    parse_solver_answer,
    check_answer_correct,
    create_judge_prompt,
    parse_judge_selection,
    get_solver_prompt,
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
    "parse_solver_answer",
    "check_answer_correct",
    "create_judge_prompt",
    "parse_judge_selection",
    "get_solver_prompt",
    "JUDGE_CONFIG",
    "N_SOLUTIONS",
    "DEMO_ROWS",
]
