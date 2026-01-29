"""
Solver-Judge Evaluation Example

This example demonstrates a two-phase evaluation workflow:
1. **Solver Phase**: Generate multiple candidate solutions for a problem using an LLM
2. **Judge Phase**: Use an LLM judge to select the best solution from the candidates

This pattern is useful for:
- Best-of-N sampling with intelligent selection
- Self-consistency evaluation
- Verifier-guided generation
- MCTS-style exploration with value models

The example uses a math/countdown problem format where:
- The solver generates solutions with <answer>...</answer> tags
- The judge evaluates and selects the most correct solution
"""

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from eval_protocol import EvaluateResult, MetricResult, reward_function
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test, SingleTurnRolloutProcessor

# ============================================================================
# Configuration
# ============================================================================

# Judge LLM configuration - customize based on your setup
JUDGE_CONFIG = {
    "model": os.getenv("JUDGE_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct"),
    "temperature": 0.0,
    "max_tokens": 4096,
    "api_key": os.getenv("FIREWORKS_API_KEY"),
    "base_url": os.getenv("FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1"),
}

# Number of candidate solutions to generate
N_SOLUTIONS = int(os.getenv("N_SOLUTIONS", "3"))


# ============================================================================
# Solver Utilities
# ============================================================================


def parse_solver_answer(response: str) -> str:
    """Extract the answer from <answer>...</answer> tags in the solver's response."""
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return f"<answer>{answer_match.group(1).strip()}</answer>"
    return "No solution found"


def get_solver_prompt(problem: str) -> List[Dict[str, str]]:
    """Create the solver prompt for a given problem."""
    return [
        {
            "role": "user",
            "content": f"{problem}\n\nOutput the final answer within <answer>...</answer> tags.",
        }
    ]


# ============================================================================
# Judge Utilities
# ============================================================================


def create_judge_prompt(problem: str, solutions: List[str]) -> str:
    """Create a prompt for the judge to evaluate and select the best solution."""
    prompt = f"""You are an expert verifier. Given a problem and multiple solution attempts, select the most correct solution.

Problem:
{problem}

Solutions to evaluate:
"""
    for i, solution in enumerate(solutions, 1):
        prompt += f"\nSolution {i}:\n{solution}\n"

    prompt += """
Evaluation criteria:
1. The solution correctly addresses the problem
2. The reasoning is sound and logical
3. The final answer is clearly marked within <answer>...</answer> tags

Output the index of your selected solution within <answer>...</answer> tags.
Example: <answer>1</answer> for the first solution, <answer>2</answer> for the second solution, etc.
If multiple solutions are correct, select the one with the clearest reasoning.
If none are correct, output <answer>0</answer>."""
    return prompt


def parse_judge_selection(response: str, num_solutions: int) -> int:
    """Parse the judge's selection from the response. Returns 0-indexed solution index or -1 if invalid."""
    answer_match = re.search(r"<answer>(\d+)</answer>", response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        try:
            selection = int(answer_match.group(1).strip())
            # Convert 1-indexed to 0-indexed, handle "0" as no valid selection
            if 1 <= selection <= num_solutions:
                return selection - 1
            return -1
        except ValueError:
            return -1
    return -1


async def run_judge(
    problem: str,
    solutions: List[str],
    config: Dict[str, Any] = JUDGE_CONFIG,
) -> Dict[str, Any]:
    """
    Run the LLM judge to select the best solution.

    Returns:
        Dictionary with:
        - selected_index: Index of selected solution (0-indexed) or -1 if none
        - selected_solution: The selected solution text or empty string
        - judge_response: Full judge response for debugging
    """
    judge_prompt = create_judge_prompt(problem, solutions)
    messages = [{"role": "user", "content": judge_prompt}]

    try:
        async with AsyncOpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
        ) as client:
            response = await client.chat.completions.create(
                model=config["model"],
                messages=messages,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )

        judge_response = response.choices[0].message.content or ""
        selected_index = parse_judge_selection(judge_response, len(solutions))

        return {
            "selected_index": selected_index,
            "selected_solution": solutions[selected_index] if selected_index >= 0 else "",
            "judge_response": judge_response,
        }
    except Exception as e:
        return {
            "selected_index": -1,
            "selected_solution": "",
            "judge_response": f"Error: {str(e)}",
        }


# ============================================================================
# Reward Function
# ============================================================================


def check_answer_correct(solution: str, ground_truth: str) -> bool:
    """
    Check if a solution's answer matches the ground truth.
    This is a simple string matching - customize for your problem domain.
    """
    # Extract answer from solution
    answer_match = re.search(r"<answer>(.*?)</answer>", solution, re.IGNORECASE | re.DOTALL)
    if not answer_match:
        return False

    extracted_answer = answer_match.group(1).strip().lower()
    expected_answer = ground_truth.strip().lower()

    # For numeric answers, try numeric comparison
    try:
        return abs(float(extracted_answer) - float(expected_answer)) < 1e-6
    except ValueError:
        pass

    # Fallback to string comparison
    return extracted_answer == expected_answer


@reward_function
def solver_judge_reward(
    messages: List[Message],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate a single solution against ground truth.

    This is the per-solution reward function used to score individual solver outputs.
    """
    if not messages:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            is_score_valid=False,
        )

    # Get the last assistant message (the solution)
    last_msg = messages[-1]
    content = last_msg.content if isinstance(last_msg, Message) else last_msg.get("content", "")

    if not content:
        return EvaluateResult(
            score=0.0,
            reason="Empty response",
            is_score_valid=True,
        )

    # Parse the answer
    parsed_answer = parse_solver_answer(content)

    # Check if answer format is correct
    has_answer_tag = "<answer>" in parsed_answer.lower() and "</answer>" in parsed_answer.lower()

    if not ground_truth:
        # If no ground truth, just score based on format
        format_score = 1.0 if has_answer_tag else 0.0
        return EvaluateResult(
            score=format_score,
            reason="Format check only (no ground truth)",
            metrics={
                "format": MetricResult(
                    score=format_score,
                    reason="Has answer tags" if has_answer_tag else "Missing answer tags",
                    is_score_valid=True,
                ),
            },
        )

    # Check correctness
    is_correct = check_answer_correct(content, ground_truth)
    correctness_score = 1.0 if is_correct else 0.0

    return EvaluateResult(
        score=correctness_score,
        reason="Correct" if is_correct else "Incorrect",
        metrics={
            "correctness": MetricResult(
                score=correctness_score,
                reason=f"Answer {'matches' if is_correct else 'does not match'} ground truth",
                is_score_valid=True,
            ),
            "format": MetricResult(
                score=1.0 if has_answer_tag else 0.0,
                reason="Has answer tags" if has_answer_tag else "Missing answer tags",
                is_score_valid=True,
            ),
        },
    )


# ============================================================================
# Demo Dataset
# ============================================================================

# Sample math problems for demonstration
DEMO_ROWS = [
    EvaluationRow(
        messages=[
            Message(
                role="user",
                content="What is 15 + 27? Show your work and provide the answer in <answer>...</answer> tags.",
            ),
        ],
        ground_truth="42",
    ),
    EvaluationRow(
        messages=[
            Message(
                role="user",
                content="Calculate 8 * 7. Show your work and provide the answer in <answer>...</answer> tags.",
            ),
        ],
        ground_truth="56",
    ),
    EvaluationRow(
        messages=[
            Message(
                role="user",
                content="What is 100 - 37? Show your work and provide the answer in <answer>...</answer> tags.",
            ),
        ],
        ground_truth="63",
    ),
]


# ============================================================================
# Evaluation Tests
# ============================================================================


@evaluation_test(
    input_rows=[DEMO_ROWS],
    completion_params=[
        {"model": "accounts/fireworks/models/llama-v3p1-8b-instruct", "temperature": 0.7}
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    mode="pointwise",
)
async def test_solver_single(row: EvaluationRow, **kwargs) -> EvaluationRow:
    """
    Basic solver evaluation - generate a single solution and score it.

    This is the simplest form of the solver evaluation.
    """
    result = solver_judge_reward(
        messages=row.messages,
        ground_truth=row.ground_truth,
    )
    row.evaluation_result = result
    return row


@evaluation_test(
    input_rows=[DEMO_ROWS],
    # Generate N solutions with different temperatures for diversity
    completion_params=[
        {"model": "accounts/fireworks/models/llama-v3p1-8b-instruct", "temperature": 0.3 + i * 0.2}
        for i in range(N_SOLUTIONS)
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    mode="groupwise",  # Process all N solutions together
)
async def test_solver_judge(rows: List[EvaluationRow], **kwargs) -> List[EvaluationRow]:
    """
    Full Solver-Judge evaluation workflow.

    1. Receive N candidate solutions (from groupwise mode)
    2. Score each solution individually
    3. Run the LLM judge to select the best solution
    4. Report both solver accuracy and judge accuracy metrics
    """
    if not rows:
        return rows

    # Get the problem from the first row (all rows share the same input)
    first_user_msg = rows[0].get_first_user_message()
    problem = first_user_msg.content if first_user_msg else ""
    ground_truth = rows[0].ground_truth

    # Extract solutions from each row
    solutions = []
    solver_scores = []

    for row in rows:
        last_assistant = row.last_assistant_message()
        solution = last_assistant.content if last_assistant else ""
        solutions.append(solution)

        # Score each solution
        result = solver_judge_reward(
            messages=row.messages,
            ground_truth=ground_truth,
        )
        solver_scores.append(result.score)
        row.evaluation_result = result

    # Calculate solver accuracy (fraction of correct solutions)
    solver_acc = sum(solver_scores) / len(solver_scores) if solver_scores else 0.0

    # Run the LLM judge to select the best solution
    judge_result = await run_judge(problem, solutions)
    selected_index = judge_result["selected_index"]
    selected_solution = judge_result["selected_solution"]

    # Evaluate the judge's selection
    if selected_index >= 0:
        judge_correct = check_answer_correct(selected_solution, str(ground_truth)) if ground_truth else False
        judge_acc = 1.0 if judge_correct else 0.0
    else:
        judge_correct = False
        judge_acc = 0.0

    # Update the selected row with comprehensive metrics
    if selected_index >= 0 and selected_index < len(rows):
        selected_row = rows[selected_index]
        selected_row.evaluation_result = EvaluateResult(
            score=judge_acc,
            reason=f"Judge selected solution {selected_index + 1} (1-indexed). {'Correct' if judge_correct else 'Incorrect'}",
            metrics={
                "solver_accuracy": MetricResult(
                    score=solver_acc,
                    reason=f"Solver accuracy: {solver_acc:.2%} ({sum(1 for s in solver_scores if s > 0)}/{len(solver_scores)} correct)",
                    is_score_valid=True,
                ),
                "judge_accuracy": MetricResult(
                    score=judge_acc,
                    reason=f"Judge selection {'correct' if judge_correct else 'incorrect'}",
                    is_score_valid=True,
                ),
                "individual_scores": MetricResult(
                    score=solver_acc,
                    reason=f"Individual solution scores: {solver_scores}",
                    is_score_valid=True,
                    data={"scores": solver_scores},
                ),
            },
        )
    else:
        # Judge failed to select - mark first row with error
        rows[0].evaluation_result = EvaluateResult(
            score=0.0,
            reason="Judge failed to select a valid solution",
            metrics={
                "solver_accuracy": MetricResult(
                    score=solver_acc,
                    reason=f"Solver accuracy: {solver_acc:.2%}",
                    is_score_valid=True,
                ),
                "judge_accuracy": MetricResult(
                    score=0.0,
                    reason="Judge failed to select",
                    is_score_valid=True,
                ),
            },
        )

    return rows


# ============================================================================
# Standalone Async Workflow (for non-pytest usage)
# ============================================================================


async def run_solver_judge_workflow(
    problem: str,
    ground_truth: Optional[str] = None,
    n_solutions: int = N_SOLUTIONS,
    solver_config: Optional[Dict[str, Any]] = None,
    judge_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the complete Solver-Judge workflow programmatically.

    This function can be used outside of pytest for integration into other systems.

    Args:
        problem: The problem to solve
        ground_truth: Optional ground truth answer for scoring
        n_solutions: Number of candidate solutions to generate
        solver_config: Configuration for the solver LLM
        judge_config: Configuration for the judge LLM

    Returns:
        Dictionary containing:
        - solutions: List of generated solutions
        - solver_scores: Individual solution scores
        - solver_accuracy: Overall solver accuracy
        - selected_index: Index of judge-selected solution
        - selected_solution: The selected solution
        - judge_accuracy: Whether judge selection was correct
    """
    solver_config = solver_config or {
        "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": os.getenv("FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1"),
    }
    judge_config = judge_config or JUDGE_CONFIG

    # Phase 1: Generate multiple solutions
    messages = get_solver_prompt(problem)

    async def generate_solution(temperature: float) -> str:
        """Generate a single solution with the given temperature."""
        async with AsyncOpenAI(
            api_key=solver_config.get("api_key"),
            base_url=solver_config.get("base_url"),
        ) as client:
            response = await client.chat.completions.create(
                model=solver_config["model"],
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""

    # Generate solutions in parallel with varying temperatures
    temperatures = [0.3 + i * 0.2 for i in range(n_solutions)]
    solutions = await asyncio.gather(*[generate_solution(t) for t in temperatures])

    # Phase 2: Score each solution
    solver_scores = []
    for solution in solutions:
        is_correct = check_answer_correct(solution, ground_truth) if ground_truth else False
        solver_scores.append(1.0 if is_correct else 0.0)

    solver_accuracy = sum(solver_scores) / len(solver_scores) if solver_scores else 0.0

    # Phase 3: Judge selects the best solution
    judge_result = await run_judge(problem, solutions, judge_config)
    selected_index = judge_result["selected_index"]
    selected_solution = judge_result["selected_solution"]

    # Phase 4: Evaluate the judge's selection
    if selected_index >= 0 and ground_truth:
        judge_correct = check_answer_correct(selected_solution, ground_truth)
        judge_accuracy = 1.0 if judge_correct else 0.0
    else:
        judge_accuracy = 0.0

    return {
        "problem": problem,
        "solutions": solutions,
        "solver_scores": solver_scores,
        "solver_accuracy": solver_accuracy,
        "selected_index": selected_index,
        "selected_solution": selected_solution,
        "judge_accuracy": judge_accuracy,
        "judge_response": judge_result["judge_response"],
    }


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Run a demo of the Solver-Judge workflow."""
    print("=" * 60)
    print("Solver-Judge Workflow Demo")
    print("=" * 60)

    # Demo problem
    problem = "What is 15 + 27? Show your work and provide the answer in <answer>...</answer> tags."
    ground_truth = "42"

    print(f"\nProblem: {problem}")
    print(f"Ground Truth: {ground_truth}")
    print("\nRunning workflow...")

    result = await run_solver_judge_workflow(
        problem=problem,
        ground_truth=ground_truth,
        n_solutions=3,
    )

    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)

    print(f"\nSolver Accuracy: {result['solver_accuracy']:.2%}")
    print(f"Individual Scores: {result['solver_scores']}")

    print(f"\nJudge Selected: Solution {result['selected_index'] + 1}")
    print(f"Judge Accuracy: {result['judge_accuracy']:.2%}")

    print("\n" + "-" * 40)
    print("Solutions Generated:")
    print("-" * 40)
    for i, sol in enumerate(result["solutions"], 1):
        preview = sol[:200] + "..." if len(sol) > 200 else sol
        score = "✓" if result["solver_scores"][i - 1] > 0 else "✗"
        selected = " [SELECTED]" if i - 1 == result["selected_index"] else ""
        print(f"\n{score} Solution {i}{selected}:")
        print(preview)


if __name__ == "__main__":
    asyncio.run(main())
