"""
Solver-Judge Evaluation Example for Countdown Task

This example demonstrates a two-phase evaluation workflow:
1. **Solver Phase**: Generate multiple candidate solutions for a problem using an LLM
2. **Judge Phase**: Use an LLM judge to select the best solution from the candidates

This pattern is useful for:
- Best-of-N sampling with intelligent selection
- Self-consistency evaluation
- Verifier-guided generation
- MCTS-style exploration with value models

The example uses the Countdown task format where:
- Given a target number and a set of available numbers
- Create an arithmetic equation using those numbers to reach the target
- Each number can only be used once
- Only basic arithmetic operations (+, -, *, /) are allowed
- The answer should be in <answer>...</answer> tags
"""

import asyncio
import os
import random
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
# Countdown Task Reward Function
# ============================================================================


def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:" if present
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    # Look for answer pattern in the entire string
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str: str, available_numbers: List[int]) -> bool:
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except Exception:
        return False


def evaluate_equation(equation_str: str) -> Optional[float]:
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def compute_score(
    solution_str: str,
    ground_truth: Dict[str, Any],
    format_score: float = 0.1,
    score: float = 1.0,
    do_print: bool = False,
) -> float:
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
        do_print: whether to print debug info
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)

    if do_print or random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str[:200]}...")

    if equation is None:
        if do_print:
            print("No equation found")
        return 0.0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except Exception:
        if do_print:
            print("Error evaluating equation")
        return format_score


# ============================================================================
# Solver Utilities
# ============================================================================


def get_countdown_prompt(target: int, numbers: List[int]) -> str:
    """Create the solver prompt for a countdown problem."""
    return f"""Countdown Problem:
Target: {target}
Available numbers: {numbers}

Use each of the available numbers exactly once to create an arithmetic expression that equals the target.
You may only use basic arithmetic operations: +, -, *, /
You may use parentheses to control the order of operations.

Think step by step, then output your final equation within <answer>...</answer> tags.
For example: <answer>(25 + 5) * 4</answer>"""


# ============================================================================
# Judge Utilities
# ============================================================================


def create_judge_prompt(target: int, numbers: List[int], solutions: List[str]) -> str:
    """Create a prompt for the judge to evaluate and select the best solution."""
    prompt = f"""You are an expert verifier. Given a countdown problem and multiple solution attempts, select a correct solution.

Problem:
Target: {target}
Available numbers: {numbers}

Solutions to evaluate:
"""
    for i, solution in enumerate(solutions, 1):
        prompt += f"\nSolution {i}:\n{solution}\n"

    prompt += """
A correct solution must satisfy the following criteria:
1. The solution uses only the given numbers.
2. Each number is used exactly once.
3. Only basic arithmetic operations (+, -, *, /) are used.
4. The calculation results in the target number.
5. The final answer is clearly marked within <answer>...</answer> tags.

Output the index of your selected solution within <answer>...</answer> tags.
Example: <answer>1</answer> for the first solution, <answer>2</answer> for the second solution, etc.
If multiple solutions are correct, output the index of the first correct solution.
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
    target: int,
    numbers: List[int],
    solutions: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the LLM judge to select the best solution.

    Args:
        target: The target number for the countdown problem
        numbers: The available numbers
        solutions: List of candidate solutions
        config: Optional judge configuration (defaults to JUDGE_CONFIG)

    Returns:
        Dictionary with:
        - selected_index: Index of selected solution (0-indexed) or -1 if none
        - selected_solution: The selected solution text or empty string
        - judge_response: Full judge response for debugging
    """
    config = config or JUDGE_CONFIG
    judge_prompt = create_judge_prompt(target, numbers, solutions)
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


def countdown_reward(
    solution_str: str,
    ground_truth: Dict[str, Any],
    format_score: float = 0.1,
) -> EvaluateResult:
    """
    Evaluate a countdown solution against the ground truth.

    Args:
        solution_str: The solution text from the model
        ground_truth: Dictionary with 'target' (int) and 'numbers' (List[int])
        format_score: Score for partial credit (correct format but wrong answer)

    Returns:
        EvaluateResult with score and detailed metrics
    """
    if not ground_truth or "target" not in ground_truth or "numbers" not in ground_truth:
        return EvaluateResult(
            score=0.0,
            reason="Missing ground truth (target or numbers)",
            is_score_valid=False,
        )

    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    # Extract the equation
    equation = extract_solution(solution_str)
    has_answer_tag = equation is not None

    if equation is None:
        return EvaluateResult(
            score=0.0,
            reason="No equation found in <answer>...</answer> tags",
            metrics={
                "format": MetricResult(score=0.0, reason="Missing answer tags", is_score_valid=True),
                "validation": MetricResult(score=0.0, reason="No equation to validate", is_score_valid=True),
                "correctness": MetricResult(score=0.0, reason="No equation to evaluate", is_score_valid=True),
            },
        )

    # Validate equation uses correct numbers
    numbers_valid = validate_equation(equation, numbers)
    if not numbers_valid:
        return EvaluateResult(
            score=format_score,
            reason=f"Equation '{equation}' does not use the available numbers correctly",
            metrics={
                "format": MetricResult(score=1.0, reason="Has answer tags", is_score_valid=True),
                "validation": MetricResult(
                    score=0.0,
                    reason=f"Numbers validation failed for: {equation}",
                    is_score_valid=True,
                ),
                "correctness": MetricResult(score=0.0, reason="Invalid equation", is_score_valid=True),
            },
        )

    # Evaluate the equation
    result = evaluate_equation(equation)
    if result is None:
        return EvaluateResult(
            score=format_score,
            reason=f"Could not evaluate equation: {equation}",
            metrics={
                "format": MetricResult(score=1.0, reason="Has answer tags", is_score_valid=True),
                "validation": MetricResult(score=1.0, reason="Numbers used correctly", is_score_valid=True),
                "correctness": MetricResult(
                    score=0.0,
                    reason=f"Evaluation error for: {equation}",
                    is_score_valid=True,
                ),
            },
        )

    # Check if result matches target
    is_correct = abs(result - target) < 1e-5
    if is_correct:
        return EvaluateResult(
            score=1.0,
            reason=f"Correct! {equation} = {result} = {target}",
            metrics={
                "format": MetricResult(score=1.0, reason="Has answer tags", is_score_valid=True),
                "validation": MetricResult(score=1.0, reason="Numbers used correctly", is_score_valid=True),
                "correctness": MetricResult(
                    score=1.0,
                    reason=f"Correct: {equation} = {target}",
                    is_score_valid=True,
                    data={"equation": equation, "result": result, "target": target},
                ),
            },
        )
    else:
        return EvaluateResult(
            score=format_score,
            reason=f"Wrong result: {equation} = {result}, target = {target}",
            metrics={
                "format": MetricResult(score=1.0, reason="Has answer tags", is_score_valid=True),
                "validation": MetricResult(score=1.0, reason="Numbers used correctly", is_score_valid=True),
                "correctness": MetricResult(
                    score=0.0,
                    reason=f"Wrong: {equation} = {result} (target: {target})",
                    is_score_valid=True,
                    data={"equation": equation, "result": result, "target": target},
                ),
            },
        )


@reward_function
def solver_judge_reward(
    messages: List[Message],
    ground_truth: Any = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate a countdown solution from messages.

    This is the per-solution reward function used to score individual solver outputs.
    The ground_truth should be a dict with 'target' (int) and 'numbers' (List[int]).
    """
    if not messages:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            is_score_valid=False,
        )

    # Get the last assistant message (the solution)
    last_msg = messages[-1]
    raw_content = last_msg.content if isinstance(last_msg, Message) else last_msg.get("content", "")

    # Handle multimodal content (list of parts) - extract text
    if isinstance(raw_content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw_content
        )
    else:
        content = str(raw_content) if raw_content else ""

    if not content:
        return EvaluateResult(
            score=0.0,
            reason="Empty response",
            is_score_valid=True,
        )

    # Parse ground_truth - should be a dict with 'target' and 'numbers'
    if isinstance(ground_truth, dict) and "target" in ground_truth and "numbers" in ground_truth:
        return countdown_reward(content, ground_truth)
    else:
        # Fallback: just check if answer tags are present
        has_answer = extract_solution(content) is not None
        return EvaluateResult(
            score=1.0 if has_answer else 0.0,
            reason="Format check only (invalid ground_truth format)",
            metrics={
                "format": MetricResult(
                    score=1.0 if has_answer else 0.0,
                    reason="Has answer tags" if has_answer else "Missing answer tags",
                    is_score_valid=True,
                ),
            },
        )


# ============================================================================
# Demo Dataset
# ============================================================================

# Sample countdown problems for demonstration
# ground_truth is a dict with 'target' and 'numbers'
DEMO_ROWS = [
    EvaluationRow(
        messages=[
            Message(
                role="user",
                content=get_countdown_prompt(target=24, numbers=[1, 2, 3, 4]),
            ),
        ],
        ground_truth={"target": 24, "numbers": [1, 2, 3, 4]},
    ),
    EvaluationRow(
        messages=[
            Message(
                role="user",
                content=get_countdown_prompt(target=100, numbers=[25, 50, 2, 1]),
            ),
        ],
        ground_truth={"target": 100, "numbers": [25, 50, 2, 1]},
    ),
    EvaluationRow(
        messages=[
            Message(
                role="user",
                content=get_countdown_prompt(target=42, numbers=[6, 7, 1, 2]),
            ),
        ],
        ground_truth={"target": 42, "numbers": [6, 7, 1, 2]},
    ),
]


# ============================================================================
# Evaluation Tests
# ============================================================================


@evaluation_test(
    input_rows=[DEMO_ROWS],
    completion_params=[
        {"model": "accounts/fireworks/models/qwen3-8b", "temperature": 0.7}
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    mode="pointwise",
)
async def test_solver_single(row: EvaluationRow, **kwargs) -> EvaluationRow:
    """
    Basic solver evaluation - generate a single solution and score it.

    This is the simplest form of the countdown solver evaluation.
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
        {"model": "accounts/fireworks/models/qwen3-8b", "temperature": 0.3 + i * 0.2}
        for i in range(N_SOLUTIONS)
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    mode="groupwise",  # Process all N solutions together
)
async def test_solver_judge(rows: List[EvaluationRow], **kwargs) -> List[EvaluationRow]:
    """
    Full Solver-Judge evaluation workflow for Countdown task.

    1. Receive N candidate solutions (from groupwise mode)
    2. Score each solution individually using countdown_reward
    3. Run the LLM judge to select the best solution
    4. Report both solver accuracy and judge accuracy metrics
    """
    if not rows:
        return rows

    # Helper to extract text from message content (handles multimodal)
    def extract_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    # Get ground_truth from the first row (all rows share the same input)
    ground_truth = rows[0].ground_truth

    # Validate ground_truth format for countdown
    if not isinstance(ground_truth, dict) or "target" not in ground_truth or "numbers" not in ground_truth:
        # Invalid ground_truth - mark all rows with error
        error_result = EvaluateResult(
            score=0.0,
            reason="Invalid ground_truth format. Expected dict with 'target' and 'numbers'.",
            is_score_valid=False,
        )
        for row in rows:
            row.evaluation_result = error_result
        return rows

    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    # Extract solutions from each row and score them individually
    solutions: List[str] = []
    solver_scores: List[float] = []

    for row in rows:
        last_assistant = row.last_assistant_message()
        solution = extract_text(last_assistant.content) if last_assistant else ""
        solutions.append(solution)

        # Score each solution individually (for metrics only)
        result = solver_judge_reward(
            messages=row.messages,
            ground_truth=ground_truth,
        )
        solver_scores.append(result.score)

    # Calculate solver accuracy (fraction of fully correct solutions, score == 1.0)
    num_correct = sum(1 for s in solver_scores if s >= 1.0)
    solver_acc = num_correct / len(solver_scores) if solver_scores else 0.0

    # Run the LLM judge to select the best solution
    judge_result = await run_judge(target, numbers, solutions)
    selected_index = judge_result["selected_index"]
    selected_solution = judge_result["selected_solution"]

    # Evaluate the judge's selection using the same countdown_reward logic
    if selected_index >= 0:
        judge_eval = countdown_reward(selected_solution, ground_truth)
        judge_correct = judge_eval.score >= 1.0
        judge_acc = 1.0 if judge_correct else 0.0
        reason = f"Judge selected solution {selected_index + 1} (1-indexed). {'Correct' if judge_correct else 'Incorrect'}"
    else:
        judge_correct = False
        judge_acc = 0.0
        reason = "Judge failed to select a valid solution"

    # Create the final evaluation result based on judge's decision
    # This is the score that matters for the overall evaluation
    final_result = EvaluateResult(
        score=judge_acc,
        reason=reason,
        metrics={
            "solver_accuracy": MetricResult(
                score=solver_acc,
                reason=f"Solver accuracy: {solver_acc:.2%} ({num_correct}/{len(solver_scores)} fully correct)",
                is_score_valid=True,
            ),
            "judge_accuracy": MetricResult(
                score=judge_acc,
                reason=f"Judge selection {'correct' if judge_correct else 'incorrect'}" if selected_index >= 0 else "Judge failed to select",
                is_score_valid=True,
            ),
            "individual_scores": MetricResult(
                score=solver_acc,
                reason=f"Individual solution scores: {solver_scores}",
                is_score_valid=True,
                data={
                    "scores": solver_scores,
                    "selected_index": selected_index,
                },
            ),
        },
    )

    # IMPORTANT: Update ALL rows with the same judge-based result
    # In groupwise mode, all rows' scores are aggregated, so they must all
    # reflect the judge's decision, not individual solver scores
    for row in rows:
        row.evaluation_result = final_result

    return rows


# ============================================================================
# Standalone Async Workflow (for non-pytest usage)
# ============================================================================


async def run_solver_judge_workflow(
    target: int,
    numbers: List[int],
    n_solutions: int = N_SOLUTIONS,
    solver_config: Optional[Dict[str, Any]] = None,
    judge_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the complete Solver-Judge workflow programmatically for Countdown task.

    This function can be used outside of pytest for integration into other systems.

    Args:
        target: The target number to reach
        numbers: The available numbers to use
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
        "model": "accounts/fireworks/models/qwen3-8b",
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": os.getenv("FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1"),
    }
    judge_config = judge_config or JUDGE_CONFIG

    ground_truth = {"target": target, "numbers": numbers}
    prompt = get_countdown_prompt(target, numbers)
    messages = [{"role": "user", "content": prompt}]

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

    # Phase 2: Score each solution using countdown_reward
    solver_scores = []
    for solution in solutions:
        result = countdown_reward(solution, ground_truth)
        solver_scores.append(result.score)

    # Solver accuracy = fraction of fully correct solutions (score == 1.0)
    num_correct = sum(1 for s in solver_scores if s >= 1.0)
    solver_accuracy = num_correct / len(solver_scores) if solver_scores else 0.0

    # Phase 3: Judge selects the best solution
    judge_result = await run_judge(target, numbers, solutions, judge_config)
    selected_index = judge_result["selected_index"]
    selected_solution = judge_result["selected_solution"]

    # Phase 4: Evaluate the judge's selection
    if selected_index >= 0:
        judge_eval = countdown_reward(selected_solution, ground_truth)
        judge_correct = judge_eval.score >= 1.0
        judge_accuracy = 1.0 if judge_correct else 0.0
    else:
        judge_accuracy = 0.0

    return {
        "target": target,
        "numbers": numbers,
        "solutions": list(solutions),
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
    """Run a demo of the Solver-Judge workflow for Countdown task."""
    print("=" * 60)
    print("Solver-Judge Workflow Demo (Countdown Task)")
    print("=" * 60)

    # Demo countdown problem: use [1, 2, 3, 4] to make 24
    target = 24
    numbers = [1, 2, 3, 4]

    print(f"\nTarget: {target}")
    print(f"Available Numbers: {numbers}")
    print("\nRunning workflow...")

    result = await run_solver_judge_workflow(
        target=target,
        numbers=numbers,
        n_solutions=3,
    )

    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)

    print(f"\nSolver Accuracy: {result['solver_accuracy']:.2%}")
    print(f"Individual Scores: {result['solver_scores']}")

    if result["selected_index"] >= 0:
        print(f"\nJudge Selected: Solution {result['selected_index'] + 1}")
    else:
        print("\nJudge: Failed to select a valid solution")
    print(f"Judge Accuracy: {result['judge_accuracy']:.2%}")

    print("\n" + "-" * 40)
    print("Solutions Generated:")
    print("-" * 40)
    for i, sol in enumerate(result["solutions"], 1):
        preview = sol[:300] + "..." if len(sol) > 300 else sol
        score_val = result["solver_scores"][i - 1]
        if score_val >= 1.0:
            score = "✓"
        elif score_val > 0:
            score = "~"  # partial credit
        else:
            score = "✗"
        selected = " [SELECTED]" if i - 1 == result["selected_index"] else ""
        print(f"\n{score} Solution {i} (score: {score_val:.2f}){selected}:")
        print(preview)


if __name__ == "__main__":
    asyncio.run(main())
