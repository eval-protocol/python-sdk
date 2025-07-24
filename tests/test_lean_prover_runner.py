"""
Basic test runner for the DeepSeek-Prover-V2 reward functions.

This is a simple script to test the lean_prover_reward and deepseek_prover_v2_reward
functions outside of the main test framework, to ensure they're working properly.

Run with:
    python tests/test_lean_prover_runner.py
"""

import json
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_protocol.models import Message  # Import Message
from eval_protocol.rewards.lean_prover import (
    deepseek_prover_v2_reward,
    lean_prover_reward,
)


# Helper to create messages list for the runner
def create_runner_messages(statement_content: str, assistant_response: str):
    # The decorator expects list of dicts or list of Message objects.
    # For direct calls in this runner, using dicts is fine.
    return [
        {
            "role": "user",
            "content": f"Prove the following statement: {statement_content}",
        },
        {"role": "assistant", "content": assistant_response},
    ]


def run_tests():
    """Run basic tests for the Lean Prover reward functions"""

    print("Testing lean_prover_reward...")

    # Test with an empty response
    print("\nTest: Empty response")
    empty_statement = "Any statement"
    empty_response_messages = create_runner_messages(empty_statement, "")
    result = lean_prover_reward(messages=empty_response_messages, ground_truth=None, statement=empty_statement)
    print(f"Score: {result['score']}")
    assert result["score"] == 0.0

    # Skip to complete proof and subgoal tests for basic functionality

    # Test with a complete proof
    print("\nTest: Complete proof")
    statement_complete = "If n is a natural number, then n + 1 > n."
    response_complete = """theorem n_lt_n_plus_one (n : ℕ) : n < n + 1 :=
begin
  apply Nat.lt_succ_self,
end
    """
    messages_complete = create_runner_messages(statement_complete, response_complete)
    result = lean_prover_reward(
        messages=messages_complete,
        ground_truth=None,
        statement=statement_complete,
        verbose=True,
    )
    print(f"Score: {result['score']}")
    # Print metrics if verbose mode was enabled
    if "metrics" in result and result["metrics"]:
        print(
            f"Metrics: {json.dumps({k: {'score': v['score'], 'reason': v['reason']} for k, v in result['metrics'].items()}, indent=2)}"
        )
    assert result["score"] >= 0.5

    print("\nTesting deepseek_prover_v2_reward...")

    # Test with a complex proof with subgoals
    print("\nTest: Complex proof with subgoals")
    statement_complex = "For all natural numbers n, the sum of the first n natural numbers is n(n+1)/2."
    response_complex = """theorem sum_naturals (n : ℕ) : ∑ i in range n, i = n * (n + 1) / 2 :=
begin
  -- We'll prove this by induction on n
  induction n with d hd,
  -- Base case: n = 0
  { simp, },
  -- Inductive step: assume true for n = d, prove for n = d + 1
  {
    have step1 : ∑ i in range (d + 1), i = (∑ i in range d, i) + d,
      by simp [sum_range_succ],
    have step2 : (∑ i in range d, i) + d = d * (d + 1) / 2 + d,
      by rw [hd],
    have step3 : d * (d + 1) / 2 + d = (d * (d + 1) + 2 * d) / 2,
      by ring,
    calc
      ∑ i in range (d + 1), i = (∑ i in range d, i) + d : by simp [sum_range_succ]
      ... = d * (d + 1) / 2 + d : by rw [hd]
      ... = (d * (d + 1) + 2 * d) / 2 : by ring
      ... = (d + 1) * ((d + 1) + 1) / 2 : by ring,
  }
end
    """
    messages_complex = create_runner_messages(statement_complex, response_complex)
    result = deepseek_prover_v2_reward(
        messages=messages_complex,
        ground_truth=None,
        statement=statement_complex,
        verbose=True,
    )
    print(f"Score: {result['score']}")
    # Print metrics if verbose mode was enabled
    if "metrics" in result and result["metrics"]:
        print(
            f"Metrics: {json.dumps({k: {'score': v['score'], 'reason': v['reason']} for k, v in result['metrics'].items()}, indent=2)}"
        )
    assert result["score"] > 0.7

    print("\nAll tests passed!")


def test_lean_prover_functions():
    """Run tests for lean prover functions through pytest integration"""
    run_tests()


if __name__ == "__main__":
    run_tests()
