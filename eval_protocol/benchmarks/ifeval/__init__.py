"""IFEval benchmark for evaluating instruction-following capabilities.

Usage:
    from eval_protocol.benchmarks.ifeval import ifeval_partial_credit_reward

    score = ifeval_partial_credit_reward(response, ground_truth)
"""

from .reward import ifeval_partial_credit_reward

__all__ = ["ifeval_partial_credit_reward"]
