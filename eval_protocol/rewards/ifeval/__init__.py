"""IFEval reward function for evaluating instruction-following capabilities.

Usage:
    # Option 1: Import spacy first to avoid cupy conflicts in some Docker environments
    import spacy
    from eval_protocol.rewards.ifeval import ifeval_partial_credit_reward

    # Option 2: Direct import (add ifeval dir to path)
    import sys
    sys.path.insert(0, '/path/to/eval_protocol/rewards/ifeval')
    from reward import ifeval_partial_credit_reward

    score = ifeval_partial_credit_reward(response, ground_truth)
"""

from .reward import ifeval_partial_credit_reward

__all__ = ["ifeval_partial_credit_reward"]
