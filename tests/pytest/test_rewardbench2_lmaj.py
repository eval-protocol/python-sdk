# Re-export the RewardBench2 LM-as-a-judge test so pytest can collect it from a test_* module.
# Import via package root to avoid linter import path confusion.
import importlib

_mod = importlib.import_module("eval_protocol.benchmarks.suites.rewardbench2")

test_rewardbench2_lmaj_pointwise = getattr(_mod, "test_rewardbench2_lmaj_pointwise")  # noqa: F401


