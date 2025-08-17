import importlib

_mod = importlib.import_module("eval_protocol.benchmarks.suites.rewardbench2_ties_ratings")

test_rewardbench2_ties_ratings = getattr(_mod, "test_rewardbench2_ties_ratings")  # noqa: F401


