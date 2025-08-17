# Suite modules are auto-imported by eval_protocol.benchmarks.run to register benchmarks.
from . import rewardbench2  # noqa: F401
from . import rewardbench2_ties_ratings  # noqa: F401
# Scratch validation suite for RB2 Ties
from . import rb2_ties_scratch  # noqa: F401
# Other suites are imported implicitly by test re-exports or direct usage


