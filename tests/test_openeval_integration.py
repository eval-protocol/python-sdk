import unittest

try:
    from openevals import exact_match
    from openevals.json import create_json_match_evaluator
    from openevals.string import levenshtein_distance

    OPENEVALS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OPENEVALS_AVAILABLE = False

from eval_protocol.integrations.openeval import adapt
from eval_protocol.models import EvaluateResult


class TestOpenEvalIntegration(unittest.TestCase):
    @unittest.skipUnless(OPENEVALS_AVAILABLE, "openevals package is required")
    def test_exact_match_wrapper(self) -> None:
        wrapped = adapt(exact_match)
        messages = [{"role": "assistant", "content": "hi"}]
        result = wrapped(messages=messages, ground_truth="hi")
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)

    @unittest.skipUnless(OPENEVALS_AVAILABLE, "openevals package is required")
    def test_levenshtein_distance_wrapper(self) -> None:
        wrapped = adapt(levenshtein_distance)
        messages = [{"role": "assistant", "content": "foo"}]
        result = wrapped(messages=messages, ground_truth="fooo")
        self.assertIsInstance(result, EvaluateResult)
        self.assertAlmostEqual(result.score, 0.75)

    @unittest.skipUnless(OPENEVALS_AVAILABLE, "openevals package is required")
    def test_json_match_wrapper(self) -> None:
        evaluator = create_json_match_evaluator(aggregator="average")
        wrapped = adapt(evaluator)
        messages = [{"role": "assistant", "content": {"a": 1, "b": 2}}]
        result = wrapped(messages=messages, ground_truth={"a": 1, "b": 2})
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
