import unittest
from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluateResult, Message
from eval_protocol.rewards.list_comparison_math_reward import (
    extract_number_list,
    list_comparison_math_reward,
    parse_number_list_from_string,
)


class TestParseNumberListFromString(unittest.TestCase):
    """Test the raw string to number list parsing utility."""

    def test_simple_list(self):
        self.assertEqual(parse_number_list_from_string("1,2,3"), [1.0, 2.0, 3.0])
        self.assertEqual(parse_number_list_from_string("1, 2.5, 3"), [1.0, 2.5, 3.0])

    def test_with_spaces(self):
        self.assertEqual(parse_number_list_from_string("  1 , 2 , 3  "), [1.0, 2.0, 3.0])

    def test_with_dollar_signs(self):
        # parse_number_list_from_string itself removes $
        self.assertEqual(parse_number_list_from_string("$1,2,3$"), [1.0, 2.0, 3.0])
        self.assertEqual(parse_number_list_from_string("1, $2$, 3"), [1.0, 2.0, 3.0])

    def test_single_number(self):
        self.assertEqual(parse_number_list_from_string("42"), [42.0])
        self.assertEqual(parse_number_list_from_string("$123.45$"), [123.45])

    def test_empty_and_invalid(self):
        self.assertIsNone(parse_number_list_from_string(""))
        self.assertIsNone(parse_number_list_from_string("  "))
        self.assertIsNone(parse_number_list_from_string("1,two,3"))
        self.assertIsNone(parse_number_list_from_string("1, 2, three"))
        self.assertIsNone(parse_number_list_from_string("abc"))

    def test_list_with_empty_elements(self):
        # "1,,2" -> [1.0, 2.0] because empty string from split is skipped
        self.assertEqual(parse_number_list_from_string("1,,2"), [1.0, 2.0])
        self.assertEqual(parse_number_list_from_string("1, ,2"), [1.0, 2.0])


class TestExtractNumberList(unittest.TestCase):
    """Test the number list extraction from text with delimiters."""

    def test_extract_from_boxed(self):
        self.assertEqual(extract_number_list("The answer is \\boxed{1,2,3}."), [[1.0, 2.0, 3.0]])
        self.assertEqual(
            extract_number_list("\\boxed{1,2} and \\boxed{3,4}"),
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.assertEqual(extract_number_list("Boxed: \\boxed{1, 2.5, 3}"), [[1.0, 2.5, 3.0]])

    def test_extract_from_dollar(self):
        self.assertEqual(extract_number_list("The set is $1,2,3$."), [[1.0, 2.0, 3.0]])
        self.assertEqual(extract_number_list("Numbers are $$1, 2, 3$$"), [[1.0, 2.0, 3.0]])
        self.assertEqual(extract_number_list("$1,2$ and also $3,4$"), [[1.0, 2.0], [3.0, 4.0]])

    def test_priority_boxed_over_dollar(self):
        # If boxed is found, dollar outside is ignored by current logic (returns after finding in boxed)
        self.assertEqual(extract_number_list("$\\boxed{1,2}$ $3,4$"), [[1.0, 2.0]])
        # If multiple boxed, all are returned
        self.assertEqual(
            extract_number_list("$\\boxed{1,2}$ and \\boxed{5,6} $3,4$"),
            [[1.0, 2.0], [5.0, 6.0]],
        )

    def test_priority_dollar_over_full_text(self):
        self.assertEqual(
            extract_number_list("The list $1,2,3$ is correct, not 4,5,6."),
            [[1.0, 2.0, 3.0]],
        )

    def test_extract_from_full_text_fallback(self):
        # Fallback only works if the *entire string* (after $ stripping) is a valid list
        self.assertEqual(extract_number_list("1, 2.5, 3"), [[1.0, 2.5, 3.0]])
        self.assertEqual(
            extract_number_list("  $1, 2, 3$  "), [[1.0, 2.0, 3.0]]
        )  # Handled by parse_number_list_from_string
        # These should not parse as lists because of surrounding text
        self.assertEqual(extract_number_list("The numbers are 1,2,3."), [])
        self.assertEqual(extract_number_list("The number is 123."), [])

    def test_no_valid_list(self):
        self.assertEqual(extract_number_list("No numbers here."), [])
        self.assertEqual(extract_number_list("\\boxed{a,b,c}"), [])
        self.assertEqual(extract_number_list("$one, two$"), [])
        self.assertEqual(extract_number_list("The number is 123."), [])

    def test_mixed_delimiters_behavior(self):
        # Current logic: if boxed found, returns only boxed. If no boxed but dollar found, returns only dollar.
        self.assertEqual(extract_number_list("Box \\boxed{1,2} and dollar $3,4$"), [[1.0, 2.0]])
        self.assertEqual(extract_number_list("Dollar $3,4$ and text 5,6"), [[3.0, 4.0]])


class TestListComparisonMathReward(unittest.TestCase):
    """Test the list_comparison_math_reward function."""

    def _create_messages(self, assistant_content: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": "What are the numbers?"},
            {"role": "assistant", "content": assistant_content},
        ]

    def test_exact_match_set_comparison(self):
        # Default: order_matters=False (set comparison)
        gen_msgs = self._create_messages("The divisors are \\boxed{1,2,3}.")
        ground_truth_str = "Answer: $3,1,2$"
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.reason and "Set match" in result.reason)
        self.assertTrue(result.reason and "Gen: [1.0, 2.0, 3.0] vs Orig: [1.0, 2.0, 3.0]" in result.reason)

    def test_exact_match_list_comparison(self):
        gen_msgs = self._create_messages("\\boxed{1,2,3}")
        ground_truth_str = "$1,2,3$"
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str, order_matters=True)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.reason and "Exact list match" in result.reason)

    def test_mismatch_set_comparison(self):
        gen_msgs = self._create_messages("$1,2,4$")
        ground_truth_str = "\\boxed{1,2,3}"
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.reason and "Set mismatch" in result.reason)
        self.assertTrue(result.reason and "Missing in generated: [3.0]" in result.reason)
        self.assertTrue(result.reason and "Extra in generated: [4.0]" in result.reason)

    def test_mismatch_list_comparison_order(self):
        gen_msgs = self._create_messages("$1,3,2$")
        ground_truth_str = "\\boxed{1,2,3}"
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str, order_matters=True)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.reason and "List mismatch (order matters)" in result.reason)

    def test_mismatch_list_comparison_value(self):
        gen_msgs = self._create_messages("$1,2,4$")
        ground_truth_str = "\\boxed{1,2,3}"
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str, order_matters=True)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.reason and "List mismatch (order matters)" in result.reason)

    def test_subset_set_comparison(self):
        # Gen is subset of Orig
        gen_msgs = self._create_messages("$1,2$")
        ground_truth_str = "\\boxed{1,2,3}"
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.reason and "Missing in generated: [3.0]" in result.reason)
        self.assertTrue(result.reason and "Extra in generated" not in result.reason)

    def test_superset_set_comparison(self):
        # Gen is superset of Orig
        gen_msgs = self._create_messages("$1,2,3,4$")
        ground_truth_str = "\\boxed{1,2,3}"
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.reason and "Extra in generated: [4.0]" in result.reason)
        self.assertTrue(result.reason and "Missing in generated" not in result.reason)

    def test_no_list_in_gen(self):
        gen_msgs = self._create_messages("The answer is three.")
        ground_truth_str = "\\boxed{1,2,3}"
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.reason and "Could not extract any number list from generated message" in result.reason)

    def test_no_list_in_orig(self):
        gen_msgs = self._create_messages("\\boxed{1,2,3}")
        ground_truth_str = "The answer is three."
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.reason and "Could not extract any number list from original message" in result.reason)

    def test_example_from_issue(self):
        # "1,3,59,177"
        gen_msgs = self._create_messages("$1,3,59,177$")
        ground_truth_str = "1,3,59,177"  # Fallback extraction for original
        result = list_comparison_math_reward(messages=gen_msgs, ground_truth=ground_truth_str)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.reason and "Set match" in result.reason)
        self.assertTrue(
            result.reason and "Gen: [1.0, 3.0, 59.0, 177.0] vs Orig: [1.0, 3.0, 59.0, 177.0]" in result.reason
        )

    def test_empty_messages(self):
        # Test with empty messages list, should fail early due to messages[-1] access
        result = list_comparison_math_reward(messages=[], ground_truth="1,2,3")
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.reason and "Invalid or missing assistant response" in result.reason)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
