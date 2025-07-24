import unittest
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, Message
from eval_protocol.rewards.multiple_choice_math_reward import (
    extract_mcq_option,
    multiple_choice_math_reward,
)


class TestExtractMCQOption(unittest.TestCase):
    """Test the MCQ option extraction utility."""

    def test_basic_extraction(self):
        self.assertEqual(extract_mcq_option("The answer is (A)."), [("(A)", "A")])
        self.assertEqual(extract_mcq_option("Choose B. for this one."), [("B.", "B")])
        self.assertEqual(extract_mcq_option("It must be [C]"), [("[C]", "C")])
        self.assertEqual(extract_mcq_option("Perhaps D is correct."), [("D", "D")])  # Standalone D followed by space
        self.assertEqual(extract_mcq_option("The final choice is E"), [("E", "E")])  # Standalone E at end of string
        self.assertEqual(extract_mcq_option("Answer: {A}"), [("{A}", "A")])

    def test_multiple_options_found(self):
        # Should extract all unique options it finds based on the pattern
        self.assertEqual(extract_mcq_option("Is it (A) or (B)?"), [("(A)", "A"), ("(B)", "B")])

    def test_no_mcq_option(self):
        self.assertEqual(extract_mcq_option("The answer is 123."), [])
        self.assertEqual(extract_mcq_option("This is just text."), [])
        self.assertEqual(extract_mcq_option("Variable v_A should be used."), [])  # Avoid 'A' in 'v_A'

    def test_case_insensitivity(self):
        self.assertEqual(extract_mcq_option("the option is (c)"), [("(c)", "C")])

    def test_various_formats(self):
        self.assertEqual(extract_mcq_option(" (A) "), [("(A)", "A")])
        self.assertEqual(extract_mcq_option("A. B. C."), [("A.", "A"), ("B.", "B"), ("C.", "C")])
        self.assertEqual(extract_mcq_option("The answer is A"), [("A", "A")])


class TestMultipleChoiceMathReward(unittest.TestCase):
    """Test the multiple_choice_math_reward function."""

    def _create_messages(self, assistant_content: str) -> List[Dict[str, str]]:
        """Creates a list of messages including a user prompt and an assistant response."""
        return [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": assistant_content},
        ]

    def _create_ground_truth(self, assistant_content: str) -> List[Dict[str, str]]:
        """Creates a ground_truth list containing a single assistant message."""
        return [{"role": "assistant", "content": assistant_content}]

    def test_perfect_match_parentheses(self):
        gen_msgs = self._create_messages("The correct option is (B).")
        gt_msgs = self._create_ground_truth("The answer is (B).")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["mcq_comparison"].is_score_valid)
        self.assertTrue(result.reason is not None and "Gen: '(B)' (B) vs Orig: '(B)' (B)" in result.reason)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["mcq_comparison"]["is_score_valid"])
        self.assertTrue(result["reason"] is not None and "Gen: '(B)' (B) vs Orig: '(B)' (B)" in result["reason"])

    def test_perfect_match_dot(self):
        gen_msgs = self._create_messages("My choice is C.")
        gt_msgs = self._create_ground_truth("C. is the one.")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result["score"], 1.0)

    def test_mismatch(self):
        gen_msgs = self._create_messages("I think it's (A).")
        gt_msgs = self._create_ground_truth("The answer is definitely (D).")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.metrics["mcq_comparison"].is_score_valid)
        self.assertTrue(result.reason is not None and "Gen: '(A)' (A) vs Orig: '(D)' (D)" in result.reason)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["metrics"]["mcq_comparison"]["is_score_valid"])
        self.assertTrue(result["reason"] is not None and "Gen: '(A)' (A) vs Orig: '(D)' (D)" in result["reason"])

    def test_gen_no_mcq_orig_has_mcq(self):
        gen_msgs = self._create_messages("The answer is 42.")
        gt_msgs = self._create_ground_truth("The answer is (A).")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertIsNotNone(result.reason)
        assert result.reason is not None  # for mypy
        self.assertTrue("Could not extract MCQ option from generated message" in result.reason)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertIsNotNone(result["reason"])
        assert result["reason"] is not None  # for mypy
        self.assertTrue("Could not extract MCQ option from generated message" in result["reason"])

    def test_orig_no_mcq(self):
        gen_msgs = self._create_messages("The answer is (B).")
        gt_msgs = self._create_ground_truth("The answer is two.")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertIsNotNone(result.reason)
        assert result.reason is not None  # for mypy
        self.assertTrue("Could not extract MCQ option from original message" in result.reason)
        self.assertTrue(result.metrics["extracted_generated_mcq"].is_score_valid)
        self.assertFalse(result.metrics["extracted_original_mcq"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertIsNotNone(result["reason"])
        assert result["reason"] is not None  # for mypy
        self.assertTrue("Could not extract MCQ option from original message" in result["reason"])
        self.assertTrue(result["metrics"]["extracted_generated_mcq"]["is_score_valid"])
        self.assertFalse(result["metrics"]["extracted_original_mcq"]["is_score_valid"])

    def test_ambiguous_generated_answer(self):
        gen_msgs = self._create_messages("It could be (A) or maybe (B).")
        gt_msgs = self._create_ground_truth("The answer is (A).")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)  # Penalized for ambiguity
        self.assertIsNotNone(result.reason)
        assert result.reason is not None  # for mypy
        self.assertTrue("Generated answer is ambiguous" in result.reason)
        self.assertTrue(
            result.metrics["ambiguous_generated_mcq"].is_score_valid == False
        )  # success is False for this metric
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertIsNotNone(result["reason"])
        assert result["reason"] is not None  # for mypy
        self.assertTrue("Generated answer is ambiguous" in result["reason"])
        self.assertTrue(result["metrics"]["ambiguous_generated_mcq"]["is_score_valid"] == False)

    def test_ambiguous_original_answer_still_compares_first(self):
        # If original is ambiguous, current logic picks the first and compares.
        gen_msgs = self._create_messages("The answer is (A).")
        gt_msgs = self._create_ground_truth("The options are (A) and (C).")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 1.0)  # Matches first extracted from original
        self.assertTrue(result.metrics["ambiguous_original_mcq"].is_score_valid == False)
        self.assertTrue(result.reason is not None and "Gen: '(A)' (A) vs Orig: '(A)' (A)" in result.reason)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["ambiguous_original_mcq"]["is_score_valid"] == False)
        self.assertTrue(result["reason"] is not None and "Gen: '(A)' (A) vs Orig: '(A)' (A)" in result["reason"])

    def test_both_ambiguous_compares_first(self):
        gen_msgs = self._create_messages("Let's say (D), or perhaps (E).")
        gt_msgs = self._create_ground_truth("Is it (D) or (A)?")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 1.0)  # D vs D
        self.assertTrue(result.metrics["ambiguous_generated_mcq"].is_score_valid == False)
        self.assertTrue(result.metrics["ambiguous_original_mcq"].is_score_valid == False)
        self.assertTrue(result.reason is not None and "Gen: '(D)' (D) vs Orig: '(D)' (D)" in result.reason)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["ambiguous_generated_mcq"]["is_score_valid"] == False)
        self.assertTrue(result["metrics"]["ambiguous_original_mcq"]["is_score_valid"] == False)
        self.assertTrue(result["reason"] is not None and "Gen: '(D)' (D) vs Orig: '(D)' (D)" in result["reason"])

    def test_empty_messages_and_ground_truth(self):
        result = multiple_choice_math_reward(messages=[], ground_truth=[])
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertIsNotNone(result.reason)
        assert result.reason is not None  # for mypy
        self.assertTrue("Missing generated messages" in result.reason)  # Checks messages first

    def test_empty_messages_only(self):
        gt_msgs = self._create_ground_truth("(A)")
        result = multiple_choice_math_reward(messages=[], ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertIsNotNone(result.reason)
        assert result.reason is not None  # for mypy
        self.assertTrue("Missing generated messages" in result.reason)

    def test_empty_ground_truth_only(self):
        gen_msgs = self._create_messages("(A)")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=[])
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertIsNotNone(result.reason)
        assert result.reason is not None  # for mypy
        self.assertTrue("Missing ground truth message" in result.reason)

    def test_missing_assistant_message_gen(self):
        # messages[-1] is not an assistant message
        gen_msgs = [
            {"role": "user", "content": "Query"},
            {"role": "user", "content": "Another query"},
        ]
        gt_msgs = self._create_ground_truth("(A)")
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertIsNotNone(result.reason)
        assert result.reason is not None  # for mypy
        self.assertTrue("Last generated message not from assistant or has no content" in result.reason)

    def test_missing_assistant_message_orig(self):
        gen_msgs = self._create_messages("(A)")
        # ground_truth[0] is not an assistant message
        gt_msgs = [{"role": "user", "content": "This is not an assistant message"}]
        result = multiple_choice_math_reward(messages=gen_msgs, ground_truth=gt_msgs)
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertIsNotNone(result.reason)
        assert result.reason is not None  # for mypy
        self.assertTrue("Invalid ground truth message: Not an assistant message or has no content" in result.reason)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
