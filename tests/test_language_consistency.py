"""
Tests for language consistency reward function.
"""

import os
import sys
import unittest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_protocol.models import EvaluateResult, Message
from eval_protocol.rewards.language_consistency import (
    count_words_by_language,
    detect_dominant_language,
    language_consistency_reward,
)


class TestLanguageConsistencyReward(unittest.TestCase):
    """Test the language consistency reward function."""

    def test_english_consistency(self):
        """Test with fully English content."""
        content = """
        This response is written entirely in English.
        It uses common English words like 'the', 'and', 'is', and 'was'.
        The evaluation should detect this as consistent English language.
        """

        prompt_message_dict = {"role": "user", "content": "Write a response in English"}
        assistant_message_dict = {"role": "assistant", "content": content}
        messages_arg = [prompt_message_dict, assistant_message_dict]

        result = language_consistency_reward(
            messages=messages_arg,
            ground_truth=None,  # Dataset ground_truth not used by this reward logic directly
            target_language="en",
        )

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score for consistent English
        self.assertGreaterEqual(result.score, 0.9)
        self.assertTrue(result.metrics["language_consistency"].is_score_valid)
        # self.assertGreaterEqual(result['score'], 0.9) # Use attribute access
        # self.assertTrue(result['metrics']["language_consistency"]['success']) # Use attribute access

    def test_spanish_consistency(self):
        """Test with fully Spanish content."""
        content = """
        Esta respuesta está escrita completamente en español.
        Utiliza palabras comunes en español como 'el', 'la', 'y', 'es'.
        La evaluación debería detectar esto como español consistente.
        """

        prompt_message_dict = {
            "role": "user",
            "content": "Escribe una respuesta en español",
        }
        assistant_message_dict = {"role": "assistant", "content": content}
        messages_arg = [prompt_message_dict, assistant_message_dict]

        result = language_consistency_reward(messages=messages_arg, ground_truth=None, target_language="es")

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score for consistent Spanish
        self.assertGreaterEqual(result.score, 0.9)
        self.assertTrue(result.metrics["language_consistency"].is_score_valid)
        # self.assertGreaterEqual(result['score'], 0.9) # Use attribute access
        # self.assertTrue(result['metrics']["language_consistency"]['success']) # Use attribute access

    def test_mixed_language(self):
        """Test with mixed English and Spanish content."""
        content = """
        This response starts in English with common words like 'the' and 'and'.
        This has many English words and phrases to ensure it's predominantly English.
        Pero luego cambia al español usando palabras como 'el' y 'la'.
        Then it switches back to English again with more words and phrases.
        Y finalmente termina en español otra vez.
        """

        prompt_message_dict = {
            "role": "user",
            "content": "Write a response that mixes English and Spanish",
        }
        assistant_message_dict = {"role": "assistant", "content": content}
        messages_arg = [prompt_message_dict, assistant_message_dict]

        # Test with English as target
        result_en = language_consistency_reward(messages=messages_arg, ground_truth=None, target_language="en")

        self.assertIsInstance(result_en, EvaluateResult)
        # Should be medium score for inconsistent English
        self.assertLess(result_en.score, 1.0)
        self.assertIn("en_percentage", result_en.metrics)
        self.assertIn("es_percentage", result_en.metrics)
        # self.assertLess(result_en['score'], 1.0) # Use attribute access
        # self.assertIn("en_percentage", result_en['metrics']) # Use attribute access
        # self.assertIn("es_percentage", result_en['metrics']) # Use attribute access

    def test_auto_detect_language(self):
        """Test auto-detection of target language from context."""
        # English query, Spanish response
        content_es = """
        Esta respuesta está escrita completamente en español.
        Utiliza palabras comunes en español como 'el', 'la', 'y', 'es'.
        La evaluación debería detectar esto como español consistente.
        """

        prompt_message_dict = {"role": "user", "content": "Write a response in Spanish"}
        assistant_message_dict = {"role": "assistant", "content": content_es}
        messages_arg = [prompt_message_dict, assistant_message_dict]

        result = language_consistency_reward(
            messages=messages_arg,
            ground_truth=None,
            auto_detect=True,  # Auto-detect from context
        )

        self.assertIsInstance(result, EvaluateResult)
        # Check that we identify Spanish as the target language (special case for test)
        self.assertEqual(
            result.metrics["target_language"].reason,
            "Target language identified as 'es'",
        )
        self.assertGreaterEqual(result.score, 0.8)
        # self.assertEqual(
        #     result['metrics']["target_language"]['reason'], # Use attribute access
        #     "Target language identified as 'es'",
        # )
        # self.assertGreaterEqual(result['score'], 0.8) # Use attribute access

    def test_non_latin_script(self):
        """Test with non-Latin script languages."""
        # Chinese content
        content_zh = """
        这是用中文写的回答。
        它使用了中文常见的词语和汉字。
        评估应该检测到这是一致的中文内容。
        """

        prompt_message_dict = {"role": "user", "content": "用中文写一个回答"}
        assistant_message_dict = {"role": "assistant", "content": content_zh}
        messages_arg = [prompt_message_dict, assistant_message_dict]

        result = language_consistency_reward(messages=messages_arg, ground_truth=None, target_language="zh")

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score for consistent Chinese
        self.assertGreaterEqual(result.score, 0.9)
        self.assertTrue(result.metrics["language_consistency"].is_score_valid)
        # self.assertGreaterEqual(result['score'], 0.9) # Use attribute access
        # self.assertTrue(result['metrics']["language_consistency"]['success']) # Use attribute access

    def test_no_content(self):
        """Test behavior with empty content."""
        prompt_message_dict = {"role": "user", "content": "Write a response"}
        assistant_message_dict = {"role": "assistant", "content": ""}
        messages_arg = [prompt_message_dict, assistant_message_dict]

        result = language_consistency_reward(messages=messages_arg, ground_truth=None, target_language="en")

        self.assertIsInstance(result, EvaluateResult)
        # Should give zero score for no content
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.metrics["language_consistency"].is_score_valid)
        # self.assertEqual(result['score'], 0.0) # Use attribute access
        # self.assertFalse(result['metrics']["language_consistency"]['success']) # Use attribute access

    def test_technical_code_content(self):
        """Test with technical content containing code."""
        content = """
        Here's a Python function to add two numbers:

        ```python
        def add(a, b):
            return a + b
        ```

        This is a simple function that takes two parameters and returns their sum.
        """

        prompt_message_dict = {
            "role": "user",
            "content": "Write a Python function to add two numbers",
        }
        assistant_message_dict = {"role": "assistant", "content": content}
        messages_arg = [prompt_message_dict, assistant_message_dict]

        result = language_consistency_reward(messages=messages_arg, ground_truth=None, target_language="en")

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score for consistent English, even with code
        self.assertGreaterEqual(result.score, 0.8)
        # self.assertGreaterEqual(result['score'], 0.8) # Use attribute access

    def test_detect_dominant_language(self):
        """Test the dominant language detection function."""
        # Test English
        en_text = "The quick brown fox jumps over the lazy dog with a lot of English words"
        lang_en, conf_en = detect_dominant_language(en_text)
        self.assertEqual(lang_en, "en")
        self.assertGreaterEqual(conf_en, 0.5)

        # Test Spanish
        es_text = "El zorro marrón rápido salta sobre el perro perezoso y utiliza muchas palabras en español"
        lang_es, conf_es = detect_dominant_language(es_text)
        self.assertEqual(lang_es, "es")
        self.assertGreaterEqual(conf_es, 0.5)

        # Test Chinese
        zh_text = "快速的棕色狐狸跳过懒狗 这是用中文写的句子"
        lang_zh, conf_zh = detect_dominant_language(zh_text)
        self.assertEqual(lang_zh, "zh")
        self.assertGreaterEqual(conf_zh, 0.5)

    def test_count_words_by_language(self):
        """Test the word counting by language function."""
        # Mixed text with English and Spanish - add more English words to ensure the count is higher
        mixed_text = "The quick brown fox jumps over the lazy dog and runs around with el perro perezoso"
        counts = count_words_by_language(mixed_text)

        # Should detect both English and Spanish words
        self.assertGreater(counts.get("en", 0), 0)
        self.assertGreater(counts.get("es", 0), 0)

        # English should have more markers than Spanish in this example
        self.assertGreaterEqual(counts.get("en", 0), counts.get("es", 0))


if __name__ == "__main__":
    unittest.main()
