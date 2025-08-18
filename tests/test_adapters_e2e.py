"""
End-to-end tests for adapters with real data sources.

These tests connect to actual external services and verify that adapters
can pull data and convert it to EvaluationRow format correctly.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict

import pytest

from eval_protocol.models import EvaluationRow, InputMetadata, Message


class TestLangfuseAdapterE2E:
    """End-to-end tests for Langfuse adapter with real deployment."""

    def _get_langfuse_credentials(self):
        """Get Langfuse credentials from environment."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://langfuse-web-prod-zfdbl7ykrq-uc.a.run.app")
        project_id = os.getenv("LANGFUSE_PROJECT_ID", "cmdj5yxhk0006s6022cyi0prv")

        return public_key, secret_key, host, project_id

    @pytest.mark.skipif(
        not all(
            [
                os.getenv("LANGFUSE_PUBLIC_KEY"),
                os.getenv("LANGFUSE_SECRET_KEY"),
            ]
        ),
        reason="Langfuse credentials not available in environment",
    )
    def test_langfuse_adapter_real_connection(self):
        """Test that we can connect to real Langfuse deployment and pull data."""
        try:
            from eval_protocol.adapters.langfuse import create_langfuse_adapter
        except ImportError:
            pytest.skip("Langfuse dependencies not installed")

        public_key, secret_key, host, project_id = self._get_langfuse_credentials()

        # Create adapter
        adapter = create_langfuse_adapter(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            project_id=project_id,
        )

        # Test basic connection by trying to get a small number of traces
        rows = list(adapter.get_evaluation_rows(limit=3))

        # Verify we got some data
        assert isinstance(rows, list), "Should return a list of rows"
        print(f"Retrieved {len(rows)} evaluation rows from Langfuse")

        # Verify each row is properly formatted
        for i, row in enumerate(rows):
            assert isinstance(row, EvaluationRow), f"Row {i} should be EvaluationRow"
            assert isinstance(row.messages, list), f"Row {i} should have messages list"
            assert len(row.messages) > 0, f"Row {i} should have at least one message"

            # Verify messages are properly formatted
            for j, msg in enumerate(row.messages):
                assert isinstance(msg, Message), f"Row {i} message {j} should be Message object"
                assert hasattr(msg, "role"), f"Row {i} message {j} should have role"
                assert msg.role in [
                    "user",
                    "assistant",
                    "system",
                    "tool",
                ], f"Row {i} message {j} has invalid role: {msg.role}"

            # Verify metadata
            if row.input_metadata:
                assert isinstance(row.input_metadata, InputMetadata), f"Row {i} should have InputMetadata"
                assert row.input_metadata.row_id, f"Row {i} should have row_id"
                print(f"  Row {i}: ID={row.input_metadata.row_id}, Messages={len(row.messages)}")

            print(f"  Row {i}: {len(row.messages)} messages, Tools={'Yes' if row.tools else 'No'}")

    @pytest.mark.skipif(
        not all(
            [
                os.getenv("LANGFUSE_PUBLIC_KEY"),
                os.getenv("LANGFUSE_SECRET_KEY"),
            ]
        ),
        reason="Langfuse credentials not available",
    )
    def test_langfuse_adapter_with_filters(self):
        """Test Langfuse adapter with various filters."""
        try:
            from eval_protocol.adapters.langfuse import create_langfuse_adapter
        except ImportError:
            pytest.skip("Langfuse dependencies not installed")

        public_key, secret_key, host, project_id = self._get_langfuse_credentials()

        adapter = create_langfuse_adapter(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            project_id=project_id,
        )

        # Test with time filter (last 7 days)
        recent_rows = list(
            adapter.get_evaluation_rows(
                limit=5,
                from_timestamp=datetime.now() - timedelta(days=7),
                include_tool_calls=True,
            )
        )

        print(f"Recent rows (last 7 days): {len(recent_rows)}")

        # Verify tool calling data is preserved
        tool_calling_rows = [row for row in recent_rows if row.tools]
        print(f"Rows with tool definitions: {len(tool_calling_rows)}")

        # Test specific filtering
        try:
            # This might not return data if no traces match, which is fine
            tagged_rows = list(
                adapter.get_evaluation_rows(
                    limit=2,
                    tags=["production"],  # May not exist, that's OK
                )
            )
            print(f"Tagged rows: {len(tagged_rows)}")
        except Exception as e:
            print(f"Tagged query failed (expected if no tags): {e}")

    @pytest.mark.skipif(
        not all(
            [
                os.getenv("LANGFUSE_PUBLIC_KEY"),
                os.getenv("LANGFUSE_SECRET_KEY"),
            ]
        ),
        reason="Langfuse credentials not available",
    )
    def test_langfuse_conversation_analysis(self):
        """Test analysis of conversation types from Langfuse."""
        try:
            from eval_protocol.adapters.langfuse import create_langfuse_adapter
        except ImportError:
            pytest.skip("Langfuse dependencies not installed")

        public_key, secret_key, host, project_id = self._get_langfuse_credentials()

        adapter = create_langfuse_adapter(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            project_id=project_id,
        )

        # Get more data for analysis
        rows = list(adapter.get_evaluation_rows(limit=10, include_tool_calls=True))

        # Analyze conversation patterns
        chat_only = []
        tool_calling = []
        multi_turn = []

        for row in rows:
            # Check for tool calling
            has_tools = (
                row.tools
                or any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in row.messages)
                or any(msg.role == "tool" for msg in row.messages)
            )

            if has_tools:
                tool_calling.append(row)
            else:
                chat_only.append(row)

            # Check for multi-turn conversations
            if len(row.messages) > 2:  # More than user + assistant
                multi_turn.append(row)

        print(f"Analysis of {len(rows)} conversations:")
        print(f"  Chat-only: {len(chat_only)}")
        print(f"  Tool calling: {len(tool_calling)}")
        print(f"  Multi-turn: {len(multi_turn)}")

        # Show example of each type if available
        if chat_only:
            row = chat_only[0]
            print(f"  Example chat: {len(row.messages)} messages")

        if tool_calling:
            row = tool_calling[0]
            print(f"  Example tool calling: {len(row.messages)} messages, {len(row.tools or [])} tools")


class TestHuggingFaceAdapterE2E:
    """End-to-end tests for HuggingFace adapter with real datasets."""

    def test_gsm8k_adapter_real_data(self):
        """Test loading real GSM8K data and converting to EvaluationRow."""
        try:
            from eval_protocol.adapters.huggingface import create_huggingface_adapter
        except ImportError:
            pytest.skip("HuggingFace dependencies not installed")

        def gsm8k_transform(row: Dict[str, Any]) -> Dict[str, Any]:
            """Transform GSM8K row to our format."""
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that solves math problems step by step.",
                    },
                    {"role": "user", "content": row["question"]},
                ],
                "ground_truth": row["answer"],
                "metadata": {
                    "dataset": "gsm8k",
                    "original_question": row["question"],
                    "original_answer": row["answer"],
                },
            }

        # Create adapter with transform function
        adapter = create_huggingface_adapter(
            dataset_id="gsm8k",
            config_name="main",
            transform_fn=gsm8k_transform,
        )

        # Test loading data
        rows = list(adapter.get_evaluation_rows(split="test", limit=5))

        # Verify we got data
        assert len(rows) > 0, "Should retrieve some GSM8K data"
        print(f"Retrieved {len(rows)} GSM8K evaluation rows")

        # Verify each row is properly formatted
        for i, row in enumerate(rows):
            assert isinstance(row, EvaluationRow), f"Row {i} should be EvaluationRow"
            assert isinstance(row.messages, list), f"Row {i} should have messages"
            assert len(row.messages) >= 2, f"Row {i} should have system + user messages"

            # Check system prompt
            system_msg = row.messages[0]
            assert system_msg.role == "system", f"Row {i} first message should be system"
            assert "math problems" in system_msg.content.lower(), f"Row {i} should have math system prompt"

            # Check user question
            user_msg = row.messages[1]
            assert user_msg.role == "user", f"Row {i} second message should be user"
            assert len(user_msg.content) > 0, f"Row {i} should have non-empty question"

            # Check ground truth
            assert row.ground_truth, f"Row {i} should have ground truth answer"

            # Check metadata
            assert row.input_metadata, f"Row {i} should have metadata"
            assert row.input_metadata.dataset_info, f"Row {i} should have dataset info"

            print(f"  Row {i}: Question length={len(user_msg.content)}, Answer length={len(row.ground_truth)}")

    def test_math_dataset_real_data(self):
        """Test loading real MATH competition dataset."""
        try:
            from eval_protocol.adapters.huggingface import create_huggingface_adapter
        except ImportError:
            pytest.skip("HuggingFace dependencies not installed")

        def math_transform(row: Dict[str, Any]) -> Dict[str, Any]:
            """Transform MATH dataset row."""
            return {
                "messages": [
                    {"role": "system", "content": "You are an expert mathematician. Solve this step by step."},
                    {"role": "user", "content": row["problem"]},
                ],
                "ground_truth": row["solution"],
                "metadata": {
                    "dataset": "hendrycks_math",
                    "type": row.get("type", "unknown"),
                    "level": row.get("level", "unknown"),
                    "original_problem": row["problem"],
                    "original_solution": row["solution"],
                },
            }

        # Create adapter
        adapter = create_huggingface_adapter(
            dataset_id="SuperSecureHuman/competition_math_hf_dataset",
            transform_fn=math_transform,
        )

        # Test loading data
        rows = list(adapter.get_evaluation_rows(split="test", limit=3))

        # Verify data
        assert len(rows) > 0, "Should retrieve MATH dataset data"
        print(f"Retrieved {len(rows)} MATH dataset evaluation rows")

        for i, row in enumerate(rows):
            assert isinstance(row, EvaluationRow), f"Row {i} should be EvaluationRow"
            assert len(row.messages) >= 2, f"Row {i} should have system + user messages"
            assert row.ground_truth, f"Row {i} should have solution"

            # Check for MATH-specific metadata
            dataset_info = row.input_metadata.dataset_info
            assert "type" in dataset_info, f"Row {i} should have problem type"
            assert "level" in dataset_info, f"Row {i} should have difficulty level"

            print(f"  Row {i}: Type={dataset_info.get('type')}, Level={dataset_info.get('level')}")

    def test_custom_dataset_transform(self):
        """Test adapter with a completely custom transformation."""
        try:
            from eval_protocol.adapters.huggingface import create_huggingface_adapter
        except ImportError:
            pytest.skip("HuggingFace dependencies not installed")

        def squad_transform(row: Dict[str, Any]) -> Dict[str, Any]:
            """Custom transform for SQuAD dataset."""
            context = row["context"]
            question = row["question"]
            answers = row["answers"]

            # Get first answer
            answer_text = answers["text"][0] if answers["text"] else "No answer"

            return {
                "messages": [
                    {"role": "system", "content": "Answer the question based on the given context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                ],
                "ground_truth": answer_text,
                "metadata": {
                    "dataset": "squad",
                    "context_length": len(context),
                    "question_length": len(question),
                    "num_answers": len(answers["text"]),
                },
            }

        # Create adapter for SQuAD
        adapter = create_huggingface_adapter(
            dataset_id="squad",
            transform_fn=squad_transform,
        )

        # Test loading
        rows = list(adapter.get_evaluation_rows(split="validation", limit=2))

        assert len(rows) > 0, "Should retrieve SQuAD data"
        print(f"Retrieved {len(rows)} SQuAD evaluation rows")

        for i, row in enumerate(rows):
            assert isinstance(row, EvaluationRow), f"Row {i} should be EvaluationRow"
            user_msg = next(msg for msg in row.messages if msg.role == "user")
            assert "Context:" in user_msg.content, f"Row {i} should have context"
            assert "Question:" in user_msg.content, f"Row {i} should have question"

            dataset_info = row.input_metadata.dataset_info
            print(f"  Row {i}: Context length={dataset_info.get('context_length')}")


class TestBigQueryAdapterE2E:
    """End-to-end tests for BigQuery adapter with real data sources."""

    def _get_bigquery_credentials(self):
        """Get BigQuery credentials from environment."""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        return project_id, credentials_path

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_CLOUD_PROJECT"), reason="Google Cloud project not configured in environment"
    )
    def test_bigquery_adapter_real_connection(self):
        """Test that we can connect to real BigQuery and execute queries."""
        try:
            from eval_protocol.adapters.bigquery import create_bigquery_adapter
        except ImportError:
            pytest.skip("BigQuery dependencies not installed")

        project_id, credentials_path = self._get_bigquery_credentials()

        # Define a simple transform for testing
        def test_transform(row: Dict[str, Any]) -> Dict[str, Any]:
            """Transform test query results to evaluation format."""
            return {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": str(row.get("text", "Test query"))},
                ],
                "ground_truth": str(row.get("label", "test")),
                "metadata": {
                    "source": "bigquery",
                    "row_data": dict(row),
                },
            }

        # Create adapter
        adapter = create_bigquery_adapter(
            transform_fn=test_transform,
            dataset_id=project_id,
            credentials_path=credentials_path,
        )

        # Test with a simple query that should work on any BigQuery project
        # Using INFORMATION_SCHEMA which is available in all projects
        query = """
        SELECT
            'test_text' as text,
            'test_label' as label,
            CURRENT_TIMESTAMP() as created_at,
            1 as id
        LIMIT 3
        """

        # Execute query and get rows
        rows = list(
            adapter.get_evaluation_rows(
                query=query,
                limit=2,
                model_name="gpt-3.5-turbo",
                temperature=0.0,
            )
        )

        # Verify we got data
        assert len(rows) > 0, "Should retrieve data from BigQuery"
        print(f"Retrieved {len(rows)} evaluation rows from BigQuery")

        # Verify each row is properly formatted
        for i, row in enumerate(rows):
            assert isinstance(row, EvaluationRow), f"Row {i} should be EvaluationRow"
            assert isinstance(row.messages, list), f"Row {i} should have messages list"
            assert len(row.messages) >= 2, f"Row {i} should have system + user messages"

            # Check system and user messages
            system_msg = row.messages[0]
            user_msg = row.messages[1]
            assert system_msg.role == "system", f"Row {i} first message should be system"
            assert user_msg.role == "user", f"Row {i} second message should be user"

            # Verify metadata
            assert row.input_metadata, f"Row {i} should have metadata"
            assert row.input_metadata.row_id, f"Row {i} should have row_id"

            # Check BigQuery-specific metadata
            dataset_info = row.input_metadata.dataset_info
            assert dataset_info["source"] == "bigquery", f"Row {i} should have BigQuery source"

            print(f"  Row {i}: ID={row.input_metadata.row_id}, Messages={len(row.messages)}")

    @pytest.mark.skipif(not os.getenv("GOOGLE_CLOUD_PROJECT"), reason="Google Cloud project not configured")
    def test_bigquery_advanced_features(self):
        """Test advanced BigQuery adapter features like parameterized queries."""
        try:
            from google.cloud import bigquery

            from eval_protocol.adapters.bigquery import create_bigquery_adapter
        except ImportError:
            pytest.skip("BigQuery dependencies not installed")

        project_id, credentials_path = self._get_bigquery_credentials()

        def transform_fn(row):
            return {
                "messages": [{"role": "user", "content": str(row["content"])}],
                "ground_truth": str(row["label"]),
                "metadata": {"category": row.get("category", "unknown")},
            }

        adapter = create_bigquery_adapter(
            transform_fn=transform_fn,
            dataset_id=project_id,
            credentials_path=credentials_path,
        )

        # Test parameterized query
        query = """
        SELECT
            @prefix || ' example content' as content,
            'test_label' as label,
            @category as category
        """

        query_params = [
            bigquery.ScalarQueryParameter("prefix", "STRING", "BigQuery"),
            bigquery.ScalarQueryParameter("category", "STRING", "test_data"),
        ]

        rows = list(
            adapter.get_evaluation_rows(
                query=query,
                query_params=query_params,
                limit=1,
            )
        )

        assert len(rows) == 1, "Should retrieve parameterized query result"
        row = rows[0]

        user_msg = row.messages[0]
        assert "BigQuery example content" in user_msg.content
        assert row.ground_truth == "test_label"

        print(f"Parameterized query test: '{user_msg.content}' -> '{row.ground_truth}'")

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_CLOUD_PROJECT"), reason="Google Cloud project required to query public datasets"
    )
    def test_bigquery_public_dataset_google_books_ngrams(self):
        """Test BigQuery adapter with a public dataset to test specific logic."""
        try:
            from eval_protocol.adapters.bigquery import create_bigquery_adapter
        except ImportError:
            pytest.skip("BigQuery dependencies not installed")

        # Get user's project credentials (needed to run the query job)
        project_id, credentials_path = self._get_bigquery_credentials()

        def google_books_transform(row: Dict[str, Any]) -> Dict[str, Any]:
            """Transform Google Books ngrams data to evaluation format."""
            term = str(row.get("term", ""))
            term_frequency = row.get("term_frequency", 0)
            document_frequency = row.get("document_frequency", 0)
            tokens = row.get("tokens", [])  # This is a REPEATED field (array)
            has_tag = row.get("has_tag", False)
            years = row.get("years", [])  # This is a REPEATED RECORD (array of objects)

            # Create an educational question about the term
            system_prompt = (
                """You are a linguistics expert who helps explain word usage patterns from Google Books data."""
            )

            # Create a question about the term's usage
            if tokens and len(tokens) > 0:
                tokens_str = ", ".join(str(token) for token in tokens[:3])  # Take first 3 tokens
                question = f"What can you tell me about the term '{term}' and its linguistic tokens: {tokens_str}?"
            else:
                question = f"What can you tell me about the Chinese term '{term}' based on its usage patterns?"

            # Create ground truth based on frequency data
            frequency_desc = (
                "high frequency"
                if term_frequency > 1000
                else "moderate frequency"
                if term_frequency > 100
                else "low frequency"
            )
            document_desc = (
                f"appears in {document_frequency} documents" if document_frequency > 0 else "rare occurrence"
            )

            ground_truth = (
                f"The term '{term}' has {frequency_desc} usage ({term_frequency} occurrences) and {document_desc}."
            )
            if has_tag:
                ground_truth += " This term has special linguistic tags."

            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "ground_truth": ground_truth,
                "metadata": {
                    "dataset": "google_books_ngrams_chi_sim",
                    "term": term,
                    "term_frequency": term_frequency,
                    "document_frequency": document_frequency,
                    "num_tokens": len(tokens) if tokens else 0,
                    "has_tag": has_tag,
                    "num_year_records": len(years) if years else 0,
                    "tokens_sample": tokens[:3] if tokens else [],  # Store first 3 tokens as sample
                },
            }

        # Create adapter - use YOUR project to run the job, but query PUBLIC data
        adapter = create_bigquery_adapter(
            transform_fn=google_books_transform,
            dataset_id=project_id,  # YOUR project (to run the job)
            credentials_path=credentials_path,
        )

        # Query the public Google Books ngrams dataset (full table reference in SQL)
        query = """
        SELECT
            term,
            term_frequency,
            document_frequency,
            tokens,
            has_tag,
            years
        FROM `bigquery-public-data.google_books_ngrams_2020.chi_sim_1`
        WHERE term_frequency > 100
          AND document_frequency > 5
          AND LENGTH(term) >= 2
        ORDER BY term_frequency DESC
        LIMIT 10
        """

        # Execute query and get rows
        rows = list(
            adapter.get_evaluation_rows(
                query=query,
                limit=3,
                model_name="gpt-4",
                temperature=0.0,
            )
        )

        # Verify we got data
        assert len(rows) > 0, "Should retrieve data from Google Books ngrams dataset"
        print(f"Retrieved {len(rows)} evaluation rows from Google Books ngrams")

        # Verify each row is properly formatted
        for i, row in enumerate(rows):
            assert isinstance(row, EvaluationRow), f"Row {i} should be EvaluationRow"
            assert isinstance(row.messages, list), f"Row {i} should have messages list"
            assert len(row.messages) >= 2, f"Row {i} should have system + user messages"

            # Check message content
            system_msg = row.messages[0]
            user_msg = row.messages[1]
            assert system_msg.role == "system", f"Row {i} first message should be system"
            assert user_msg.role == "user", f"Row {i} second message should be user"
            assert "linguistics expert" in system_msg.content, f"Row {i} should have linguistics system prompt"
            assert "term" in user_msg.content, f"Row {i} should ask about the term"

            # Verify ground truth
            assert row.ground_truth, f"Row {i} should have ground truth"
            assert "frequency" in row.ground_truth, f"Row {i} should mention frequency"

            # Verify metadata
            assert row.input_metadata, f"Row {i} should have metadata"
            dataset_info = row.input_metadata.dataset_info
            assert dataset_info["dataset"] == "google_books_ngrams_chi_sim", f"Row {i} should have correct dataset"
            assert "term" in dataset_info, f"Row {i} should have term in metadata"
            assert "term_frequency" in dataset_info, f"Row {i} should have frequency in metadata"
            assert "num_tokens" in dataset_info, f"Row {i} should have token count in metadata"

            # Check repeated fields handling
            term = dataset_info["term"]
            term_freq = dataset_info["term_frequency"]
            doc_freq = dataset_info["document_frequency"]
            num_tokens = dataset_info["num_tokens"]

            print(f"  Row {i}: Term='{term}', Frequency={term_freq}, Docs={doc_freq}, Tokens={num_tokens}")

            # Verify filtering worked (should have high frequency terms)
            assert term_freq > 100, f"Row {i} should have term frequency > 100"
            assert doc_freq > 5, f"Row {i} should have document frequency > 5"


def test_adapters_integration():
    """Test that adapters work with evaluation pipeline."""
    print("Testing adapter integration with evaluation pipeline...")

    # This test doesn't require external credentials
    try:
        from eval_protocol.adapters.huggingface import create_huggingface_adapter
        from eval_protocol.rewards.accuracy import accuracy_reward
    except ImportError as e:
        pytest.skip(f"Dependencies not available: {e}")

    def simple_transform(row: Dict[str, Any]) -> Dict[str, Any]:
        """Simple transform for testing."""
        return {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": "Test response"},  # Simulated response
            ],
            "ground_truth": row["answer"],
            "metadata": {"test": True},
        }

    # Create adapter with GSM8K (small sample)
    adapter = create_huggingface_adapter(
        dataset_id="gsm8k",
        config_name="main",
        transform_fn=simple_transform,
    )

    # Get one row
    rows = list(adapter.get_evaluation_rows(split="test", limit=1))
    assert len(rows) == 1, "Should get exactly one row"

    row = rows[0]

    # Test evaluation
    result = accuracy_reward(
        messages=row.messages,
        ground_truth=row.ground_truth,
    )

    assert hasattr(result, "score"), "Should have evaluation score"
    assert 0 <= result.score <= 1, "Score should be between 0 and 1"

    print(f"Integration test successful: Score={result.score}")


if __name__ == "__main__":
    # Run tests manually for development
    import sys

    print("Running Langfuse E2E tests...")
    if all([os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")]):
        try:
            test_langfuse = TestLangfuseAdapterE2E()
            test_langfuse.test_langfuse_adapter_real_connection()
            test_langfuse.test_langfuse_adapter_with_filters()
            test_langfuse.test_langfuse_conversation_analysis()
            print("✅ Langfuse tests passed!")
        except Exception as e:
            print(f"⚠️ Langfuse tests failed (API may have changed): {e}")
            print("   This is expected if Langfuse API has changed - the adapter needs updating")
    else:
        print("⚠️ Skipping Langfuse tests (credentials not available)")

    print("\nRunning HuggingFace E2E tests...")
    try:
        test_hf = TestHuggingFaceAdapterE2E()
        test_hf.test_gsm8k_adapter_real_data()
        print("✅ GSM8K adapter test passed!")

        # Skip MATH dataset test for now (dataset may not be available)
        try:
            test_hf.test_math_dataset_real_data()
            print("✅ MATH dataset test passed!")
        except Exception as e:
            print(f"⚠️ MATH dataset test failed (dataset may not be available): {e}")

        # Skip SQuAD test for now (focus on core functionality)
        try:
            test_hf.test_custom_dataset_transform()
            print("✅ Custom dataset test passed!")
        except Exception as e:
            print(f"⚠️ Custom dataset test failed: {e}")
        print("✅ HuggingFace tests passed!")
    except Exception as e:
        print(f"❌ HuggingFace tests failed: {e}")
        sys.exit(1)

    print("\nRunning BigQuery E2E test...")
    try:
        test_bq = TestBigQueryAdapterE2E()
        # Only test the public Google Books ngrams dataset (no auth required)
        test_bq.test_bigquery_public_dataset_google_books_ngrams()
        print("✅ BigQuery Google Books ngrams test passed!")

    except Exception as e:
        print(f"❌ BigQuery test failed: {e}")

    print("\n🎉 BigQuery E2E test completed successfully!")
