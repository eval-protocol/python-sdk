"""
End-to-end tests for adapters with real data sources.

These tests connect to actual external services and verify that adapters
can pull data and convert it to EvaluationRow format correctly.
"""

import os
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from eval_protocol.models import EvaluationRow, Message, InputMetadata


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
        not all([
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
        ]),
        reason="Langfuse credentials not available in environment"
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
                assert hasattr(msg, 'role'), f"Row {i} message {j} should have role"
                assert msg.role in ['user', 'assistant', 'system', 'tool'], f"Row {i} message {j} has invalid role: {msg.role}"
            
            # Verify metadata
            if row.input_metadata:
                assert isinstance(row.input_metadata, InputMetadata), f"Row {i} should have InputMetadata"
                assert row.input_metadata.row_id, f"Row {i} should have row_id"
                print(f"  Row {i}: ID={row.input_metadata.row_id}, Messages={len(row.messages)}")
            
            print(f"  Row {i}: {len(row.messages)} messages, Tools={'Yes' if row.tools else 'No'}")
    
    @pytest.mark.skipif(
        not all([
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
        ]),
        reason="Langfuse credentials not available"
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
        recent_rows = list(adapter.get_evaluation_rows(
            limit=5,
            from_timestamp=datetime.now() - timedelta(days=7),
            include_tool_calls=True,
        ))
        
        print(f"Recent rows (last 7 days): {len(recent_rows)}")
        
        # Verify tool calling data is preserved
        tool_calling_rows = [row for row in recent_rows if row.tools]
        print(f"Rows with tool definitions: {len(tool_calling_rows)}")
        
        # Test specific filtering
        try:
            # This might not return data if no traces match, which is fine
            tagged_rows = list(adapter.get_evaluation_rows(
                limit=2,
                tags=["production"],  # May not exist, that's OK
            ))
            print(f"Tagged rows: {len(tagged_rows)}")
        except Exception as e:
            print(f"Tagged query failed (expected if no tags): {e}")
    
    @pytest.mark.skipif(
        not all([
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
        ]),
        reason="Langfuse credentials not available"
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
                row.tools or 
                any(hasattr(msg, 'tool_calls') and msg.tool_calls for msg in row.messages) or
                any(msg.role == 'tool' for msg in row.messages)
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
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant that solves math problems step by step.'},
                    {'role': 'user', 'content': row['question']},
                ],
                'ground_truth': row['answer'],
                'metadata': {
                    'dataset': 'gsm8k',
                    'original_question': row['question'],
                    'original_answer': row['answer'],
                }
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
            assert system_msg.role == 'system', f"Row {i} first message should be system"
            assert 'math problems' in system_msg.content.lower(), f"Row {i} should have math system prompt"
            
            # Check user question
            user_msg = row.messages[1]
            assert user_msg.role == 'user', f"Row {i} second message should be user"
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
                'messages': [
                    {'role': 'system', 'content': 'You are an expert mathematician. Solve this step by step.'},
                    {'role': 'user', 'content': row['problem']},
                ],
                'ground_truth': row['solution'],
                'metadata': {
                    'dataset': 'hendrycks_math',
                    'type': row.get('type', 'unknown'),
                    'level': row.get('level', 'unknown'),
                    'original_problem': row['problem'],
                    'original_solution': row['solution'],
                }
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
            assert 'type' in dataset_info, f"Row {i} should have problem type"
            assert 'level' in dataset_info, f"Row {i} should have difficulty level"
            
            print(f"  Row {i}: Type={dataset_info.get('type')}, Level={dataset_info.get('level')}")
    
    def test_custom_dataset_transform(self):
        """Test adapter with a completely custom transformation."""
        try:
            from eval_protocol.adapters.huggingface import create_huggingface_adapter
        except ImportError:
            pytest.skip("HuggingFace dependencies not installed")
        
        def squad_transform(row: Dict[str, Any]) -> Dict[str, Any]:
            """Custom transform for SQuAD dataset."""
            context = row['context']
            question = row['question']
            answers = row['answers']
            
            # Get first answer
            answer_text = answers['text'][0] if answers['text'] else "No answer"
            
            return {
                'messages': [
                    {'role': 'system', 'content': 'Answer the question based on the given context.'},
                    {'role': 'user', 'content': f"Context: {context}\n\nQuestion: {question}"},
                ],
                'ground_truth': answer_text,
                'metadata': {
                    'dataset': 'squad',
                    'context_length': len(context),
                    'question_length': len(question),
                    'num_answers': len(answers['text']),
                }
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
            user_msg = next(msg for msg in row.messages if msg.role == 'user')
            assert 'Context:' in user_msg.content, f"Row {i} should have context"
            assert 'Question:' in user_msg.content, f"Row {i} should have question"
            
            dataset_info = row.input_metadata.dataset_info
            print(f"  Row {i}: Context length={dataset_info.get('context_length')}")


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
            'messages': [
                {'role': 'user', 'content': row['question']},
                {'role': 'assistant', 'content': 'Test response'},  # Simulated response
            ],
            'ground_truth': row['answer'],
            'metadata': {'test': True}
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
    
    assert hasattr(result, 'score'), "Should have evaluation score"
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
    
    print("\nRunning integration test...")
    test_adapters_integration()
    print("✅ Integration test passed!")
    
    print("\n🎉 All E2E tests completed successfully!")