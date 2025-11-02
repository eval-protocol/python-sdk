"""
Simple test to verify that @evaluation_test decorated functions
are automatically discoverable by pytest.
"""

import pytest
from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test


# Example 1: Function without 'test_' prefix - will be auto-registered
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "Test message"}])
    ]]
)
async def my_custom_eval(row: EvaluationRow) -> EvaluationRow:
    """
    This function doesn't start with 'test_', but @evaluation_test
    will automatically register it as 'test_my_custom_eval'.
    """
    row.evaluation_result = EvaluateResult(score=1.0)
    return row


# Example 2: Function with proper 'test_' prefix 
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "Another test"}])
    ]]
)
async def test_proper_eval(row: EvaluationRow) -> EvaluationRow:
    """This already follows pytest conventions."""
    row.evaluation_result = EvaluateResult(score=1.0)
    return row


if __name__ == "__main__":
    # Run collection to show both tests are discovered
    pytest.main([__file__, "--collect-only", "-v"])

