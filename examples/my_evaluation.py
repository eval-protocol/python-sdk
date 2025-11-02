"""
Example evaluation file with non-standard naming.

This file is named 'my_evaluation.py' (not test_*.py),
but can still be discovered using --ep-discover-all flag.

Run with:
    pytest examples/my_evaluation.py --ep-discover-all -v
"""

from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test


# Function also doesn't start with 'test_', but will be auto-registered
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "Custom evaluation"}])  # pyright: ignore[reportArgumentType]
    ]]
)
async def custom_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    This evaluation is in a file called 'my_evaluation.py' 
    and the function is called 'custom_evaluation'.
    
    Neither follows pytest conventions, but both work with:
    - Function: auto-registered as 'test_custom_evaluation'
    - File: discovered with --ep-discover-all flag
    """
    row.evaluation_result = EvaluateResult(
        score=1.0,
        reason="Custom evaluation completed"
    )
    return row


if __name__ == "__main__":
    print("="*70)
    print("Non-standard File and Function Naming Example")
    print("="*70)
    print()
    print("File name: my_evaluation.py (not test_*.py)")
    print("Function name: custom_evaluation (not test_*)")
    print()
    print("To discover and run this test:")
    print("  pytest examples/my_evaluation.py --ep-discover-all -v")
    print()
    print("Or explicitly specify the file:")
    print("  pytest examples/my_evaluation.py -v")
    print("="*70)

