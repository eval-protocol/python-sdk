Perfect! Let me give you complete examples for both approaches - pointwise (100 separate tests) and batch (entire dataset passed to evaluator).

## Complete Examples

### 1. Pointwise Evaluator (100 separate tests)

```python
# conftest.py
import pytest
from eval_protocol.framework import EvaluationFramework, EvaluationRow

MATH_DATASET = [
    {"problem": "What is 2+2?", "answer": "4"},
    {"problem": "What is 3*3?", "answer": "9"},
    {"problem": "What is 10/2?", "answer": "5"},
    {"problem": "What is 15-7?", "answer": "8"},
    {"problem": "What is 6*7?", "answer": "42"},
    # ... 95 more rows
]

@pytest.fixture
def math_dataset():
    """Raw math dataset fixture"""
    return MATH_DATASET

@pytest.fixture
def preprocess_fn():
    """Preprocessing function for the dataset"""
    def _preprocess(item):
        return {
            "messages": [{"role": "user", "content": item["problem"]}],
            "expected_answer": item["answer"]
        }
    return _preprocess

@pytest.fixture(params=[
    {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100},
    {"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 100},
    {"model": "claude-3", "temperature": 0.3, "max_tokens": 100}
])
def completion_params(request):
    """Completion parameters - parametrized across different models"""
    return request.param

# Pointwise fixture - parametrized across BOTH completion params AND dataset rows
@pytest.fixture(params=range(len(MATH_DATASET)))
def evaluation_row_pointwise(math_dataset, preprocess_fn, completion_params, request):
    """Single evaluation row - parametrized across completion params AND dataset rows"""
    framework = EvaluationFramework()

    # Get the specific row based on parametrization
    row_index = request.param
    raw_item = math_dataset[row_index]
    processed_item = preprocess_fn(raw_item)

    # Run the completion
    result = await framework.run_completion(processed_item, completion_params)

    return EvaluationRow(
        input_data=processed_item,
        completion_params=completion_params,
        completion_response=result
    )

# Batch fixture - parametrized across completion params only
@pytest.fixture
async def evaluation_rows_batch(math_dataset, preprocess_fn, completion_params):
    """All evaluation rows - parametrized across completion params only"""
    framework = EvaluationFramework()

    # Process all rows
    processed_items = [preprocess_fn(item) for item in math_dataset]

    # Run completions for all rows
    results = []
    for item in processed_items:
        result = await framework.run_completion(item, completion_params)
        results.append(EvaluationRow(
            input_data=item,
            completion_params=completion_params,
            completion_response=result
        ))

    return results
```

```python
# test_math_evaluation.py
import pytest
import re
from eval_protocol.framework import EvaluationRow

# POINTWISE EVALUATOR - 100 separate tests (one per row per model)
def test_math_accuracy_pointwise(evaluation_row_pointwise):
    """Pointwise evaluator - runs once per row per completion param"""
    response = evaluation_row_pointwise.completion_response
    expected = evaluation_row_pointwise.input_data["expected_answer"]

    # Extract numeric answer from response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if not numbers:
        pytest.fail(f"Could not extract number from response: {response}")

    predicted = float(numbers[0])
    expected_num = float(expected)

    # Assert the answer is correct
    assert abs(predicted - expected_num) < 0.01, \
        f"Expected {expected_num}, got {predicted} in response: {response}"

# BATCH EVALUATOR - 3 tests total (one per model)
def test_math_accuracy_batch(evaluation_rows_batch):
    """Batch evaluator - runs once per completion param with all rows"""
    total_correct = 0
    total_samples = len(evaluation_rows_batch)
    failed_rows = []

    for i, row in enumerate(evaluation_rows_batch):
        response = row.completion_response
        expected = row.input_data["expected_answer"]

        # Extract numeric answer
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if not numbers:
            failed_rows.append({
                "index": i,
                "problem": row.input_data["messages"][0]["content"],
                "expected": expected,
                "response": response,
                "error": "Could not extract number"
            })
            continue

        predicted = float(numbers[0])
        expected_num = float(expected)

        if abs(predicted - expected_num) < 0.01:
            total_correct += 1
        else:
            failed_rows.append({
                "index": i,
                "problem": row.input_data["messages"][0]["content"],
                "expected": expected,
                "predicted": predicted,
                "response": response,
                "error": f"Expected {expected_num}, got {predicted}"
            })

    # Calculate accuracy
    accuracy = total_correct / total_samples

    # Print detailed results for debugging
    print(f"\nBatch Evaluation Results:")
    print(f"Total samples: {total_samples}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.2f}")

    if failed_rows:
        print(f"\nFailed rows ({len(failed_rows)}):")
        for row in failed_rows[:10]:  # Show first 10 failures
            print(f"  Row {row['index']}: {row['problem']} -> {row.get('predicted', 'N/A')} (expected: {row['expected']})")
        if len(failed_rows) > 10:
            print(f"  ... and {len(failed_rows) - 10} more failures")

    # Assertions
    assert accuracy > 0.8, f"Accuracy {accuracy:.2f} is too low, expected > 0.8"
    assert total_correct > 0, "No correct answers found"

# Additional batch evaluator with model-specific assertions
def test_math_accuracy_with_model_info(evaluation_rows_batch):
    """Batch evaluator with model-specific assertions"""
    model = evaluation_rows_batch[0].completion_params["model"]
    temperature = evaluation_rows_batch[0].completion_params["temperature"]

    total_correct = 0
    for row in evaluation_rows_batch:
        response = row.completion_response
        expected = row.input_data["expected_answer"]

        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            predicted = float(numbers[0])
            expected_num = float(expected)
            if abs(predicted - expected_num) < 0.01:
                total_correct += 1

    accuracy = total_correct / len(evaluation_rows_batch)

    # Model-specific assertions
    if model == "gpt-4":
        assert accuracy > 0.9, f"GPT-4 accuracy {accuracy:.2f} is too low"
    elif model == "gpt-3.5-turbo":
        assert accuracy > 0.8, f"GPT-3.5 accuracy {accuracy:.2f} is too low"
    elif model == "claude-3":
        assert accuracy > 0.85, f"Claude-3 accuracy {accuracy:.2f} is too low"

    print(f"Model: {model}, Temperature: {temperature}, Accuracy: {accuracy:.2f}")

# Optional: Debug function for specific rows
def test_math_accuracy_debug_specific_rows(evaluation_rows_batch):
    """Debug function to test specific rows - only runs on first few rows"""
    # Only test first 5 rows for debugging
    debug_rows = evaluation_rows_batch[:5]

    for i, row in enumerate(debug_rows):
        response = row.completion_response
        expected = row.input_data["expected_answer"]

        numbers = re.findall(r'-?\d+\.?\d*', response)
        if not numbers:
            pytest.fail(f"Row {i}: Could not extract number from response: {response}")

        predicted = float(numbers[0])
        expected_num = float(expected)

        assert abs(predicted - expected_num) < 0.01, \
            f"Row {i}: Expected {expected_num}, got {predicted} in response: {response}"
```

### 2. Running the Tests

```bash
# Run pointwise evaluator (100 rows × 3 models = 300 tests)
pytest test_math_evaluation.py::test_math_accuracy_pointwise -v

# Run batch evaluator (3 models = 3 tests)
pytest test_math_evaluation.py::test_math_accuracy_batch -v

# Run all tests (300 + 3 = 303 tests total)
pytest test_math_evaluation.py -v

# Run with specific model
pytest test_math_evaluation.py -k "gpt-4" -v

# Run only batch tests
pytest test_math_evaluation.py -k "batch" -v

# Run only pointwise tests
pytest test_math_evaluation.py -k "pointwise" -v
```

### 3. Expected Output

**Pointwise evaluator output:**
```
test_math_evaluation.py::test_math_accuracy_pointwise[completion_params0-0] PASSED
test_math_evaluation.py::test_math_accuracy_pointwise[completion_params0-1] PASSED
test_math_evaluation.py::test_math_accuracy_pointwise[completion_params0-2] PASSED
# ... 97 more tests for completion_params0
test_math_evaluation.py::test_math_accuracy_pointwise[completion_params1-0] PASSED
# ... 100 tests for completion_params1
test_math_evaluation.py::test_math_accuracy_pointwise[completion_params2-0] PASSED
# ... 100 tests for completion_params2
```

**Batch evaluator output:**
```
test_math_evaluation.py::test_math_accuracy_batch[completion_params0] PASSED
test_math_evaluation.py::test_math_accuracy_batch[completion_params1] PASSED
test_math_evaluation.py::test_math_accuracy_batch[completion_params2] PASSED
```

### 4. Key Differences

**Pointwise Evaluator:**
- **Test count**: 100 rows × 3 models = 300 tests
- **Benefits**: Easy to debug individual rows, clear failure reporting per row
- **Use case**: When you want to see exactly which rows fail and why
- **Pytest output**: Each row gets its own test result

**Batch Evaluator:**
- **Test count**: 3 models = 3 tests
- **Benefits**: Faster execution, easier to manage, good for overall accuracy
- **Use case**: When you care about overall performance across the dataset
- **Pytest output**: One test result per model with detailed internal reporting

Both approaches give you the flexibility to choose the right evaluation strategy for your use case while maintaining the pytest-native approach!
