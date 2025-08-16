# Adapter Examples

This directory contains examples demonstrating how to use the various data source adapters available in the Eval Protocol system.

## Available Adapters

### 1. Langfuse Adapter (`langfuse_example.py`)

Connects to Langfuse deployments to pull conversation traces and convert them to EvaluationRow format.

**Features:**
- Pull chat conversations and tool calling traces
- Filter by time ranges, tags, users, and sessions
- Preserve metadata from Langfuse traces
- Support for both cloud and self-hosted Langfuse instances

**Prerequisites:**
```bash
pip install 'eval-protocol[langfuse]'
```

**Environment Variables:**
```bash
export LANGFUSE_PUBLIC_KEY="your_public_key"
export LANGFUSE_SECRET_KEY="your_secret_key"
export LANGFUSE_HOST="https://your-langfuse-instance.com"  # optional
export LANGFUSE_PROJECT_ID="your_project_id"  # optional
```

### 2. HuggingFace Adapter (`huggingface_example.py`)

Loads datasets from HuggingFace Hub and converts them to EvaluationRow format.

**Features:**
- Generic adapter with arbitrary transformation functions
- Built-in convenience functions for popular datasets (GSM8K, MATH, etc.)
- Support for local dataset files (JSON, JSONL, CSV)
- Full control over data transformation and formatting
- Support for dataset revisions/commits

**Prerequisites:**
```bash
pip install 'eval-protocol[huggingface]'
```

## Running the Examples

### Basic Usage

```bash
# Run Langfuse example
python examples/adapters/langfuse_example.py

# Run HuggingFace example
python examples/adapters/huggingface_example.py

# Run GSM8K replacement example
python examples/adapters/gsm8k_replacement_example.py
```

### With Environment Setup

```bash
# Set up Langfuse credentials
export LANGFUSE_PUBLIC_KEY="pk_..."
export LANGFUSE_SECRET_KEY="sk_..."
python examples/adapters/langfuse_example.py

# HuggingFace works without credentials for public datasets
python examples/adapters/huggingface_example.py
```

## Integration Patterns

### 1. Replace Static Dataset Files

Instead of using static JSONL files, use adapters to pull fresh data:

```python
# Old approach
input_dataset=["development/gsm8k_sample.jsonl"]

# New approach with HuggingFace adapter
from eval_protocol.adapters.huggingface import create_gsm8k_adapter

adapter = create_gsm8k_adapter()
evaluation_rows = list(adapter.get_evaluation_rows(split="test", limit=100))

# Or for complete control:
def custom_gsm8k_transform(row):
    return {
        'messages': [
            {'role': 'system', 'content': 'Your custom prompt'},
            {'role': 'user', 'content': row['question']}
        ],
        'ground_truth': row['answer'],
        'metadata': {'custom_field': 'value'}
    }

from eval_protocol.adapters.huggingface import create_huggingface_adapter
custom_adapter = create_huggingface_adapter(
    dataset_id="gsm8k",
    config_name="main",
    transform_fn=custom_gsm8k_transform
)
```

### 2. Real-time Data from Production Systems

Pull recent conversations from your production systems:

```python
from eval_protocol.adapters.langfuse import create_langfuse_adapter
from datetime import datetime, timedelta

adapter = create_langfuse_adapter(...)
recent_rows = list(adapter.get_evaluation_rows(
    from_timestamp=datetime.now() - timedelta(hours=24),
    tags=["production"],
))
```

### 3. Batch Processing Large Datasets

Process datasets in manageable batches:

```python
batch_size = 100
for offset in range(0, 1000, batch_size):
    batch = list(adapter.get_evaluation_rows(
        limit=batch_size,
        offset=offset,
    ))
    # Process batch...
```

## Common Use Cases

### Evaluation Pipeline Integration

```python
from eval_protocol.adapters.huggingface import create_gsm8k_adapter
from eval_protocol.rewards.math import math_reward

# Load dataset
adapter = create_gsm8k_adapter()
rows = list(adapter.get_evaluation_rows(limit=10))

# Run evaluation
for row in rows:
    # Add model response (you would generate this)
    row.messages.append(Message(role="assistant", content="..."))

    # Evaluate
    result = math_reward(messages=row.messages, ground_truth=row.ground_truth)
    print(f"Score: {result.score}")
```

### Training Data Preparation

```python
# Convert to training format
training_data = []
for row in adapter.get_evaluation_rows():
    training_data.append({
        "messages": [{"role": msg.role, "content": msg.content} for msg in row.messages],
        "ground_truth": row.ground_truth,
    })
```

### A/B Testing with Live Data

```python
# Compare models on recent production data
langfuse_adapter = create_langfuse_adapter(...)
recent_conversations = list(langfuse_adapter.get_evaluation_rows(
    from_timestamp=datetime.now() - timedelta(days=1),
    limit=100,
))

# Test both models on the same data
for row in recent_conversations:
    # Test model A and B, compare results...
```

## Custom Adapter Development

### Option 1: Use Generic HuggingFace Adapter (Recommended)

For datasets on HuggingFace Hub, use the generic adapter with a transform function:

```python
from eval_protocol.adapters.huggingface import create_huggingface_adapter

def my_transform(row):
    return {
        'messages': [
            {'role': 'system', 'content': 'Your system prompt'},
            {'role': 'user', 'content': row['input_field']},
        ],
        'ground_truth': row['output_field'],
        'metadata': {'custom': row.get('metadata_field')}
    }

adapter = create_huggingface_adapter(
    dataset_id="your-dataset-id",
    transform_fn=my_transform,
    revision="main"  # optional
)
```

### Option 2: Create Custom Adapter Class

See `eval_protocol/adapters/CONTRIBUTING.md` for detailed instructions on creating full custom adapters.

Quick template:

```python
from eval_protocol.models import EvaluationRow, Message, InputMetadata

class MyCustomAdapter:
    def __init__(self, **config):
        # Initialize your data source connection
        pass

    def get_evaluation_rows(self, **kwargs) -> Iterator[EvaluationRow]:
        # Fetch data and convert to EvaluationRow format
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you have the right optional dependencies installed
   ```bash
   pip install 'eval-protocol[langfuse]'  # for Langfuse
   pip install 'eval-protocol[huggingface]'  # for HuggingFace
   pip install 'eval-protocol[adapters]'  # for all adapters
   ```

2. **Authentication Errors**: Check your environment variables and API keys

3. **Network Errors**: Verify connectivity to external services

4. **Data Format Issues**: Check that your data source has the expected fields

### Debug Mode

Enable debug logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your adapter code here...
```

### Getting Help

- Check the main [CONTRIBUTING.md](../../development/CONTRIBUTING.md) for project setup
- Review adapter-specific documentation in `eval_protocol/adapters/CONTRIBUTING.md`
- Open an issue on GitHub for bugs or feature requests
- Join the community discussions for questions

## Contributing

We welcome contributions of new adapters! Popular integrations that would be valuable:

- **Database adapters**: PostgreSQL, MongoDB, etc.
- **API adapters**: OpenAI Evals, Anthropic datasets, etc.
- **File format adapters**: Parquet, Excel, etc.
- **Monitoring platform adapters**: DataDog, New Relic, etc.

See the adapter contributing guide for detailed instructions.
