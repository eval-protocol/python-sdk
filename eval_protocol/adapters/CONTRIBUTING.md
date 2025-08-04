# Adapter Contributing Guide

This guide explains how to create custom adapters for the Eval Protocol system. Adapters allow you to integrate various data sources and convert them to the `EvaluationRow` format used by the evaluation pipeline.

## Overview

Adapters are responsible for:
1. **Data Ingestion**: Loading data from external sources (APIs, databases, files, etc.)
2. **Format Conversion**: Converting the source data to `EvaluationRow` format
3. **Metadata Extraction**: Preserving relevant metadata from the source system
4. **Error Handling**: Gracefully handling failures and logging issues

## Creating a New Adapter

### 1. Basic Adapter Structure

Create a new Python file in `eval_protocol/adapters/` following this template:

```python
"""Your Custom Adapter for Eval Protocol."""

from typing import Any, Dict, Iterator, List, Optional
import logging

from eval_protocol.models import EvaluationRow, Message, InputMetadata, CompletionParams

logger = logging.getLogger(__name__)

# Optional dependency handling
try:
    import your_external_library
    DEPENDENCY_AVAILABLE = True
except ImportError:
    DEPENDENCY_AVAILABLE = False
    logger.warning("your_external_library not installed. Install with: pip install 'eval-protocol[your_adapter]'")


class YourCustomAdapter:
    """Adapter for integrating with Your Custom Data Source.
    
    This adapter loads data from Your Custom Data Source and converts it
    to EvaluationRow format for use in evaluation pipelines.
    
    Examples:
        Basic usage:
        >>> adapter = YourCustomAdapter(api_key="your_key")
        >>> rows = list(adapter.get_evaluation_rows(limit=10))
    """
    
    def __init__(self, **config):
        """Initialize the adapter with configuration."""
        if not DEPENDENCY_AVAILABLE:
            raise ImportError("your_external_library not installed")
        
        # Initialize your client/connection here
        self.client = your_external_library.Client(**config)
    
    def get_evaluation_rows(self, **kwargs) -> Iterator[EvaluationRow]:
        """Main method to fetch and convert data to EvaluationRow format.
        
        Args:
            **kwargs: Adapter-specific parameters
            
        Yields:
            EvaluationRow: Converted evaluation rows
        """
        # Implement your data fetching logic
        raw_data = self.client.fetch_data(**kwargs)
        
        for item in raw_data:
            try:
                eval_row = self._convert_to_evaluation_row(item)
                if eval_row:
                    yield eval_row
            except Exception as e:
                logger.warning(f"Failed to convert item: {e}")
                continue
    
    def _convert_to_evaluation_row(self, raw_item: Any) -> Optional[EvaluationRow]:
        """Convert a raw data item to EvaluationRow format.
        
        Args:
            raw_item: Raw data item from your source
            
        Returns:
            EvaluationRow or None if conversion fails
        """
        # Extract messages from your data format
        messages = self._extract_messages(raw_item)
        
        # Extract metadata
        input_metadata = self._create_input_metadata(raw_item)
        
        # Extract ground truth if available
        ground_truth = self._extract_ground_truth(raw_item)
        
        # Extract tools if available (for tool calling scenarios)
        tools = self._extract_tools(raw_item)
        
        return EvaluationRow(
            messages=messages,
            tools=tools,
            input_metadata=input_metadata,
            ground_truth=ground_truth,
        )
    
    def _extract_messages(self, raw_item: Any) -> List[Message]:
        """Extract conversation messages from raw data."""
        # Implement message extraction logic
        # Convert your data format to List[Message]
        pass
    
    def _create_input_metadata(self, raw_item: Any) -> InputMetadata:
        """Create InputMetadata from raw data."""
        # Implement metadata extraction
        pass
    
    def _extract_ground_truth(self, raw_item: Any) -> Optional[str]:
        """Extract ground truth if available."""
        # Implement ground truth extraction
        pass
    
    def _extract_tools(self, raw_item: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool definitions if available."""
        # Implement tool extraction for tool calling scenarios
        pass


# Factory function (recommended)
def create_your_custom_adapter(**config) -> YourCustomAdapter:
    """Factory function to create your custom adapter."""
    return YourCustomAdapter(**config)
```

### 2. Key Components

#### Messages
Convert your data to a list of `Message` objects representing the conversation:

```python
from eval_protocol.models import Message

# Basic message
message = Message(role="user", content="Hello, world!")

# Message with tool calls
message = Message(
    role="assistant",
    content="I'll help you with that calculation.",
    tool_calls=[{
        "id": "call_123",
        "type": "function", 
        "function": {
            "name": "calculate",
            "arguments": '{"x": 5, "y": 3}'
        }
    }]
)

# Tool response message
message = Message(
    role="tool",
    content="Result: 8",
    tool_call_id="call_123"
)
```

#### Input Metadata
Preserve important metadata from your source system:

```python
from eval_protocol.models import InputMetadata, CompletionParams

input_metadata = InputMetadata(
    row_id="unique_row_identifier",
    completion_params=CompletionParams(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    ),
    dataset_info={
        "source_system": "your_system_name",
        "original_id": "source_item_id",
        "custom_field": "custom_value",
    },
    session_data={
        "user_id": "user123",
        "session_id": "session456", 
        "timestamp": "2024-01-01T00:00:00Z",
    }
)
```

#### Ground Truth
Include expected answers if available:

```python
# Simple string ground truth
ground_truth = "The correct answer is 42"

# For math problems, include the final answer
ground_truth = "#### 42"  # GSM8K format
```

#### Tools (for Tool Calling)
Include tool definitions for tool calling scenarios:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

### 3. Optional Dependencies

Add your adapter's dependencies to `pyproject.toml`:

```toml
[project.optional-dependencies]
your_adapter = [
    "your-external-library>=1.0.0",
    "other-dependency>=2.0.0",
]
```

Users can then install with:
```bash
pip install 'eval-protocol[your_adapter]'
```

### 4. Error Handling

Implement robust error handling:

```python
import logging

logger = logging.getLogger(__name__)

def get_evaluation_rows(self, **kwargs) -> Iterator[EvaluationRow]:
    try:
        data = self.client.fetch_data(**kwargs)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return
    
    for item in data:
        try:
            row = self._convert_to_evaluation_row(item)
            if row:
                yield row
        except Exception as e:
            logger.warning(f"Failed to convert item {item.get('id', 'unknown')}: {e}")
            continue
```

### 5. Update Package Exports

Add your adapter to `eval_protocol/adapters/__init__.py`:

```python
# Add conditional import
try:
    from .your_adapter import YourCustomAdapter, create_your_custom_adapter
    __all__.extend(["YourCustomAdapter", "create_your_custom_adapter"])
except ImportError:
    pass
```

## Testing Your Adapter

Create comprehensive tests for your adapter:

```python
# tests/test_adapters/test_your_adapter.py
import pytest
from unittest.mock import Mock, patch

from eval_protocol.adapters.your_adapter import YourCustomAdapter
from eval_protocol.models import EvaluationRow


class TestYourCustomAdapter:
    """Test suite for YourCustomAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = YourCustomAdapter(api_key="test_key")
        assert adapter.client is not None
    
    def test_get_evaluation_rows(self):
        """Test conversion to EvaluationRow format."""
        adapter = YourCustomAdapter(api_key="test_key")
        
        # Mock the external API response
        with patch.object(adapter.client, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = [
                # Mock data in your format
                {"id": "1", "question": "Test?", "answer": "Yes"}
            ]
            
            rows = list(adapter.get_evaluation_rows(limit=1))
            
            assert len(rows) == 1
            assert isinstance(rows[0], EvaluationRow)
            assert len(rows[0].messages) > 0
    
    def test_error_handling(self):
        """Test error handling."""
        adapter = YourCustomAdapter(api_key="test_key")
        
        with patch.object(adapter.client, 'fetch_data') as mock_fetch:
            mock_fetch.side_effect = Exception("API Error")
            
            rows = list(adapter.get_evaluation_rows())
            assert len(rows) == 0  # Should handle error gracefully
```

## Examples by Data Source Type

### Chat/Conversation Data

For simple chat data:

```python
def _extract_messages(self, conversation: Dict) -> List[Message]:
    messages = []
    
    # Add system prompt if available
    if conversation.get('system_prompt'):
        messages.append(Message(role="system", content=conversation['system_prompt']))
    
    # Add conversation turns
    for turn in conversation['turns']:
        messages.append(Message(
            role=turn['role'],
            content=turn['content']
        ))
    
    return messages
```

### Tool Calling Data

For tool calling scenarios:

```python
def _extract_messages(self, trace: Dict) -> List[Message]:
    messages = []
    
    for step in trace['steps']:
        if step['type'] == 'user_message':
            messages.append(Message(role="user", content=step['content']))
        
        elif step['type'] == 'assistant_message':
            message = Message(role="assistant", content=step.get('content'))
            
            # Add tool calls if present
            if step.get('tool_calls'):
                message.tool_calls = step['tool_calls']
            
            messages.append(message)
        
        elif step['type'] == 'tool_response':
            messages.append(Message(
                role="tool",
                content=step['content'],
                tool_call_id=step['tool_call_id']
            ))
    
    return messages
```

### Dataset Files

For dataset files with transformation functions:

```python
from eval_protocol.adapters.huggingface import create_huggingface_adapter

def my_dataset_transform(row):
    """Transform dataset row to evaluation format."""
    return {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': row['input_text']},
        ],
        'ground_truth': row['expected_output'],
        'metadata': {
            'dataset': 'my_custom_dataset',
            'category': row.get('category'),
            'difficulty': row.get('difficulty_level'),
        }
    }

# For HuggingFace datasets
adapter = create_huggingface_adapter(
    dataset_id="my-org/my-dataset",
    transform_fn=my_dataset_transform,
    config_name="default",  # optional
    revision="main",        # optional
)

# For local files
adapter = HuggingFaceAdapter.from_local(
    path="./my_dataset.jsonl",
    transform_fn=my_dataset_transform,
)

# Use the adapter
rows = list(adapter.get_evaluation_rows(split="test", limit=100))
```

## Best Practices

### 1. Graceful Degradation
- Handle missing optional fields gracefully
- Provide sensible defaults
- Don't fail the entire batch for one bad item

### 2. Logging
- Use structured logging with appropriate levels
- Include context like item IDs in error messages
- Log successful conversions at DEBUG level

### 3. Performance
- Use iterators/generators for large datasets
- Implement pagination for API sources
- Consider caching for expensive operations

### 4. Documentation
- Include comprehensive docstrings
- Provide usage examples
- Document expected data formats

### 5. Type Safety
- Use type hints throughout
- Validate input parameters
- Handle type conversion errors

## Integration Checklist

Before submitting your adapter:

- [ ] Handles optional dependencies correctly
- [ ] Includes comprehensive error handling
- [ ] Has factory function for easy instantiation
- [ ] Added to `pyproject.toml` optional dependencies
- [ ] Added to `__init__.py` exports
- [ ] Includes unit tests
- [ ] Has proper documentation and examples
- [ ] Follows project coding standards (black, isort, mypy)
- [ ] Tested with real data from your source system

## Getting Help

- Check existing adapters for patterns and examples
- Review the main [CONTRIBUTING.md](../../../development/CONTRIBUTING.md) for project conventions
- Open an issue for questions or feature requests
- Submit a draft PR for early feedback

## Using the Generic HuggingFace Adapter

For datasets hosted on HuggingFace Hub, you can often use the generic `HuggingFaceAdapter` instead of creating a completely custom adapter:

```python
from eval_protocol.adapters.huggingface import create_huggingface_adapter

def my_transform_function(row):
    """Transform function for your specific dataset."""
    return {
        'messages': [
            {'role': 'system', 'content': 'Your system prompt here'},
            {'role': 'user', 'content': row['question']},  # Adjust field names
        ],
        'ground_truth': row['answer'],  # Adjust field name
        'metadata': {
            'dataset_specific_field': row.get('category'),
            'custom_metadata': 'value',
        },
        'tools': row.get('tools'),  # If your dataset has tool definitions
    }

adapter = create_huggingface_adapter(
    dataset_id="your-org/your-dataset",
    transform_fn=my_transform_function,
    config_name="default",    # optional
    revision="main",          # optional commit/branch
)

rows = list(adapter.get_evaluation_rows(split="test", limit=100))
```

This approach is often simpler than creating a full custom adapter, especially for straightforward dataset transformations.

## Adapter Ideas

Here are some potential adapters that would be valuable:

- **OpenAI Evals**: Load data from OpenAI's evals repository
- **LLM Evaluation Datasets**: MMLU, HellaSwag, etc.
- **Chat Platforms**: Discord, Slack conversation exports  
- **Monitoring Tools**: Other observability platforms
- **Custom APIs**: Company-specific data sources
- **File Formats**: Parquet, Excel, database exports
- **Research Datasets**: Academic benchmarks and competitions

We welcome contributions for any of these or other creative integrations!