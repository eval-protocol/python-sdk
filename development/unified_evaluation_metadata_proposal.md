### Proposal: Unified Evaluation Metadata Standard

### 1. Overview

This document proposes a unified standard for handling evaluation data within the `eval-protocol` library. The core idea is to leverage two enhanced data models as the universal data structures for all evaluation scenarios:

1. **`EvaluateResult`** - The unified evaluation result model for both per-turn and per-trajectory evaluations
2. **`EvaluationRow`** - The canonical data structure that encapsulates messages, tools, and evaluation results

This approach simplifies the existing data pipeline by providing clear, consistent data models that can handle both traditional row-wise evaluation and trajectory-based RL evaluation scenarios.

### 2. Motivation

The primary goals of this proposal are to:
- **Simplify the Data Model**: Use canonical classes for evaluation data, reducing complexity and eliminating the need for data model translation.
- **Improve Developer Experience**: Provide a clear and consistent API for developers writing reward functions and handling evaluation data.
- **Unify Turn and Trajectory Data**: Seamlessly represent both granular per-turn rewards and aggregated trajectory-level results within the same structure.
- **Support Both Evaluation Paradigms**: Handle both traditional batch evaluation (row-wise) and interactive trajectory evaluation (RL) scenarios.

### 3. The Unified Models

#### 3.1. Enhanced `EvaluateResult`

The `eval_protocol.models.EvaluateResult` model serves as the universal evaluation result structure. It has been extended to support both per-turn and per-trajectory evaluation scenarios:

**Key Enhancements:**
- `trajectory_info`: Additional trajectory-level information (duration, steps, termination_reason, etc.)
- `final_control_plane_info`: The final control plane state that led to termination
- Enhanced `StepOutput` with `terminated` and `control_plane_info` fields

#### 3.2. New `EvaluationRow` Model

The new `EvaluationRow` model serves as the canonical data structure for evaluation units with a clean, organized structure:

```python
class EvaluationRow(BaseModel):
    """
    Unified data structure for a single evaluation unit that contains messages, 
    tools, and evaluation results. This can represent either a single turn evaluation
    or a complete trajectory evaluation.
    """
    
    # Core conversation data
    messages: List[Message]
    
    # Tool and function call information  
    tools: Optional[List[Dict[str, Any]]] = None
    
    # Input-related metadata (grouped together for cleaner organization)
    input_metadata: Optional[Dict[str, Any]] = None
    
    # Unified evaluation result
    evaluation_result: Optional[EvaluateResult] = None
```

**Key Features:**
- **Clean Organization**: Only four main fields for maximum clarity
- `is_trajectory_evaluation()`: Determines if this represents a trajectory vs. single-turn evaluation
- `get_conversation_length()`, `get_assistant_messages()`, `get_user_messages()`: Helper methods for data analysis
- `get_input_metadata(key, default)`: Helper method to access input metadata
- Full serialization/deserialization support via Pydantic

**Input Metadata Structure:**
The `input_metadata` field can contain any relevant input information:
```python
input_metadata = {
    "row_id": "unique_identifier",
    "dataset_info": {"source": "math_problems", "difficulty": "easy"},
    "model_config": {"model": "gpt-4", "temperature": 0.1},
    "session_data": {"seed": 42, "timestamp": "2024-01-01T00:00:00Z"},
    "custom_fields": {...}  # Any additional custom metadata
}
```

### 4. Usage in Practice

#### 4.1. Per-Turn Evaluation (Row-wise)
```python
# Single turn conversation
messages = [
    Message(role="user", content="What is 2+2?"),
    Message(role="assistant", content="2+2 equals 4.")
]

# Evaluation result for this turn
evaluation_result = EvaluateResult(
    score=1.0,
    reason="Correct answer",
    metrics={"accuracy": MetricResult(score=1.0, reason="Perfect")}
)

# Create evaluation row
row = EvaluationRow(
    messages=messages,
    evaluation_result=evaluation_result,
    input_metadata={
        "row_id": "math_001",
        "dataset_info": {"source": "math_eval"},
        "model_config": {"model": "gpt-4", "temperature": 0.0}
    }
)

# Check type: row.is_trajectory_evaluation() -> False
# Access metadata: row.get_input_metadata("row_id") -> "math_001"
```

#### 4.2. Per-Trajectory Evaluation (RL)
```python
# Multi-turn trajectory
messages = [
    Message(role="user", content="Start task"),
    Message(role="assistant", content="Starting step 1"),
    Message(role="user", content="Continue"),
    Message(role="assistant", content="Completing step 2")
]

# Step-by-step evaluation results
step_outputs = [
    StepOutput(
        step_index=0, 
        base_reward=0.3, 
        terminated=False
    ),
    StepOutput(
        step_index=1, 
        base_reward=0.7, 
        terminated=True,
        control_plane_info={"task_completed": True}
    )
]

# Final trajectory evaluation
evaluation_result = EvaluateResult(
    score=0.5,
    reason="Task completed in 2 steps",
    step_outputs=step_outputs,
    trajectory_info={
        "duration": 45.2,
        "steps": 2,
        "termination_reason": "task_completed"
    },
    final_control_plane_info={"task_completed": True}
)

# Create evaluation row
row = EvaluationRow(
    messages=messages,
    evaluation_result=evaluation_result,
    input_metadata={
        "row_id": "trajectory_001",
        "session_data": {"seed": 123, "environment": "gridworld"},
        "execution_info": {"max_steps": 10, "timeout": 60}
    }
)

# Check type: row.is_trajectory_evaluation() -> True
```

#### 4.3. With Tool Usage
```python
messages = [
    Message(role="user", content="Search for recent papers on ML"),
    Message(
        role="assistant", 
        content="I'll search for recent ML papers.",
        tool_calls=[{
            "id": "search_1", 
            "type": "function", 
            "function": {"name": "search_papers", "arguments": '{"query": "machine learning", "recent": true}'}
        }]
    )
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search academic papers",
            "parameters": {"type": "object", "properties": {...}}
        }
    }
]

row = EvaluationRow(
    messages=messages,
    tools=tools,
    evaluation_result=EvaluateResult(score=0.9, reason="Good tool usage"),
    input_metadata={
        "row_id": "search_001",
        "dataset_info": {"category": "tool_usage", "complexity": "medium"}
    }
)
```

### 5. Enhanced Data Models

#### 5.1. Extended `StepOutput`

```python
class StepOutput(BaseModel):
    step_index: Union[int, str]
    base_reward: float
    terminated: bool = False  # NEW: Environment termination signal
    control_plane_info: Optional[Dict[str, Any]] = None  # NEW: Control plane data
    metrics: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None
```

#### 5.2. Extended `EvaluateResult`

```python
class EvaluateResult(BaseModel):
    score: float
    is_score_valid: bool = True
    reason: Optional[str] = None
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)
    step_outputs: Optional[List[StepOutput]] = None
    error: Optional[str] = None
    
    # NEW: Unified trajectory and row-wise support
    trajectory_info: Optional[Dict[str, Any]] = None
    final_control_plane_info: Optional[Dict[str, Any]] = None
```

### 6. JSON Schema for `EvaluationRow`

A comprehensive JSON schema will be developed for the `EvaluationRow` structure to ensure:
- Proper validation of evaluation data
- Clear documentation of the data format
- Support for tooling and integrations
- Standardized serialization across different components

The schema will cover:
- Message structure validation
- Tools and function call formatting
- Evaluation result structure (both single-turn and trajectory)
- Input metadata field specifications
- Proper handling of optional fields

### 7. Impact on Existing Components

#### 7.1. Trajectory Integration

The unified approach necessitates updates to the `eval_protocol.mcp.types.Trajectory` dataclass:

```python
@dataclass
class Trajectory:
    """Represents a complete rollout trajectory with unified evaluation results."""
    session: MCPSession
    terminated: bool
    steps: int
    duration: float
    termination_reason: str

    # Unified evaluation and conversation history
    conversation_history: List[Message]
    evaluation_result: Optional[EvaluateResult] = None

    # Environment-specific data (can be deprecated if fully captured in messages/evaluation)
    observations: List[Any] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
```

#### 7.2. Control Plane Integration

The control plane logic in `execution/manager.py` will be simplified to directly populate the unified data structures:

- `control_plane_steps` and `control_plane_summary` data will be integrated into `StepOutput` objects
- Final trajectory state will be captured in `EvaluateResult.final_control_plane_info`
- Step-level control plane data will be stored in `StepOutput.control_plane_info`

### 8. Backward Compatibility

- `EvaluateResult` maintains full backward compatibility for existing reward functions
- Existing reward functions continue to work without modification
- New fields are optional and default to appropriate values
- The enhanced models provide additional capabilities without breaking existing code

### 9. Benefits

1. **Clean and Simple Structure**: Only four main fields in `EvaluationRow` for maximum clarity
2. **Flexible Input Metadata**: Single `input_metadata` field can contain any relevant context
3. **Unified Data Flow**: Single data structure handles both evaluation paradigms
4. **Developer Friendly**: Clear helper methods and intuitive data access patterns
5. **Extensible**: Easy to add new capabilities without structural changes
6. **Type Safe**: Full Pydantic validation and type hints throughout
7. **Serialization Ready**: Built-in JSON serialization/deserialization support

This unified approach provides a robust yet simple foundation for evaluation data handling while maintaining clarity and backward compatibility. 