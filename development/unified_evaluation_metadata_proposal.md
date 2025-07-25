### Proposal: Unified Evaluation Metadata Standard

### 1. Overview

This document proposes a unified standard for handling evaluation data within the `eval-protocol` library. The core idea is to leverage a single, enhanced `EvaluateResult` model as the universal data structure for all evaluation scenarios, including per-turn, per-trajectory, and Reinforcement Learning (RL) contexts.

This approach simplifies the existing data pipeline by eliminating the need for a separate `StandardizedMetadata` class, making `EvaluateResult` directly compatible with the `metadata` field of OpenAI's message objects.

### 2. Motivation

The primary goals of this proposal are to:
- **Simplify the Data Model**: Use one canonical class for evaluation results, reducing complexity and removing the need for data model translation.
- **Improve Developer Experience**: Provide a clear and consistent API for developers writing reward functions. The output is always an `EvaluateResult`, regardless of the evaluation's scope.
- **Ensure Compatibility**: Natively handle the constraints of OpenAI's `metadata` field (e.g., string-only values, key/value length limits) within the core data model.
- **Unify Turn and Trajectory Data**: Seamlessly represent both granular per-turn rewards and aggregated trajectory-level results within the same structure.

### 3. The Unified Model: `EvaluateResult`

The `eval_protocol.models.EvaluateResult` pydantic model will be the single source of truth for all evaluation data. It is already used by reward functions and contains fields for an overall score, detailed metrics, and `step_outputs` for RL-style rewards.

We will extend `EvaluateResult` with two key methods to handle serialization to and from the OpenAI-compatible format.

#### 3.1. Proposed `EvaluateResult` Extensions

```python
# In eval_protocol/models.py

class EvaluateResult(BaseModel):
    """The complete result of an evaluator."""
    score: float
    is_score_valid: bool = True
    reason: Optional[str] = None
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)
    step_outputs: Optional[List[StepOutput]] = None
    error: Optional[str] = None

    def to_openai_metadata(self) -> Dict[str, str]:
        """
        Serializes the EvaluateResult into a dictionary compatible with
        OpenAI's message metadata field.

        This involves:
        1. Promoting the 'score' to a top-level key.
        2. Packing the rest of the data into a single JSON string under the 'info' key.
        3. Validating and truncating the 'info' JSON to meet the 512-char limit.
        """
        # 1. Promote score
        metadata = {"eval_protocol_score": str(self.score)}

        # 2. Pack the rest into an 'info' dictionary
        info_dict = {
            "is_score_valid": self.is_score_valid,
            "reason": self.reason,
            "metrics": {k: v.dict() for k, v in self.metrics.items()},
            "step_outputs": [so.dict() for so in self.step_outputs] if self.step_outputs else None,
            "error": self.error,
        }
        # Remove null values for compactness
        info_dict = {k: v for k, v in info_dict.items() if v is not None}

        # 3. Serialize to JSON and validate/truncate
        info_json = json.dumps(info_dict, separators=(",", ":"))
        if len(info_json) > 512:
            # Implement intelligent truncation here, e.g., drop less critical fields
            # For now, we'll just show a placeholder for the logic
            truncated_info = {"error": "info_truncated", "reason": self.reason}
            info_json = json.dumps(truncated_info)
            # Potentially log a warning that data was lost

        metadata["eval_protocol_info"] = info_json
        return metadata

    @classmethod
    def from_openai_metadata(cls, metadata: Dict[str, str]) -> "EvaluateResult":
        """
        Deserializes an EvaluateResult from an OpenAI-compatible metadata dictionary.
        """
        score = float(metadata.get("eval_protocol_score", "0.0"))
        info_json = metadata.get("eval_protocol_info", "{}")

        try:
            info_data = json.loads(info_json)
        except json.JSONDecodeError:
            info_data = {"error": "invalid_json_in_metadata"}

        return cls(score=score, **info_data)

# ... other models like MetricResult, StepOutput remain the same ...
```

### 4. Usage in Practice

#### Per-Turn Evaluation
1. A reward function runs after an agent turn and returns a complete `EvaluateResult` object.
2. This object is serialized: `metadata_dict = turn_result.to_openai_metadata()`.
3. The `metadata_dict` is attached to the corresponding `Message` object's `metadata` field.

#### Per-Trajectory Evaluation (e.g., for RL)
1. At the end of an episode, a final evaluation is performed on the entire trajectory.
2. The reward function returns a single `EvaluateResult`. Its `step_outputs` field contains the list of all per-step rewards and metrics gathered during the episode.
3. This final `EvaluateResult` is serialized: `final_metadata = trajectory_result.to_openai_metadata()`.
4. The `final_metadata` is attached to the *final message* of the trajectory, providing a complete, aggregated record.

### 5. JSON Schema for `eval_protocol_info`

The schema below describes the structure of the JSON string that will be the value of the `eval_protocol_info` key in the metadata.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "EvaluateResultInfo",
  "description": "Schema for the content of the 'eval_protocol_info' field, representing a serialized EvaluateResult object (excluding the top-level score).",
  "type": "object",
  "properties": {
    "is_score_valid": {
      "type": "boolean",
      "description": "Whether the overall score is valid.",
      "default": true
    },
    "reason": {
      "type": ["string", "null"],
      "description": "Optional explanation for the overall score."
    },
    "metrics": {
      "type": "object",
      "description": "Dictionary of component metrics for detailed evaluation.",
      "additionalProperties": { "$ref": "#/definitions/MetricResult" }
    },
    "step_outputs": {
      "type": ["array", "null"],
      "description": "For RL, a list of outputs for each conceptual step.",
      "items": { "$ref": "#/definitions/StepOutput" }
    },
    "error": {
      "type": ["string", "null"],
      "description": "Optional error message if evaluation failed."
    },
    "final_control_plane_info": {
        "type": ["object", "null"],
        "description": "Final state information from the control plane that caused termination or conclusion.",
        "additionalProperties": true
    }
  },
  "definitions": {
    "MetricResult": {
      "type": "object",
      "properties": {
        "is_score_valid": { "type": "boolean", "default": true },
        "score": { "type": "number", "minimum": 0, "maximum": 1 },
        "reason": { "type": "string" }
      },
      "required": ["score", "reason"]
    },
    "StepOutput": {
      "type": "object",
      "properties": {
        "step_index": { "type": ["integer", "string"] },
        "base_reward": { "type": "number" },
        "terminated": { 
            "type": "boolean",
            "description": "Whether the episode terminated at this step."
        },
        "control_plane_info": {
            "type": ["object", "null"],
            "description": "Structured information from the environment's control plane for this step.",
            "additionalProperties": true
        },
        "metrics": { "type": "object", "additionalProperties": true },
        "reason": { "type": ["string", "null"] }
      },
      "required": ["step_index", "base_reward", "terminated"]
    }
  }
}
```

### 6. Impact on the `Trajectory` and Control Plane Integration

This unified approach necessitates a corresponding simplification and update to the `eval_protocol.mcp.types.Trajectory` dataclass and a clear mapping from the control plane logic in `execution/manager.py`.

#### 6.1. Extending Core Data Models

To properly handle control plane data, we must first extend our core data models.

**`StepOutput` Extension**: We will add `terminated` and `control_plane_info` to capture the state at each step.

```python
# In eval_protocol/models.py
class StepOutput(BaseModel):
    step_index: Union[int, str]
    base_reward: float
    terminated: bool = Field(description="Whether the environment signaled termination at this step.")
    control_plane_info: Optional[Dict[str, Any]] = Field(default=None, description="Structured info from the environment's control plane.")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None
```

**`EvaluateResult` Extension**: We will add `final_control_plane_info` to capture the final state that caused the episode to end.

```python
# In eval_protocol/models.py
class EvaluateResult(BaseModel):
    score: float
    # ... other fields
    step_outputs: Optional[List[StepOutput]] = None
    final_control_plane_info: Optional[Dict[str, Any]] = Field(default=None, description="The final control plane state that led to termination.")
    error: Optional[str] = None
    # ... serialization methods
```

#### 6.2. Mapping Control Plane Logic from `manager.py`

With the extended models, we can now clearly map the logic from `_execute_rollout` in `manager.py`.

**Old Logic (`manager.py`)**:
```python
# Create and append control_plane_step dictionary
control_plane_step = {
    "step": step - 1,
    "reward": reward,
    "terminated": rollout_end,
    "info": info.get("control_plane", {}),
    "tool_calls": tool_calls_summary,
}
trajectory.control_plane_steps.append(control_plane_step)

# Create and update control_plane_summary dictionary
if rollout_end:
    trajectory.control_plane_summary.update({
        "total_reward": trajectory.total_reward,
        "control_plane_source": info.get("control_plane", {}),
        # ... other summary fields
    })
```

**New Logic (Conceptual)**:
The `_execute_rollout` function would no longer build `control_plane_steps` or `control_plane_summary`. Instead, it would create a list of `StepOutput` objects.

```python
# Inside the rollout loop in manager.py
step_outputs = []
# ... after envs.step() returns reward, rollout_end, info

# The tool_calls_summary can be added to metrics
step_metrics = {"tool_calls": tool_calls_summary}

# Create a StepOutput for each interaction
new_step_output = StepOutput(
    step_index=step - 1,
    base_reward=reward,
    terminated=rollout_end,
    control_plane_info=info.get("control_plane", {}),
    metrics=step_metrics
)
step_outputs.append(new_step_output)

# ... at the end of the rollout

# The final EvaluateResult is assembled
final_eval_result = EvaluateResult(
    score=total_reward,
    step_outputs=step_outputs,
    # The final info that caused termination is the control_plane_info from the last step
    final_control_plane_info=step_outputs[-1].control_plane_info if step_outputs and step_outputs[-1].terminated else None,
    reason=trajectory.termination_reason,
    # ... other metrics
)

# The new Trajectory object holds this single result
trajectory.evaluation_result = final_eval_result
```

#### 6.3. Proposed `Trajectory` Definition

This leads to a much cleaner `Trajectory` object, where all evaluation-related information is neatly encapsulated within the `EvaluateResult`.

```python
# Proposed changes for: eval_protocol/mcp/types.py
from eval_protocol.models import EvaluateResult, Message

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
This revised structure directly incorporates the control plane's data into our primary evaluation models, making the system more robust and the data flow more explicit. 