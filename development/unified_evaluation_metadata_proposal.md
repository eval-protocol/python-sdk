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
        "metrics": { "type": "object", "additionalProperties": true },
        "reason": { "type": ["string", "null"] }
      },
      "required": ["step_index", "base_reward"]
    }
  }
}
```

### 6. Impact on the `Trajectory` Class

This unified approach necessitates a corresponding simplification and update to the `eval_protocol.mcp.types.Trajectory` dataclass. The goal is to make `EvaluateResult` the single container for all evaluation-related data, removing redundant or overlapping fields from `Trajectory`.

#### 6.1. Current `Trajectory` Definition

```python
# From: eval_protocol/mcp/types.py

@dataclass
class Trajectory:
    """Represents a complete rollout trajectory."""
    session: MCPSession
    observations: List[Any]
    actions: List[str]
    rewards: List[float]
    terminated: bool
    total_reward: float
    steps: int
    duration: float
    control_plane_steps: List[Dict[str, Any]]
    control_plane_summary: Dict[str, Any]
    termination_reason: str
    conversation_history: List[Dict[str, Any]]
    llm_usage_summary: Dict[str, int] = field(default_factory=dict)
```

#### 6.2. Proposed `Trajectory` Definition

The updated `Trajectory` will store the final `EvaluateResult` for the entire episode. This single object will contain the overall score, per-step rewards, and all other evaluation metrics, making the trajectory object cleaner and more consistent with our data model.

```python
# Proposed changes for: eval_protocol/mcp/types.py
from eval_protocol.models import EvaluateResult, Message # Assuming Message model is accessible

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

#### 6.3. Mapping of Deprecated Fields

The old fields will be consolidated into the `evaluation_result` object as follows:

- **`total_reward: float`** -> `evaluation_result.score`
- **`rewards: List[float]`** -> Contained within `evaluation_result.step_outputs`, where each `StepOutput` has a `base_reward`.
- **`control_plane_summary: Dict`** -> The `reason`, `metrics`, and `error` fields of `evaluation_result`.
- **`control_plane_steps: List[Dict]`** -> The details from these steps can be incorporated into the `metrics` of each corresponding `StepOutput` in `evaluation_result.step_outputs`.
- **`llm_usage_summary: Dict`** -> Can be added as a custom metric in `evaluation_result.metrics`.
- **`conversation_history: List[Dict]`** -> Becomes `conversation_history: List[Message]`. Each `Message` object in the list can now carry its own per-turn evaluation metadata, providing a complete, self-contained record. 