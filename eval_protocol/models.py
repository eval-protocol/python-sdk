import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from openai.types import CompletionUsage
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessageToolCall,
    FunctionCall,
)
from pydantic import BaseModel, ConfigDict, Field

from eval_protocol.get_pep440_version import get_pep440_version
from eval_protocol.human_id import generate_id


class ChatCompletionContentPartTextParam(BaseModel):
    text: str = Field(..., description="The text content.")
    type: Literal["text"] = Field("text", description="The type of the content part.")


class Message(BaseModel):
    """Chat message model with trajectory evaluation support."""

    role: str  # assistant, user, system, tool
    content: Optional[Union[str, List[ChatCompletionContentPartTextParam]]] = Field(
        default="", description="The content of the message."
    )
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    function_call: Optional[FunctionCall] = None
    control_plane_step: Optional[Dict[str, Any]] = None

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict) and "role" not in obj:
            raise ValueError("Role is required")
        return super().model_validate(obj, *args, **kwargs)


class MetricResult(BaseModel):
    """Result of a single metric evaluation.

    Attributes:
        is_score_valid (bool): Whether the score is valid for this metric (required).
        score (float): The score for this metric.
        reason (str): Explanation for the score.
    """

    is_score_valid: bool = True
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str

    def __getitem__(self, key: str) -> Any:
        if key in self.__fields__:  # Changed to __fields__ for Pydantic v1 compatibility
            value = getattr(self, key)
            return value
        raise KeyError(f"'{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.__fields__  # Changed to __fields__

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.__fields__.keys()  # Changed to __fields__

    def values(self):
        # For consistency with __getitem__ returning raw attribute values (including nested models)
        return [getattr(self, key) for key in self.__fields__.keys()]  # Changed to __fields__

    def items(self):
        return [(key, getattr(self, key)) for key in self.__fields__.keys()]  # Changed to __fields__

    def __iter__(self):
        return iter(self.__fields__.keys())  # Changed to __fields__


class StepOutput(BaseModel):
    """Defines the base reward and other metrics for a single conceptual step within a rollout,
    as determined by the user's reward function.
    """

    step_index: Union[int, str] = Field(
        description="User-defined index for the step (e.g., assistant message index, turn number). This is used by the system to map this output to the internal StepData."
    )
    base_reward: float = Field(description="Base reward calculated by the user's reward function for this step.")
    terminated: bool = Field(default=False, description="Whether the environment signaled termination at this step.")
    control_plane_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured info from the environment's control plane."
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional dictionary of custom metrics for this step.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional explanation for the step's base reward or metrics.",
    )


class EvaluateResult(BaseModel):
    """The complete result of an evaluator.
    For standard evaluation, it provides an overall score and component metrics.
    For Reinforcement Learning, it can also provide per-step base rewards via 'step_outputs'.

    This unified model serves both per-turn and per-trajectory evaluation scenarios.

    Attributes:
        score (float): The overall evaluation score.
        is_score_valid (bool): Whether the overall score is valid. Defaults to True.
        reason (Optional[str]): Optional explanation for the overall score.
        metrics (Dict[str, MetricResult]): Dictionary of component metrics for detailed evaluation.
        step_outputs (Optional[List[StepOutput]]): For RL, a list of outputs for each conceptual step,
                                                  providing base rewards.
        error (Optional[str]): Optional error message if evaluation failed.
        trajectory_info (Optional[Dict[str, Any]]): Additional trajectory-level information.
        final_control_plane_info (Optional[Dict[str, Any]]): The final control plane state that led to termination.
    """

    score: float = Field(..., description="The overall evaluation score, typically between 0.0 and 1.0.")
    is_score_valid: bool = Field(default=True, description="Whether the overall score is valid.")
    reason: Optional[str] = Field(default=None, description="Optional explanation for the overall score.")
    metrics: Dict[str, MetricResult] = Field(
        default_factory=dict,
        description="Dictionary of component metrics for detailed breakdown.",
    )

    # New field for RL per-step base rewards
    step_outputs: Optional[List[StepOutput]] = Field(
        default=None,
        description="For RL, a list of outputs for each conceptual step, providing base rewards.",
    )

    error: Optional[str] = Field(
        default=None,
        description="Optional error message if the evaluation itself encountered an issue.",
    )

    # New fields for unified trajectory and row-wise results
    trajectory_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional trajectory-level information (duration, steps, termination_reason, etc.).",
    )

    final_control_plane_info: Optional[Dict[str, Any]] = Field(
        default=None, description="The final control plane state that led to termination."
    )

    def __getitem__(self, key: str) -> Any:
        if key in self.__fields__:  # Changed to __fields__
            value = getattr(self, key)
            # If the value is a dict of MetricResult, and we want __getitem__ on metrics
            # to return a dict of dicts (rather than dict of MetricResult objects),
            # we'd need special handling here.
            # For now, return the raw attribute value, consistent with MetricResult.__getitem__
            return value
        raise KeyError(f"'{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.__fields__  # Changed to __fields__

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.__fields__.keys()  # Changed to __fields__

    def values(self):
        # For consistency with __getitem__ returning raw attribute values
        return [getattr(self, key) for key in self.__fields__.keys()]  # Changed to __fields__

    def items(self):
        return [(key, getattr(self, key)) for key in self.__fields__.keys()]  # Changed to __fields__

    def __iter__(self):
        return iter(self.__fields__.keys())  # Changed to __fields__


class CompletionParams(BaseModel):
    """Configuration for the language model used in the session."""

    model: str = Field(..., description="Model identifier (e.g., 'gpt-4.1', 'fireworks/llama')")
    temperature: Optional[float] = Field(None, description="Temperature setting for model generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    max_tool_calls: Optional[int] = Field(None, description="Maximum tool calls per turn")


class InputMetadata(BaseModel):
    """Comprehensive metadata for input to evaluation and logging systems."""

    model_config = ConfigDict(extra="allow")

    row_id: Optional[str] = Field(default_factory=generate_id, description="Unique string to ID the row")
    completion_params: Optional[CompletionParams] = Field(None, description="Completion endpoint parameters used")
    dataset_info: Optional[Dict[str, Any]] = Field(
        None, description="Dataset row details: seed, system_prompt, environment_context, etc"
    )
    session_data: Optional[Dict[str, Any]] = Field(
        None, description="Session metadata like timestamp (input only, no duration/usage)"
    )


class EvaluationThreshold(BaseModel):
    """Threshold configuration for evaluation tests.

    The success field is required - tests must specify a minimum success rate.
    The standard_deviation field is optional - if provided, tests must also meet the maximum standard deviation requirement.
    """

    success: float = Field(
        ..., description="Minimum success rate threshold (fraction of total score, 0.0 to 1.0)", ge=0.0, le=1.0
    )
    standard_deviation: Optional[float] = Field(
        None, description="Maximum standard deviation threshold (fraction of total score, 0.0 to 1.0)", ge=0.0, le=1.0
    )


class EvalMetadata(BaseModel):
    """Metadata about the evaluation that was run."""

    name: str = Field(..., description="Name of the evaluation")
    description: Optional[str] = Field(None, description="Description of the evaluation")
    version: str = Field(
        default_factory=get_pep440_version,
        description="Version of the evaluation. Should be populated with a PEP 440 version string.",
    )
    status: Optional[Literal["running", "finished", "error", "stopped"]] = Field(
        None, description="Status of the evaluation"
    )
    num_runs: int = Field(..., description="Number of times the evaluation was repeated")
    aggregation_method: str = Field(..., description="Method used to aggregate scores across runs")
    passed_threshold: Optional[EvaluationThreshold] = Field(
        None, description="Threshold configuration for test success"
    )
    passed: Optional[bool] = Field(None, description="Whether the evaluation passed based on the threshold")


class ExecutionMetadata(BaseModel):
    """Metadata about the execution of the evaluation."""

    invocation_id: Optional[str] = Field(
        default_factory=generate_id,
        description="The ID of the invocation that this row belongs to.",
    )

    experiment_id: Optional[str] = Field(
        default_factory=generate_id,
        description="The ID of the experiment that this row belongs to.",
    )

    rollout_id: Optional[str] = Field(
        default_factory=generate_id,
        description="The ID of the rollout that this row belongs to.",
    )

    run_id: Optional[str] = Field(
        None,
        description=("The ID of the run that this row belongs to."),
    )


class RolloutStatus(BaseModel):
    """Status of the rollout."""

    """
    running: Unfinished rollout which is still in progress.
    finished: Rollout finished successfully.
    error: Rollout failed.
    stopped: Rollout terminated unexpectedly (e.g. max step, control plane signal, user stop).
    """
    status: Literal["running", "finished", "error"] = Field("running", description="Status of the rollout.")
    termination_reason: Optional[str] = Field(
        "", description="reason of the rollout status, mapped to values in TerminationReason"
    )


class EvaluationRow(BaseModel):
    """
    Unified data structure for a single evaluation unit that contains messages,
    tools, and evaluation results. This can represent either a single turn evaluation
    or a complete trajectory evaluation.

    This model serves as the canonical format for evaluation data across the system,
    supporting both row-wise batch evaluation and trajectory-based RL evaluation.
    """

    # Core OpenAI ChatCompletion compatible conversation data
    messages: List[Message] = Field(description="List of messages in the conversation. Also known as a trajectory.")

    # Tool and function call information
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Available tools/functions that were provided to the agent."
    )

    # Input-related metadata (grouped together for cleaner organization)
    input_metadata: InputMetadata = Field(
        default_factory=InputMetadata,
        description="Metadata related to the input (dataset info, model config, session data, etc.).",
    )

    rollout_status: RolloutStatus = Field(
        default_factory=RolloutStatus,
        description="The status of the rollout.",
    )

    # Ground truth reference (moved from EvaluateResult to top level)
    ground_truth: Optional[str] = Field(
        default=None, description="Optional ground truth reference for this evaluation."
    )

    # Unified evaluation result
    evaluation_result: Optional[EvaluateResult] = Field(
        default=None, description="The evaluation result for this row/trajectory."
    )

    execution_metadata: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata,
        description="Metadata about the execution of the evaluation.",
    )

    # LLM usage statistics
    usage: Optional[CompletionUsage] = Field(
        default=None, description="Token usage statistics from LLM calls during execution."
    )

    created_at: datetime = Field(default_factory=datetime.now, description="The timestamp when the row was created.")

    eval_metadata: Optional[EvalMetadata] = Field(
        default=None, description="Metadata about the evaluation that was run."
    )

    pid: Optional[int] = Field(
        None,
        description="The PID of the process that created the row. This is used by the evaluation watcher to detect stopped evaluations.",
    )

    def is_trajectory_evaluation(self) -> bool:
        """
        Returns True if this represents a trajectory evaluation (has step_outputs),
        False if it represents a single turn evaluation.
        """
        return (
            self.evaluation_result is not None
            and self.evaluation_result.step_outputs is not None
            and len(self.evaluation_result.step_outputs) > 0
        )

    def get_conversation_length(self) -> int:
        """Returns the number of messages in the conversation."""
        return len(self.messages)

    def get_system_message(self) -> Message:
        """Returns the system message from the conversation. Returns empty Message if none found."""
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        if not system_messages:
            return Message(role="system", content="")
        return system_messages[0]

    def get_assistant_messages(self) -> List[Message]:
        """Returns only the assistant messages from the conversation."""
        return [msg for msg in self.messages if msg.role == "assistant"]

    def get_user_messages(self) -> List[Message]:
        """Returns only the user messages from the conversation."""
        return [msg for msg in self.messages if msg.role == "user"]

    def get_input_metadata(self, key: str, default: Any = None) -> Any:
        """Helper method to get a specific value from input_metadata (InputMetadata fields)."""
        if self.input_metadata is None:
            return default
        return getattr(self.input_metadata, key, default)

    def get_steps(self) -> int:
        """Get number of steps from control_plane_step data."""
        return len([msg for msg in self.messages if msg.control_plane_step])

    def get_total_reward(self) -> float:
        """Get total reward from control_plane_step data."""
        messages_with_control_plane = [msg for msg in self.messages if msg.control_plane_step]
        return (
            sum(msg.control_plane_step["reward"] for msg in messages_with_control_plane)
            if messages_with_control_plane
            else 0.0
        )

    def get_terminated(self) -> bool:
        """Get termination status from control_plane_step data."""
        messages_with_control_plane = [msg for msg in self.messages if msg.control_plane_step]
        return (
            any(msg.control_plane_step["terminated"] for msg in messages_with_control_plane)
            if messages_with_control_plane
            else False
        )

    def get_termination_reason(self) -> str:
        """Get termination reason from the final control_plane_step data."""
        # Find the last message with control_plane_step that has termination_reason
        for msg in reversed(self.messages):
            if msg.control_plane_step and msg.control_plane_step.get("termination_reason"):
                return msg.control_plane_step["termination_reason"]
        return "unknown"


# Original dataclass-based models for backwards compatibility
# These are deprecated and will be removed in a future version
# Use EvaluateResult and MetricResult instead
# MetricRewardOutput and RewardOutput are fully removed.


# --- Models for New Agent Evaluation Framework (V2) ---


class ResourceServerConfig(BaseModel):
    """
    Configuration for a resource server required by a task.
    """

    start_command: str = Field(
        description="The command to start the server. The string '{port}' will be replaced with a dynamically allocated free port."
    )
    health_check_url: str = Field(
        description="The URL to poll to check if the server is ready. The string '{port}' will be replaced with the allocated port."
    )


class EvaluationCriteriaModel(BaseModel):
    """
    Defines criteria for evaluating task success, often by querying the final state of a resource.
    """

    final_state_query: Optional[str] = Field(
        default=None,
        description="A query (e.g., SQL) to run on the final state of the resource.",
    )
    expected_query_result_transform: Optional[str] = Field(
        default=None,
        description="A Python lambda string (e.g., 'lambda x: x > 0') to transform and evaluate the query result to a boolean.",
    )

    # Explicit fields for ground truth data for BFCL evaluation
    ground_truth_function_calls: Optional[List[List[str]]] = Field(
        default=None, description="Ground truth function calls for BFCL evaluation."
    )
    ground_truth_comparable_state: Optional[Dict[str, Any]] = Field(
        default=None, description="Ground truth comparable state for BFCL evaluation."
    )

    # Future: Could include other complex evaluation logic or references


class TaskDefinitionModel(BaseModel):
    """
    Pydantic model for validating the structure of a V2 agent evaluation task definition file (YAML/JSON).
    """

    name: str = Field(description="Unique name for the task.")
    description: Optional[str] = Field(default=None, description="A brief description of the task.")

    resource_type: str = Field(
        description="The type of ForkableResource to use (e.g., 'SQLResource', 'PythonStateResource', 'FileSystemResource', 'DockerResource')."
    )
    base_resource_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dictionary passed to the base resource's setup() method.",
    )

    tools_module_path: Optional[str] = Field(
        default=None,
        description="Optional Python import path to a module containing custom tool functions for this task.",
    )
    reward_function_path: str = Field(
        description="Python import path to the reward function (e.g., 'my_module.my_reward_func')."
    )

    goal_description: Optional[str] = Field(
        default=None,
        description="A human-readable description of the agent's goal for this task.",
    )
    evaluation_criteria: Optional[EvaluationCriteriaModel] = Field(
        default=None,
        description="Criteria used by the Orchestrator to determine if the primary goal was achieved.",
    )

    initial_user_prompt: Optional[str] = Field(
        default=None,
        description="The initial prompt or message to start the agent interaction. Deprecated if 'messages' field is used for multi-turn.",
    )
    messages: Optional[List[Dict[str, Any]]] = Field(  # Explicit field for initial/multi-turn messages
        default=None,
        description="A list of messages to start the conversation, can represent multiple user turns for sequential processing.",
    )

    # PoC / Task specific parameters
    poc_max_turns: int = Field(
        default=3,
        ge=1,
        description="For PoC Orchestrator, the maximum number of interaction turns.",
    )

    # Allow other custom fields to be captured if needed by specific tasks or resources
    # These will be accessible via `model_extra` if `model_config` has `extra = 'allow'`
    # Or define a specific field:
    # custom_task_params: Dict[str, Any] = Field(default_factory=dict)
    resource_server: Optional[ResourceServerConfig] = Field(
        default=None,
        description="Configuration for a background server required for the task.",
    )

    num_rollouts: int = Field(
        default=1,
        ge=1,
        description="Number of parallel rollouts to execute for this task definition.",
    )

    # Data-driven evaluation fields
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to dataset file (JSONL) containing experimental conditions for data-driven evaluation.",
    )
    num_rollouts_per_sample: int = Field(
        default=1,
        ge=1,
        description="Number of rollouts to execute per sample from the dataset.",
    )

    class Config:
        extra = "allow"  # Allow and capture extra fields not explicitly defined
        # For Pydantic v2, it's model_config = {"extra": "allow"}
        # Assuming Pydantic v1 style for now based on existing file, can update if needed.
        # If using Pydantic v1, `Config.extra = "allow"` is correct.
        # For Pydantic v2, this should be:
        # from pydantic import ConfigDict
        # model_config = ConfigDict(extra='allow')
        # For Pydantic v1, `Config.extra = "allow"` is correct.


class MCPConfigurationServerStdio(BaseModel):
    """Represents a MCP configuration server."""

    command: str  # command to run the MCP server
    args: List[str] = Field(default_factory=list)  # to pass to the command
    env: List[str] = Field(default_factory=list)  # List of environment variables to verify exist in the environment


class MCPConfigurationServerUrl(BaseModel):
    """Represents a Remote MCP configuration server."""

    url: str  # url to the MCP server


class MCPMultiClientConfiguration(BaseModel):
    """Represents a MCP configuration."""

    mcpServers: Dict[str, Union[MCPConfigurationServerStdio, MCPConfigurationServerUrl]]
