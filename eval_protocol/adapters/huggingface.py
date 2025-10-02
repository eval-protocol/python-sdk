"""HuggingFace Datasets adapter for Eval Protocol.

This adapter allows loading datasets from HuggingFace Hub with arbitrary
transformation functions to convert them to EvaluationRow format.
"""

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional

from eval_protocol.models import CompletionParams, EvaluationRow, InputMetadata, Message
from .base import BaseAdapter

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset, DatasetDict, load_dataset  # pyright: ignore[reportAttributeAccessIssue]
except ImportError:
    raise ImportError("HuggingFace datasets not installed. Install with: pip install 'eval-protocol[huggingface]'")

# Type alias for transformation function
TransformFunction = Callable[[Dict[str, Any]], Dict[str, Any]]


class HuggingFaceAdapter(BaseAdapter):
    """Generic adapter to load HuggingFace datasets with custom transformations.

    This adapter loads datasets from HuggingFace Hub and applies a user-provided
    transformation function to convert each row to the format expected by
    EvaluationRow.

    The transformation function should take a dataset row dictionary and return:
    {
        'messages': List[Dict] - list of message dictionaries with 'role' and 'content'
        'ground_truth': Optional[str] - expected answer/output
        'metadata': Optional[Dict] - any additional metadata to preserve
        'tools': Optional[List[Dict]] - tool definitions for tool calling scenarios
    }

    Examples:
        Simple Q&A dataset:
        >>> def transform(row):
        ...     return {
        ...         'messages': [{'role': 'user', 'content': row['question']}],
        ...         'ground_truth': row['answer'],
        ...         'metadata': {'category': row.get('category')}
        ...     }
        >>> adapter = HuggingFaceAdapter("my-dataset", transform_fn=transform)
        >>> rows = list(adapter.get_evaluation_rows(split="test", limit=10))

        Math problems with system prompt:
        >>> def gsm8k_transform(row):
        ...     return {
        ...         'messages': [
        ...             {'role': 'system', 'content': 'Solve step by step.'},
        ...             {'role': 'user', 'content': row['question']}
        ...         ],
        ...         'ground_truth': row['answer'],
        ...         'metadata': {'dataset': 'gsm8k'}
        ...     }
        >>> adapter = HuggingFaceAdapter("gsm8k", config_name="main", transform_fn=gsm8k_transform)
    """

    def __init__(
        self,
        dataset_id: str,
        transform_fn: TransformFunction,
        config_name: Optional[str] = None,
        revision: Optional[str] = None,
        **load_dataset_kwargs,
    ):
        """Initialize the HuggingFace adapter.

        Args:
            dataset_id: HuggingFace dataset identifier (e.g., "gsm8k", "squad", "org/dataset")
            transform_fn: Function to transform dataset rows to evaluation format
            config_name: Optional dataset configuration name
            revision: Optional dataset revision/commit hash
            **load_dataset_kwargs: Additional arguments to pass to load_dataset
        """
        self.dataset_id = dataset_id
        self.transform_fn = transform_fn
        self.config_name = config_name
        self.revision = revision
        self.load_dataset_kwargs = load_dataset_kwargs

        # Load the dataset
        self.dataset = self._load_dataset()

    @classmethod
    def from_local(
        cls,
        path: str,
        transform_fn: TransformFunction,
        **load_dataset_kwargs,
    ) -> "HuggingFaceAdapter":
        """Create adapter from local dataset file.

        Args:
            path: Path to local dataset file (JSON, JSONL, CSV, etc.)
            transform_fn: Function to transform dataset rows
            **load_dataset_kwargs: Additional arguments to pass to load_dataset

        Returns:
            HuggingFaceAdapter instance
        """
        # Determine file format
        if path.endswith(".jsonl"):
            dataset_type = "json"
        elif path.endswith(".json"):
            dataset_type = "json"
        elif path.endswith(".csv"):
            dataset_type = "csv"
        elif path.endswith(".parquet"):
            dataset_type = "parquet"
        else:
            # Let HuggingFace auto-detect
            dataset_type = None

        load_kwargs = {"data_files": path, **load_dataset_kwargs}

        return cls(dataset_id=dataset_type or "json", transform_fn=transform_fn, **load_kwargs)

    def _load_dataset(self) -> "Dataset | DatasetDict":
        """Load the dataset from HuggingFace Hub or local source."""
        try:
            kwargs = {}
            if self.config_name:
                kwargs["name"] = self.config_name
            if self.revision:
                kwargs["revision"] = self.revision

            kwargs.update(self.load_dataset_kwargs)

            return load_dataset(self.dataset_id, **kwargs)

        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to load dataset %s: %s", self.dataset_id, e)
            raise

    def get_evaluation_rows(
        self,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **completion_params_kwargs,
    ) -> Iterator[EvaluationRow]:
        """Convert dataset entries to EvaluationRow format.

        Args:
            split: Dataset split to use (if dataset has multiple splits)
            limit: Maximum number of rows to return
            offset: Number of rows to skip
            model_name: Model name for completion parameters
            temperature: Temperature for completion parameters
            max_tokens: Max tokens for completion parameters
            **completion_params_kwargs: Additional completion parameters

        Yields:
            EvaluationRow: Converted evaluation rows
        """
        # Select dataset split
        dataset = self.dataset
        if isinstance(self.dataset, DatasetDict):
            if split is None:
                # Use first available split
                split = list(self.dataset.keys())[0]
                logger.info("No split specified, using: %s", split)
            dataset = self.dataset[split]
        elif split is not None:
            logger.warning("Split '%s' specified but dataset is not split", split)

        # Apply offset and limit
        total_rows = len(dataset)
        end_idx = min(offset + limit, total_rows) if limit else total_rows

        if offset >= total_rows:
            logger.warning("Offset %d is greater than dataset size %d", offset, total_rows)
            return

        # Create completion parameters
        completion_params: CompletionParams = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **completion_params_kwargs,
        }

        # Convert each row
        for i in range(offset, end_idx):
            try:
                raw_row = dataset[i]
                eval_row = self._convert_row_to_evaluation_row(raw_row, i, completion_params, split)
                yield eval_row
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning("Failed to convert row %d: %s", i, e)
                continue

    def _convert_row_to_evaluation_row(
        self,
        raw_row: Dict[str, Any],
        row_index: int,
        completion_params: CompletionParams,
        split: Optional[str] = None,
    ) -> EvaluationRow:
        """Convert a single dataset row to EvaluationRow format.

        Args:
            raw_row: Raw dataset row dictionary
            row_index: Index of the row in the dataset
            completion_params: Completion parameters to use
            split: Dataset split name

        Returns:
            EvaluationRow object
        """
        # Apply user transformation
        transformed = self.transform_fn(raw_row)

        # Validate required fields
        if "messages" not in transformed:
            raise ValueError("Transform function must return 'messages' field")

        # Convert message dictionaries to Message objects
        messages = []
        for msg_dict in transformed["messages"]:
            if not isinstance(msg_dict, dict):
                raise ValueError("Each message must be a dictionary")
            if "role" not in msg_dict:
                raise ValueError("Each message must have a 'role' field")

            messages.append(
                Message(
                    role=msg_dict["role"],
                    content=msg_dict.get("content"),
                    name=msg_dict.get("name"),
                    tool_call_id=msg_dict.get("tool_call_id"),
                    tool_calls=msg_dict.get("tool_calls"),
                    function_call=msg_dict.get("function_call"),
                )
            )

        # Extract other fields
        ground_truth = transformed.get("ground_truth")
        tools = transformed.get("tools")
        user_metadata = transformed.get("metadata", {})

        # Create dataset info
        dataset_info = {
            "dataset_id": self.dataset_id,
            "config_name": self.config_name,
            "revision": self.revision,
            "split": split,
            "row_index": row_index,
            "transform_function": (
                self.transform_fn.__name__ if hasattr(self.transform_fn, "__name__") else "anonymous"
            ),
        }

        # Add user metadata
        dataset_info.update(user_metadata)

        # Add original row data (with prefix to avoid conflicts)
        for key, value in raw_row.items():
            dataset_info[f"original_{key}"] = value

        # Create input metadata
        input_metadata = InputMetadata(
            row_id=f"{self.dataset_id}_{row_index}",
            completion_params=completion_params,
            dataset_info=dataset_info,
            session_data={
                "dataset_source": "huggingface",
                "timestamp": None,
            },
        )

        return EvaluationRow(
            messages=messages,
            tools=tools,
            input_metadata=input_metadata,
            ground_truth=str(ground_truth) if ground_truth is not None else None,
        )

    def get_splits(self) -> List[str]:
        """Get available dataset splits.

        Returns:
            List of available split names
        """
        if isinstance(self.dataset, DatasetDict):
            return list(self.dataset.keys())
        else:
            return ["train"]  # Default split name for non-split datasets

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset.

        Returns:
            Dictionary with dataset information
        """
        info = {
            "dataset_id": self.dataset_id,
            "config_name": self.config_name,
            "revision": self.revision,
            "splits": self.get_splits(),
            "transform_function": (
                self.transform_fn.__name__ if hasattr(self.transform_fn, "__name__") else "anonymous"
            ),
        }

        # Add split sizes
        if isinstance(self.dataset, DatasetDict):
            info["split_sizes"] = {split: len(data) for split, data in self.dataset.items()}
        else:
            info["total_size"] = len(self.dataset)

        return info


def create_huggingface_adapter(
    dataset_id: str,
    transform_fn: TransformFunction,
    config_name: Optional[str] = None,
    revision: Optional[str] = None,
    **load_dataset_kwargs,
) -> HuggingFaceAdapter:
    """Factory function to create a HuggingFace adapter.

    Args:
        dataset_id: HuggingFace dataset identifier
        transform_fn: Function to transform dataset rows to evaluation format
        config_name: Optional configuration name
        revision: Optional dataset revision/commit hash
        **load_dataset_kwargs: Additional arguments for load_dataset

    Returns:
        HuggingFaceAdapter instance
    """
    return HuggingFaceAdapter(
        dataset_id=dataset_id,
        transform_fn=transform_fn,
        config_name=config_name,
        revision=revision,
        **load_dataset_kwargs,
    )


# Convenience functions for common datasets
def create_gsm8k_adapter(
    system_prompt: Optional[str] = None,
    revision: Optional[str] = None,
) -> HuggingFaceAdapter:
    """Create adapter specifically configured for GSM8K dataset.

    Args:
        system_prompt: Optional system prompt for math problems
        revision: Optional dataset revision/commit

    Returns:
        HuggingFaceAdapter configured for GSM8K
    """
    default_system_prompt = (
        "You are a helpful assistant that solves math problems step by step. "
        "Show your work and provide the final answer."
    )

    system_content = system_prompt or default_system_prompt

    def gsm8k_transform(row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform GSM8K row to evaluation format."""
        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": row["question"]},
            ],
            "ground_truth": row["answer"],
            "metadata": {
                "dataset": "gsm8k",
                "question_length": len(row["question"]),
                "answer_length": len(row["answer"]),
            },
        }

    return create_huggingface_adapter(
        dataset_id="gsm8k",
        config_name="main",
        transform_fn=gsm8k_transform,
        revision=revision,
    )


def create_math_adapter(
    system_prompt: Optional[str] = None,
    revision: Optional[str] = None,
) -> HuggingFaceAdapter:
    """Create adapter specifically configured for MATH competition dataset.

    Args:
        system_prompt: Optional system prompt for math problems
        revision: Optional dataset revision/commit

    Returns:
        HuggingFaceAdapter configured for MATH dataset
    """
    default_system_prompt = (
        "You are an expert mathematician. Solve this advanced math problem step by step, showing detailed work."
    )

    system_content = system_prompt or default_system_prompt

    def math_transform(row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform MATH dataset row to evaluation format."""
        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": row["problem"]},
            ],
            "ground_truth": row["solution"],
            "metadata": {
                "dataset": "hendrycks_math",
                "type": row.get("type", "unknown"),
                "level": row.get("level", "unknown"),
                "problem_length": len(row["problem"]),
                "solution_length": len(row["solution"]),
            },
        }

    return create_huggingface_adapter(
        dataset_id="hendrycks/competition_math",
        transform_fn=math_transform,
        revision=revision,
    )
