import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from datasets import Dataset

try:
    from datasets import Dataset

    HAS_DATASETS_LIB = True
except ImportError:
    HAS_DATASETS_LIB = False
    if not TYPE_CHECKING:

        class Dataset:
            """Placeholder for HuggingFace Dataset for when the library is not installed."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass


def load_jsonl_to_hf_dataset(
    dataset_path: str,
    transform_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    prompt_column: str = "prompt",
    required_columns: Optional[List[str]] = None,
    dataset_filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Optional["Dataset"]:
    """
    Loads a JSONL file into a HuggingFace Dataset, optionally applying a
    transformation to each sample and ensuring required columns are present.

    Args:
        dataset_path: Path to the JSONL file.
        transform_fn: An optional function to apply to each raw dictionary (sample)
                      from the JSONL file. It should take a dict and return a dict
                      (or None to skip the sample).
        prompt_column: The name of the column expected to contain the prompt for TRL.
                       Defaults to "prompt".
        required_columns: A list of column names that must be present in the
                          final dataset (after transformation).
        dataset_filter_fn: An optional function to filter samples from the dataset
                           after transformation. It should take a dict and return
                           True to keep the sample, False to discard.

    Returns:
        A HuggingFace Dataset object, or None if the datasets library is not installed
        or if an error occurs.
    """
    if not HAS_DATASETS_LIB:
        print("The 'datasets' library is not installed. Please install it with 'pip install datasets'.")
        return None

    processed_samples: List[Dict[str, Any]] = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    raw_sample = json.loads(line.strip())

                    transformed_sample: Optional[Dict[str, Any]]
                    if transform_fn:
                        transformed_sample = transform_fn(raw_sample)
                    else:
                        transformed_sample = raw_sample

                    if transformed_sample is None:
                        continue

                    if dataset_filter_fn and not dataset_filter_fn(transformed_sample):
                        continue

                    processed_samples.append(transformed_sample)

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_number} in {dataset_path} due to JSON decode error: {e}")
                except Exception as e:
                    print(
                        f"Warning: Skipping line {line_number} in {dataset_path} due to error in transform_fn or filter_fn: {e}"
                    )

        if not processed_samples:
            print(f"Warning: No samples were processed from {dataset_path}.")
            return Dataset.from_list([])

        hf_dataset = Dataset.from_list(processed_samples)

        if prompt_column not in hf_dataset.column_names:
            raise ValueError(
                f"Dataset from {dataset_path} must contain a '{prompt_column}' column after transformation."
            )

        final_required_columns = set(required_columns or [])
        final_required_columns.add(prompt_column)

        for col in final_required_columns:
            if col not in hf_dataset.column_names:
                raise ValueError(
                    f"Dataset from {dataset_path} must contain a '{col}' column after transformation for the reward function/TRL."
                )

        return hf_dataset

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return None
    except ValueError as ve:
        print(f"Error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading dataset {dataset_path}: {e}")
        return None
