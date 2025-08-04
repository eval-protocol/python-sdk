import argparse
import json
import logging

from datasets import Dataset, DatasetDict, load_dataset

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_dataset_to_prompts(
    dataset_name: str,
    split: str,
    output_file: str,
    id_column: str,
    query_column: str,
    ground_truth_column: str,
    dataset_config_name: str = None,
    max_samples: int = None,
):
    """
    Loads a dataset from Hugging Face, extracts relevant columns,
    and saves it as a JSONL file in the prompt format.

    Args:
        dataset_name: Name or path of the Hugging Face dataset.
        split: Dataset split to use (e.g., "train", "test").
        output_file: Path to save the output JSONL file.
        id_column: Name of the column containing the sample ID.
        query_column: Name of the column containing the user query/question.
        ground_truth_column: Name of the column containing the ground truth for evaluation.
        dataset_config_name: Specific configuration of the dataset if needed.
        max_samples: Maximum number of samples to convert.
    """
    logger.info(f"Starting dataset conversion: {dataset_name} (config: {dataset_config_name}, split: {split})")
    try:
        dataset = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=split,
            streaming=False,
            trust_remote_code=True,
        )
        logger.info(f"Successfully loaded dataset. Total rows in split '{split}': {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise

    if not isinstance(dataset, (Dataset, DatasetDict)):  # Should be Dataset after split
        logger.error(f"Loaded dataset is not of expected type (Dataset). Got: {type(dataset)}")
        raise TypeError("Loaded dataset is not of the expected type.")

    # Ensure specified columns exist
    required_columns = {id_column, query_column, ground_truth_column}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        logger.error(f"Dataset is missing required columns: {missing_columns}. Available: {dataset.column_names}")
        raise ValueError(f"Dataset missing columns: {missing_columns}")

    logger.info(f"Writing converted samples to {output_file}")
    count = 0
    with open(output_file, "w") as f:
        for i, example in enumerate(dataset):
            if max_samples is not None and count >= max_samples:
                logger.info(f"Reached max_samples limit of {max_samples}.")
                break

            try:
                prompt_record = {
                    "id": str(example[id_column]),
                    "user_query": str(example[query_column]),
                    "ground_truth_for_eval": example[
                        ground_truth_column
                    ],  # Keep as original type, likely str for JSON
                }
                f.write(json.dumps(prompt_record) + "\n")
                count += 1
                if count % 1000 == 0:
                    logger.info(f"Processed {count} samples...")
            except KeyError as e:
                logger.warning(f"Skipping sample {i} due to missing key: {e}. Data: {example}")
            except Exception as e:
                logger.warning(f"Skipping sample {i} due to error: {e}. Data: {example}")

    logger.info(f"Successfully converted {count} samples to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face dataset to JSONL prompt format.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name or path of the Hugging Face dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split to use (e.g., 'train', 'test').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSONL file.",
    )
    parser.add_argument("--id_column", type=str, required=True, help="Name of the column for sample ID.")
    parser.add_argument(
        "--query_column",
        type=str,
        required=True,
        help="Name of the column for user query.",
    )
    parser.add_argument(
        "--ground_truth_column",
        type=str,
        required=True,
        help="Name of the column for ground truth.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Specific configuration of the dataset.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert.",
    )

    args = parser.parse_args()

    convert_dataset_to_prompts(
        dataset_name=args.dataset_name,
        split=args.split,
        output_file=args.output_file,
        id_column=args.id_column,
        query_column=args.query_column,
        ground_truth_column=args.ground_truth_column,
        dataset_config_name=args.dataset_config_name,
        max_samples=args.max_samples,
    )
