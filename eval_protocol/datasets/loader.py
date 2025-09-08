"""
Hydra-based dataset loading and processing.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import datasets
from datasets import Dataset, DatasetDict
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

import importlib  # Added for dynamic function import

# Placeholder for Fireworks API client if needed in the future
# from ..fireworks_client import FireworksClient # Example

# --- Preprocessing Functions ---
# These can be moved to a separate processors.py if they grow numerous.


def transform_codeparrot_apps_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms a single sample from codeparrot/apps dataset to include
    a 'transformed_ground_truth' field compatible with apps_coding_reward.
    """
    gt_dict = {}
    # fn_name can be None or missing for some APPS problems (standard input based)
    if example.get("fn_name"):
        gt_dict["fn_name"] = example["fn_name"]

    input_output_str = example.get("input_output")
    if input_output_str:
        try:
            parsed_io = json.loads(input_output_str)
            # Ensure 'inputs' and 'outputs' keys exist in the parsed JSON
            # and are lists, as expected by apps_testing_util.py
            gt_dict["inputs"] = parsed_io.get("inputs", [])
            gt_dict["outputs"] = parsed_io.get("outputs", [])
            if not isinstance(gt_dict["inputs"], list) or not isinstance(gt_dict["outputs"], list):
                logger.warning(
                    f"Parsed input_output for problem_id {example.get('problem_id', 'Unknown')} "
                    f"does not contain 'inputs'/'outputs' as lists. IO: {input_output_str}"
                )
                # Fallback to empty lists if types are wrong to prevent downstream errors
                gt_dict["inputs"] = [] if not isinstance(gt_dict["inputs"], list) else gt_dict["inputs"]
                gt_dict["outputs"] = [] if not isinstance(gt_dict["outputs"], list) else gt_dict["outputs"]

        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse input_output JSON for problem_id {example.get('problem_id', 'Unknown')}. "
                f"Content: {input_output_str}"
            )
            # Initialize to empty lists to prevent downstream errors if JSON is malformed
            gt_dict["inputs"] = []
            gt_dict["outputs"] = []
    else:
        # If input_output field is missing or empty, provide empty lists
        gt_dict["inputs"] = []
        gt_dict["outputs"] = []
        logger.warning(f"Missing or empty input_output field for problem_id {example.get('problem_id', 'Unknown')}.")

    example["transformed_ground_truth"] = json.dumps(gt_dict)
    return example


# --- End Preprocessing Functions ---


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON in file {file_path}: {e} on line: {line.strip()}")
    return data


def load_and_process_dataset(
    source_type: str,
    path_or_name: str,
    split: Optional[str] = None,
    config_name: Optional[str] = None,
    data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None,
    max_samples: Optional[int] = None,
    # column_mapping: Optional[Dict[str, str]] = None, # To be used for processing
    # preprocessing_steps: Optional[List[str]] = None, # To be implemented
    hf_extra_load_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,  # Catch-all for other params
) -> Union[Dataset, DatasetDict]:
    """
    Loads a dataset from the specified source.

    Args:
        source_type: Type of dataset source ("huggingface", "jsonl", "fireworks").
        path_or_name: Path to file or Hugging Face dataset name/ID.
        split: Dataset split (e.g., "train", "test"). For HF, this is passed to load_dataset.
               For jsonl loaded via HF, this is also passed.
        config_name: Specific configuration of a Hugging Face dataset (its 'name').
        data_files: Path(s) to local data files for Hugging Face's load_dataset
                    (e.g., for loading local jsonl, csv into HF Dataset).
        max_samples: Maximum number of samples to load.
        hf_extra_load_params: Extra kwargs for Hugging Face's `datasets.load_dataset()`.
        kwargs: Additional arguments.

    Returns:
        Loaded dataset, typically as Hugging Face Dataset or DatasetDict.
    """
    # Hugging Face load_dataset always returns Dataset or DatasetDict in our supported modes
    loaded_dataset: Union[Dataset, DatasetDict]

    # Prepare kwargs for datasets.load_dataset, separating out custom ones
    load_kwargs_for_hf = hf_extra_load_params.copy() if hf_extra_load_params else {}

    # Pop custom parameters from kwargs before they are merged
    column_mapping_from_kwargs = kwargs.pop("column_mapping", None)
    preprocessing_steps_from_kwargs = kwargs.pop("preprocessing_steps", None)
    dataset_description = kwargs.pop("description", "No description provided.")

    # Pop all reward-kit specific metadata fields not intended for datasets.load_dataset
    eval_protocol_specific_keys = [
        "dataset_name",
        "pretty_name",
        "final_columns",
        "column_transformations",
        "output_columns_creation",
        "preprocess_functions",
        "postprocess_functions",
        "_target_",
        "dataset_type",
    ]

    for key in eval_protocol_specific_keys:
        if key in kwargs:
            logger.debug(f"Filtering out reward-kit specific config key: {key}")
            kwargs.pop(key, None)

    logger.info(f"Dataset description: {dataset_description}")

    load_kwargs_for_hf.update(kwargs)  # Merge remaining kwargs (actual HF load_dataset params)

    if source_type == "huggingface":
        if config_name:  # config_name is a standard HF param
            load_kwargs_for_hf["name"] = config_name
        # The 'split' argument for datasets.load_dataset can be complex.
        # If data_files is a dict mapping splits to files, 'split' might not be needed here,
        # as load_dataset will return a DatasetDict.
        # If data_files is a single file/list, or path_or_name is a hub ID, 'split' is used.
        if split and not (isinstance(data_files, dict) and split in data_files):
            load_kwargs_for_hf["split"] = split

        # trust_remote_code will be handled by HF_DATASETS_TRUST_REMOTE_CODE=1 env var

        loaded_dataset = datasets.load_dataset(
            path_or_name,
            data_files=data_files,
            # trust_remote_code removed, rely on env var
            **load_kwargs_for_hf,  # Remaining kwargs (e.g. download_mode if re-added)
        )
    elif source_type == "jsonl":
        # Using Hugging Face's 'json' loader for consistency and features.
        # trust_remote_code will be handled by HF_DATASETS_TRUST_REMOTE_CODE=1 env var
        # path_or_name can be a direct path to a .jsonl file for single file loading.
        # data_files can be used for more complex setups (multiple files, multiple splits).

        effective_data_files = data_files
        if not effective_data_files and path_or_name:
            if not path_or_name.endswith(".jsonl"):
                raise ValueError(
                    f"For source_type 'jsonl' without 'data_files', 'path_or_name' must be a .jsonl file. Got: {path_or_name}"
                )
            # If path_or_name is a single jsonl file, use it as data_files for the specified split or default 'train'
            effective_data_files = {split if split else "train": path_or_name}

        if not effective_data_files:
            raise ValueError(
                "For source_type 'jsonl', either 'path_or_name' to a .jsonl file or 'data_files' must be provided."
            )

        # The 'split' kwarg to load_dataset for local files behaves such that if data_files is a dict,
        # it returns a DatasetDict, and then you select the split. If data_files is a single path/list,
        # 'split' selects that split.
        hf_split_param = split
        if isinstance(effective_data_files, dict) and split:
            hf_split_param = None

        loaded_dataset = datasets.load_dataset(
            "json",
            data_files=effective_data_files,
            split=hf_split_param,
            # trust_remote_code removed, rely on env var
            **load_kwargs_for_hf,
        )

        if split and isinstance(loaded_dataset, DatasetDict):
            if split not in loaded_dataset:
                raise ValueError(
                    f"Split '{split}' not found in loaded jsonl DatasetDict. Available splits: {list(loaded_dataset.keys())}"
                )
            loaded_dataset = loaded_dataset[split]
        elif split and not isinstance(loaded_dataset, DatasetDict) and hf_split_param == split:
            pass
        elif not split and isinstance(loaded_dataset, DatasetDict):
            logger.info(
                f"Loaded multiple splits from JSONL: {list(loaded_dataset.keys())}. No specific split requested via 'split' arg."
            )

    elif source_type == "fireworks":
        # Placeholder for Fireworks dataset loading.
        # This would likely involve an API call to download a JSONL, then load it.
        # For now, it's not implemented.
        # Example:
        # client = FireworksClient() # Assuming a client exists
        # downloaded_file_path = client.download_dataset(path_or_name) # path_or_name is Fireworks dataset ID
        # loaded_dataset = datasets.load_dataset("json", data_files=downloaded_file_path, split=split, **load_kwargs)
        # os.remove(downloaded_file_path) # Clean up temp file
        raise NotImplementedError(
            "Fireworks dataset loading (source_type='fireworks') is not yet implemented. "
            "If you have a JSONL file from Fireworks, use source_type='jsonl'."
        )
    else:
        raise ValueError(f"Unsupported source_type: '{source_type}'. Must be 'huggingface', 'jsonl', or 'fireworks'.")

    if max_samples is not None and max_samples > 0:
        if isinstance(loaded_dataset, Dataset):
            if len(loaded_dataset) > max_samples:
                loaded_dataset = loaded_dataset.select(range(max_samples))
        elif isinstance(loaded_dataset, DatasetDict):
            for s_name in loaded_dataset.keys():
                if len(loaded_dataset[s_name]) > max_samples:
                    loaded_dataset[s_name] = loaded_dataset[s_name].select(range(max_samples))

    # Apply column mapping if provided
    if column_mapping_from_kwargs and isinstance(loaded_dataset, (Dataset, DatasetDict)):
        logger.info(f"Applying column mapping: {column_mapping_from_kwargs}")
        # Note: Column mapping should happen *after* preprocessing if preprocessors add new columns
        # that are then mapped. Or, mapping happens first, and preprocessors use the new names.
        # Current Hugging Face `map` function adds new columns, doesn't modify in place by default,
        # so preprocessors creating 'transformed_ground_truth' is fine before mapping it.
        # Let's assume mapping is done *after* preprocessing for now.
        pass  # Deferred until after preprocessing

    # Apply preprocessing steps
    if preprocessing_steps_from_kwargs and isinstance(loaded_dataset, (Dataset, DatasetDict)):
        logger.info(f"Applying preprocessing steps: {preprocessing_steps_from_kwargs}")
        for step_path in preprocessing_steps_from_kwargs:
            try:
                module_path, func_name = step_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                preprocessor_func = getattr(module, func_name)

                if isinstance(loaded_dataset, Dataset):
                    # Pass existing column names to avoid issues if map tries to remove them by default
                    # and they are needed by subsequent steps or final output.
                    # However, if the preprocessor is designed to remove columns, this might interfere.
                    # For now, assume preprocessors add/modify columns.
                    # `batched=False` is default for `map` but can be specified by preprocessor if needed.
                    loaded_dataset = loaded_dataset.map(preprocessor_func)
                elif isinstance(loaded_dataset, DatasetDict):
                    for s_name in loaded_dataset.keys():
                        logger.info(f"Applying preprocessor {func_name} to split '{s_name}'")
                        loaded_dataset[s_name] = loaded_dataset[s_name].map(preprocessor_func)
                logger.info(f"Successfully applied preprocessor: {step_path}")
            except Exception as e:
                logger.error(
                    f"Failed to apply preprocessing step {step_path}: {e}",
                    exc_info=True,
                )
                raise  # Re-raise to halt execution if a preprocessor fails

    # Apply column mapping (now after preprocessing)
    if column_mapping_from_kwargs and isinstance(loaded_dataset, (Dataset, DatasetDict)):
        logger.info(f"Applying column mapping (post-preprocessing): {column_mapping_from_kwargs}")
        if isinstance(loaded_dataset, Dataset):
            # Filter out mappings where the old name is null/empty or doesn't exist
            # column_mapping_from_kwargs format: {new_name: old_name}
            valid_mapping = {
                old_name: new_name
                for new_name, old_name in column_mapping_from_kwargs.items()
                if old_name and old_name in loaded_dataset.column_names
            }
            if valid_mapping:
                # Ensure no attempt to rename to an existing column not part of this specific mapping op
                # This is complex; rename_columns handles conflicts by appending '_'.
                # For safety, let's check if a 'new' name is already a column and not the 'old' one.
                final_mapping = {}
                for old_name, new_name in valid_mapping.items():
                    if new_name in loaded_dataset.column_names and new_name != old_name:
                        logger.warning(
                            f"Attempting to map column '{old_name}' to '{new_name}', but '{new_name}' already exists and is not '{old_name}'. This may lead to unexpected behavior or errors. Skipping this specific rename."
                        )
                    else:
                        final_mapping[old_name] = new_name

                if final_mapping:
                    loaded_dataset = loaded_dataset.rename_columns(final_mapping)
                else:
                    logger.info("Column mapping resulted in no columns to rename after validation.")
            else:
                logger.warning(
                    "Column mapping provided but resulted in no valid columns to rename (original columns not found or new names empty)."
                )

        elif isinstance(loaded_dataset, DatasetDict):
            for s_name in loaded_dataset.keys():
                current_split_dataset = loaded_dataset[s_name]
                valid_mapping = {
                    old_name: new_name
                    for new_name, old_name in column_mapping_from_kwargs.items()
                    if old_name and old_name in current_split_dataset.column_names
                }
                if valid_mapping:
                    final_mapping = {}
                    for old_name, new_name in valid_mapping.items():
                        if new_name in current_split_dataset.column_names and new_name != old_name:
                            logger.warning(
                                f"For split '{s_name}', attempting to map column '{old_name}' to '{new_name}', but '{new_name}' already exists and is not '{old_name}'. Skipping this specific rename for the split."
                            )
                        else:
                            final_mapping[old_name] = new_name

                    if final_mapping:
                        loaded_dataset[s_name] = current_split_dataset.rename_columns(final_mapping)
                    else:
                        logger.info(
                            f"Column mapping for split '{s_name}' resulted in no columns to rename after validation."
                        )
                else:
                    logger.warning(f"Column mapping for split '{s_name}' resulted in no valid columns to rename.")

    return loaded_dataset


def apply_column_mapping(dataset: Dataset, column_mapping: Dict[str, str]) -> Dataset:
    """
    Apply column mapping to rename dataset columns.

    Args:
        dataset: The dataset to rename columns in
        column_mapping: Dict mapping new names to existing column names

    Returns:
        Dataset with renamed columns
    """
    # Filter out null mappings and reverse the mapping (old_name -> new_name)
    rename_mapping = {}
    for new_name, old_name in column_mapping.items():
        if old_name is not None and old_name in dataset.column_names:
            rename_mapping[old_name] = new_name

    if rename_mapping:
        dataset = dataset.rename_columns(rename_mapping)

    return dataset


def convert_to_evaluation_format(
    dataset: Dataset,
    system_prompt: Optional[str] = None,
    query_column: str = "query",
    ground_truth_column: str = "ground_truth",
) -> Dataset:
    """
    Convert dataset to evaluation format with user_query and ground_truth_for_eval.

    Args:
        dataset: Input dataset
        system_prompt: Optional system prompt to prepend to queries
        query_column: Name of the query/question column
        ground_truth_column: Name of the ground truth/answer column

    Returns:
        Dataset in evaluation format
    """

    def transform_example(example):
        # Keep user query separate from system prompt
        user_query = example.get(query_column, "")

        # Extract ground truth
        ground_truth = example.get(ground_truth_column, "")

        # Create evaluation format with separate system prompt
        result = {"user_query": user_query, "ground_truth_for_eval": ground_truth}
        if system_prompt:
            result["system_prompt"] = system_prompt

        # Preserve id if it exists
        if "id" in example:
            result["id"] = example["id"]
        elif query_column in example:
            # Generate a simple id from the query if no id exists
            result["id"] = str(hash(example[query_column]))[1:8]  # Simple hash-based id

        return result

    return dataset.map(transform_example)


def load_derived_dataset(
    base_dataset: Union[str, DictConfig],
    system_prompt: Optional[str] = None,
    output_format: str = "evaluation_format",
    transformations: Optional[List[str]] = None,
    derived_column_mapping: Optional[Dict[str, str]] = None,
    derived_max_samples: Optional[int] = None,
    **kwargs: Any,
) -> Dataset:
    """
    Load a derived dataset that references a base dataset and applies transformations.

    Args:
        base_dataset: Either a string name of a dataset config or a DictConfig
        system_prompt: Optional system prompt to add to queries
        output_format: Format to convert the dataset to
        transformations: List of additional transformations to apply
        derived_column_mapping: Column mapping for the derived dataset
        derived_max_samples: Maximum samples for the derived dataset
        kwargs: Additional arguments

    Returns:
        Transformed dataset
    """
    # Load base dataset
    if isinstance(base_dataset, str):
        # Load base dataset configuration by name
        # Try to find the config in the current Hydra config search path first
        try:
            from hydra.core.global_hydra import GlobalHydra

            # Check if Hydra is already initialized
            if GlobalHydra.instance().is_initialized():
                # Try to use existing Hydra context first
                try:
                    base_cfg = compose(config_name=f"dataset/{base_dataset}")
                except Exception:
                    # If that fails, try using the project root config directory
                    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../conf"))
                    if os.path.exists(config_dir):
                        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
                            base_cfg = compose(config_name=f"dataset/{base_dataset}")
                    else:
                        raise FileNotFoundError(f"Config directory not found: {config_dir}")
            else:
                # Try to initialize with the project root config path
                config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../conf"))
                if os.path.exists(config_dir):
                    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
                        base_cfg = compose(config_name=f"dataset/{base_dataset}")
                else:
                    raise FileNotFoundError(f"Config directory not found: {config_dir}")
        except Exception as e:
            raise ValueError(f"Failed to load base dataset config '{base_dataset}': {e}")

        # The compose() returns a config with nested 'dataset' key if it's a full config
        if "dataset" in base_cfg:
            base_dataset_cfg = base_cfg.dataset
        else:
            base_dataset_cfg = base_cfg

        # Instantiate the base dataset
        base_loaded_dataset = instantiate(base_dataset_cfg)
    elif isinstance(base_dataset, DictConfig):
        # Base dataset is already a config object
        base_loaded_dataset = instantiate(base_dataset)
    else:
        raise ValueError(f"base_dataset must be a string or DictConfig, got {type(base_dataset)}")

    # Ensure we have a Dataset (not DatasetDict)
    if isinstance(base_loaded_dataset, DatasetDict):
        # Use the first available split or 'train' if available
        if "train" in base_loaded_dataset:
            dataset = base_loaded_dataset["train"]
        else:
            dataset = list(base_loaded_dataset.values())[0]
    else:
        dataset = base_loaded_dataset

    # Apply derived column mapping if provided
    if derived_column_mapping:
        dataset = apply_column_mapping(dataset, derived_column_mapping)

    # Apply max samples if specified
    if derived_max_samples is not None and derived_max_samples > 0:
        if len(dataset) > derived_max_samples:
            dataset = dataset.select(range(derived_max_samples))

    # Apply format conversion
    if output_format == "evaluation_format":
        dataset = convert_to_evaluation_format(
            dataset,
            system_prompt=system_prompt,
            query_column="query",
            ground_truth_column="ground_truth",
        )
    elif output_format == "conversation_format":
        # TODO: Implement conversation format conversion if needed
        raise NotImplementedError("conversation_format not yet implemented")
    elif output_format == "jsonl":
        # Keep as-is, already in a compatible format
        pass
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    # TODO: Apply additional transformations if specified
    if transformations:
        raise NotImplementedError("Custom transformations not yet implemented")

    return dataset
