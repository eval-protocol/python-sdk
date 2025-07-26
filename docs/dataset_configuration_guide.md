# Dataset Configuration Guide

This guide explains the structure and fields used in YAML configuration files for datasets within the Reward Kit. These configurations are typically located in `conf/dataset/` or within an example's `conf/dataset/` directory (e.g., `examples/math_example/conf/dataset/`). They are processed by `eval_protocol.datasets.loader.py` using Hydra.

There are two main types of dataset configurations: **Base Datasets** and **Derived Datasets**.

## 1. Base Dataset Configuration

A base dataset configuration defines the connection to a raw data source and performs initial processing like column mapping.

**Example File**: `conf/dataset/base_dataset.yaml` (schema), `examples/math_example/conf/dataset/gsm8k.yaml` (concrete example)

### Key Fields:

*   **`_target_`** (Required)
    *   **Description**: Specifies the Python function to instantiate for loading this dataset.
    *   **Typical Value**: `eval_protocol.datasets.loader.load_and_process_dataset`
    *   **Example**: `_target_: eval_protocol.datasets.loader.load_and_process_dataset`

*   **`source_type`** (Required)
    *   **Description**: Defines the type of the data source.
    *   **Supported Values**:
        *   `"huggingface"`: For datasets hosted on the Hugging Face Hub.
        *   `"jsonl"`: For local datasets in JSON Lines format.
        *   `"fireworks"`: (Not yet implemented) For datasets hosted on Fireworks AI.
    *   **Example**: `source_type: huggingface`

*   **`path_or_name`** (Required)
    *   **Description**: Identifier for the dataset.
        *   For `huggingface`: The Hugging Face dataset name (e.g., `"gsm8k"`, `"cais/mmlu"`).
        *   For `jsonl`: Path to the `.jsonl` file (e.g., `"data/my_data.jsonl"`).
    *   **Example**: `path_or_name: "gsm8k"`

*   **`split`** (Optional)
    *   **Description**: Specifies the dataset split to load (e.g., `"train"`, `"test"`, `"validation"`). If loading a Hugging Face `DatasetDict` or multiple JSONL files mapped via `data_files`, this selects the split after loading.
    *   **Default**: `"train"`
    *   **Example**: `split: "test"`

*   **`config_name`** (Optional)
    *   **Description**: For Hugging Face datasets with multiple configurations (e.g., `"main"`, `"all"` for `gsm8k`). Corresponds to the `name` parameter in Hugging Face's `load_dataset`.
    *   **Default**: `null`
    *   **Example**: `config_name: "main"` (for `gsm8k`)

*   **`data_files`** (Optional)
    *   **Description**: Used for loading local files (like JSONL, CSV) with Hugging Face's `datasets.load_dataset`. Can be a single file path, a list, or a dictionary mapping split names to file paths.
    *   **Example**: `data_files: {"train": "path/to/train.jsonl", "test": "path/to/test.jsonl"}`

*   **`max_samples`** (Optional)
    *   **Description**: Maximum number of samples to load from the dataset (or from each split if a `DatasetDict` is loaded). If `null` or `0`, all samples are loaded.
    *   **Default**: `null`
    *   **Example**: `max_samples: 100`

*   **`column_mapping`** (Optional)
    *   **Description**: A dictionary to rename columns from the source dataset to a standard internal format. Keys are the new standard names (e.g., `"query"`, `"ground_truth"`), and values are the original column names in the source dataset. This mapping is applied by `eval_protocol.datasets.loader.py`.
    *   **Default**: `{"query": "query", "ground_truth": "ground_truth", "solution": null}`
    *   **Example (`gsm8k.yaml`)**:
        ```yaml
        column_mapping:
          query: "question"
          ground_truth: "answer"
        ```

*   **`preprocessing_steps`** (Optional)
    *   **Description**: A list of strings, where each string is a Python import path to a preprocessing function (e.g., `"eval_protocol.datasets.loader.transform_codeparrot_apps_sample"`). These functions are applied to the dataset after loading and before column mapping.
    *   **Default**: `[]`
    *   **Example**: `preprocessing_steps: ["my_module.my_preprocessor_func"]`

*   **`hf_extra_load_params`** (Optional)
    *   **Description**: A dictionary of extra parameters to pass directly to Hugging Face's `datasets.load_dataset()` (e.g., `trust_remote_code: True`).
    *   **Default**: `{}`
    *   **Example**: `hf_extra_load_params: {trust_remote_code: True}`

*   **`description`** (Optional, Metadata)
    *   **Description**: A brief description of the dataset configuration for documentation purposes.
    *   **Example**: `description: "GSM8K (Grade School Math 8K) dataset."`

## 2. Derived Dataset Configuration

A derived dataset configuration references a base dataset and applies further transformations, such as adding system prompts, changing the output format, or applying different column mappings or sample limits.

**Example File**: `examples/math_example/conf/dataset/base_derived_dataset.yaml` (schema), `examples/math_example/conf/dataset/gsm8k_math_prompts.yaml` (concrete example)

### Key Fields:

*   **`_target_`** (Required)
    *   **Description**: Specifies the Python function to instantiate for loading this derived dataset.
    *   **Typical Value**: `eval_protocol.datasets.loader.load_derived_dataset`
    *   **Example**: `_target_: eval_protocol.datasets.loader.load_derived_dataset`

*   **`base_dataset`** (Required)
    *   **Description**: A reference to the base dataset configuration to derive from. This can be the name of another dataset configuration file (e.g., `"gsm8k"`, which would load `conf/dataset/gsm8k.yaml`) or a full inline base dataset configuration object.
    *   **Example**: `base_dataset: "gsm8k"`

*   **`system_prompt`** (Optional)
    *   **Description**: A string that will be used as the system prompt. In the `evaluation_format`, this prompt is added as a `system_prompt` field alongside `user_query`.
    *   **Default**: `null`
    *   **Example (`gsm8k_math_prompts.yaml`)**: `"Solve the following math problem. Show your work clearly. Put the final numerical answer between <answer> and </answer> tags."`

*   **`output_format`** (Optional)
    *   **Description**: Specifies the final format for the derived dataset.
    *   **Supported Values**:
        *   `"evaluation_format"`: Converts dataset records to include `user_query`, `ground_truth_for_eval`, and optionally `system_prompt` and `id`. This is the standard format for many evaluation scenarios.
        *   `"conversation_format"`: (Not yet implemented) Converts to a list of messages.
        *   `"jsonl"`: Keeps records in a format suitable for direct JSONL output (typically implies minimal transformation beyond base loading and initial mapping).
    *   **Default**: `"evaluation_format"`
    *   **Example**: `output_format: "evaluation_format"`

*   **`transformations`** (Optional)
    *   **Description**: A list of additional transformation functions to apply after the base dataset is loaded and initial derived processing (like system prompt addition) is done. (Currently not fully implemented in `loader.py`).
    *   **Default**: `[]`

*   **`derived_column_mapping`** (Optional)
    *   **Description**: A dictionary for column mapping applied *after* the base dataset is loaded and *before* the `output_format` conversion. This can override or extend the base dataset's `column_mapping`. Keys are new names, values are names from the loaded base dataset.
    *   **Default**: `{}`
    *   **Example (`gsm8k_math_prompts.yaml`)**:
        ```yaml
        derived_column_mapping:
          query: "question"       # Maps 'question' from base gsm8k to 'query'
          ground_truth: "answer"  # Maps 'answer' from base gsm8k to 'ground_truth'
        ```
        *Note: These mapped columns (`query`, `ground_truth`) are then used by `convert_to_evaluation_format` to create `user_query` and `ground_truth_for_eval`.*

*   **`derived_max_samples`** (Optional)
    *   **Description**: Maximum number of samples for this derived dataset. If specified, this overrides any `max_samples` from the base dataset configuration for the purpose of this derived dataset.
    *   **Default**: `null`
    *   **Example**: `derived_max_samples: 5`

*   **`description`** (Optional, Metadata)
    *   **Description**: A brief description of this derived dataset configuration.
    *   **Example**: `description: "GSM8K dataset with math-specific system prompt in evaluation format."`

## How Configurations are Loaded

The `eval_protocol.datasets.loader.py` script uses Hydra to:
1.  Compose these YAML configurations.
2.  Instantiate the appropriate loader function (`load_and_process_dataset` or `load_derived_dataset`) with the parameters defined in the YAML.
3.  The loader functions then use these parameters to fetch data (e.g., from Hugging Face or local files), apply mappings, execute preprocessing steps, and format the data as requested.

This structured configuration approach allows for flexible and reproducible dataset management within the Reward Kit.
