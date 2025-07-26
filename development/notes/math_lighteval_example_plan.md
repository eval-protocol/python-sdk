# Detailed Plan: MATH-lighteval Example for `eval-protocol run`

This plan details the steps to create a new example using the `DigitalLearningGmbH/MATH-lighteval` dataset and the existing `eval_protocol.rewards.math.math_reward` function, leveraging the `eval-protocol run` CLI. This follows the general playbook outlined in `verl_replication_playbook.md`.

**Target `verl` Pairing:**
*   **Dataset:** `DigitalLearningGmbH/MATH-lighteval` (Hugging Face)
*   **`verl` Reward Logic:** `verl.utils.reward_score.math.compute_score`

**Phase 1: Reward Function**
1.  **Decision:** Use the existing `eval_protocol.rewards.math.math_reward`.
2.  **Verification (Mental Check):**
    *   `MATH-lighteval` answers often use `\\boxed{}`. Our `math_reward`'s `extract_numbers` function prioritizes `\\boxed{}`. This should be compatible.
    *   The `ground_truth_for_eval` will be the full solution string from `MATH-lighteval`, and `math_reward` will extract the answer from it.

**Phase 2: Data Preparation**
1.  **Conversion Script:** Use the existing `examples/math_example/convert_dataset.py`.
    *   This script takes a source dataset and converts it to `{"id": ..., "user_query": ..., "ground_truth_for_eval": ...}` JSONL.
2.  **Source Dataset Configuration:**
    *   Create `conf/dataset/math_lighteval_source.yaml`:
        ```yaml
        # conf/dataset/math_lighteval_source.yaml
        # For examples/math_example/convert_dataset.py to read raw MATH-lighteval
        defaults:
          - base_dataset # Inherits _target_ for loader, etc.
          - _self_

        source_type: huggingface
        path_or_name: "DigitalLearningGmbH/MATH-lighteval"
        # config_name: "all" # Or "default", check HF dataset page for available configs. Assume default if not specified.

        description: "Source configuration for DigitalLearningGmbH/MATH-lighteval dataset."

        column_mapping:
          # 'convert_dataset.py' expects 'query' and 'ground_truth' from its source config's mapping
          query: "problem"        # Maps 'problem' column in HF dataset to 'query' for convert_dataset.py
          ground_truth: "solution"  # Maps 'solution' column in HF dataset to 'ground_truth' for convert_dataset.py
        ```
3.  **Generate Sample Prompt JSONL:**
    *   Run `convert_dataset.py` to create `development/math_lighteval_sample_prompts.jsonl`.
    *   Command:
        ```bash
        .venv/bin/python examples/math_example/convert_dataset.py dataset=math_lighteval_source dataset.split=test output.file_path=development/math_lighteval_sample_prompts.jsonl dataset.max_samples=10
        ```
    *   This sample file **should be committed** to the repository.
4.  **(Optional) Generate Full Prompt JSONL:**
    *   Document how to run `convert_dataset.py` for the full dataset, e.g., outputting to `math_lighteval_full_test_prompts.jsonl` (which would be gitignored).

**Phase 3: Hydra Prompt Dataset Configuration**
1.  Create `conf/dataset/math_lighteval_prompts.yaml`:
    ```yaml
    # conf/dataset/math_lighteval_prompts.yaml
    # For `eval-protocol run` to use the sample prompt JSONL
    defaults:
      - base_dataset
      - _self_

    source_type: "jsonl"
    path_or_name: "development/math_lighteval_sample_prompts.jsonl"
    split: "train" # JSONL is usually a single split

    description: "Sample prompts from MATH-lighteval (10 examples from test set)."

    column_mapping:
      id: "id"
      user_query: "user_query"
      ground_truth_for_eval: "ground_truth_for_eval"
      query: null # Standard base fields not directly used from this pre-formatted file
      ground_truth: null
      solution: null
    ```

**Phase 4: Example Run Configuration**
1.  Create example directory: `examples/math_lighteval_example/conf/`
2.  Create `examples/math_lighteval_example/conf/run_eval.yaml`:
    ```yaml
    # examples/math_lighteval_example/conf/run_eval.yaml
    defaults:
      - dataset: math_lighteval_prompts # Points to the sample prompt dataset config
      - override hydra/job_logging: default
      - override hydra/hydra_logging: default
      - _self_

    hydra:
      searchpath:
        - file://${oc.env:PWD}/conf # To find global dataset configs

    system_prompt: "Please solve the following math problem. Show your reasoning steps clearly. Enclose your final numerical answer in \\boxed{} tags."

    generation:
      enabled: true
      model_name: "accounts/fireworks/models/llama-v3p1-8b-instruct" # Or another suitable model
      temperature: 0.0
      max_tokens: 1536 # Math problems can have long solutions
      cache_dir: "outputs/generated_responses_cache_math_lighteval" # Relative to Hydra run output dir
      api_params: {rate_limit_qps: 1.0, max_retries: 3, max_concurrent_requests: 5}

    reward:
      function_path: "eval_protocol.rewards.math.math_reward"
      params:
        tolerance: 0.001
        # absolute_tolerance: 1e-8 # Default in math_reward
        require_units: false # Typically false for MATH dataset

    evaluation_params:
      limit_samples: null # Process all samples in the (sample) dataset by default

    output:
      results_file: "math_lighteval_example_results.jsonl"

    logging_params:
      batch_log_interval: 2
    ```

**Phase 5: Minimal Example Code File (Optional)**
*   No `main.py` or custom reward code needed in `examples/math_lighteval_example/` as we are using a library reward function. A simple placeholder `__init__.py` might be useful if the directory contains other Python utilities in the future, or just the `conf` and `README.md`.

**Phase 6: Documentation**
1.  Create `examples/math_lighteval_example/README.md`:
    *   Describe the MATH-lighteval dataset and the task.
    *   Instructions for generating `development/math_lighteval_sample_prompts.jsonl` using `convert_dataset.py` with `conf/dataset/math_lighteval_source.yaml`.
    *   Command to run the example:
        ```bash
        eval-protocol run --config-dir examples/math_lighteval_example/conf --config-name run_eval
        ```
    *   Explain key configuration overrides (e.g., `evaluation_params.limit_samples`, `generation.model_name`).

**Phase 7: Testing**
1.  Manually execute steps from Phase 2 (Data Prep) and Phase 6 (Run command from README).
2.  Verify outputs, caching, and evaluation scores for a few samples.
