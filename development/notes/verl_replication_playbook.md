# Playbook: Replicating `verl` Examples in `eval-protocol`

This document outlines a general playbook for adapting dataset/reward function pairings from the `verl` project (or similar sources) into new examples for `eval-protocol`, leveraging the `eval-protocol run` CLI command and its underlying evaluation pipeline.

## I. General Playbook Steps

For each `verl` dataset/reward function pair to replicate:

1.  **Identify Target `verl` Pairing & Analyze Components:**
    *   Clearly identify the source dataset name (e.g., Hugging Face ID) and the specific `verl` reward function module/method used (e.g., `verl.utils.reward_score.prime_code.compute_score`).
    *   Thoroughly review the `verl` reward function's logic:
        *   What are its exact inputs (e.g., model solution string, ground truth format)?
        *   What preprocessing or extraction does it do on the solution and ground truth?
        *   What are the success criteria and how is the score calculated?
        *   Does it require external dependencies or environment setups (e.g., code execution sandbox - to be initially deferred if too complex)?
    *   Examine the source dataset:
        *   What are the relevant columns for prompt, solution/answer, and any necessary metadata (e.g., test cases for coding)?
        *   How is the ground truth structured in the source dataset?

2.  **Adapt or Implement Reward Function in `eval-protocol`:**
    *   **Decision Point:**
        *   Can an existing `eval_protocol.rewards.*` function be used directly or with minor parameter adjustments?
        *   Is the `verl` logic simple enough to be a new standalone function in `eval_protocol.rewards/`?
        *   Is the logic highly specific to this example and best placed in `examples/new_example_name/custom_rewards.py`?
    *   **Implementation:**
        *   Create the new Python file for the reward function if needed.
        *   Ensure the function signature is `def reward_name(messages: List[Message], ground_truth: str, **kwargs) -> EvaluateResult:`.
        *   Adapt the `verl` logic:
            *   Extract the assistant's response from `messages[-1].content`.
            *   Parse `ground_truth` string if it contains complex data (e.g., JSON for test cases).
            *   Implement the core scoring logic.
            *   Return an `EvaluateResult` object, populating `score`, `reason`, and any relevant `metrics`.
        *   Add the `@reward_function` decorator.
        *   Add clear docstrings and comments.

3.  **Data Preparation (Prompt Dataset Creation):**
    *   **Conversion Script:**
        *   Determine if `examples/math_example/convert_dataset.py` can be reused (by configuring its source column mappings) or if a new, dedicated script (e.g., `scripts/convert_new_dataset_type.py`) is needed due to unique source data structures.
        *   The script must output a JSONL file where each line is an object:
            ```json
            {"id": "unique_sample_id", "user_query": "The prompt/question for the model", "ground_truth_for_eval": "The string data the reward function expects as ground truth"}
            ```
        *   Ensure robust `id` generation (e.g., from a dataset ID column or `dataset_name_split_index`).
    *   **Execution:** Run the conversion script to generate:
        *   A small sample prompt JSONL (e.g., 5-10 samples) in `development/` for quick testing (e.g., `development/new_example_sample_prompts.jsonl`). This file *should* be committed.
        *   Optionally, instructions on how to generate the full prompt dataset (e.g., `new_example_full_test_prompts.jsonl`), which would typically be gitignored.

4.  **Create Hydra Dataset Configurations (`conf/dataset/`):**
    *   **Source Dataset Config (if using `convert_dataset.py`):**
        *   E.g., `conf/dataset/new_example_source.yaml`.
        *   Defines `source_type`, `path_or_name` for the original dataset, and `column_mapping` for `convert_dataset.py` to find the raw query and ground truth parts.
    *   **Prompt Dataset Config (for `eval-protocol run`):**
        *   E.g., `conf/dataset/new_example_prompts.yaml` (for the sample) and potentially `new_example_full_prompts.yaml` (for the full set).
        *   `source_type: jsonl`.
        *   `path_or_name`: Points to the generated prompt JSONL file (e.g., `development/new_example_sample_prompts.jsonl`).
        *   `column_mapping`: Maps `id`, `user_query`, `ground_truth_for_eval` to the keys in the JSONL. Sets standard `query`, `ground_truth`, `solution` from `base_dataset.yaml` to `null` as they are not directly used from this pre-formatted prompt dataset.

5.  **Create Example Run Configuration:**
    *   Create a new example directory: `examples/new_example_name/`.
    *   Inside, create `conf/run_eval.yaml` (or `run_new_example_eval.yaml`).
    *   This file configures the `eval-protocol run` command:
        *   `defaults`: Includes `- dataset: new_example_prompts` (pointing to the sample prompt dataset config).
        *   `hydra.searchpath`: Ensure `['file://${oc.env:PWD}/conf']` is present so global dataset configs are found.
        *   `system_prompt`: Tailored to the task.
        *   `generation`:
            *   `enabled`: `true` (usually).
            *   `model_name`: A suitable default model.
            *   `cache_dir`: e.g., `"generated_responses_cache_new_example"`.
            *   Other generation parameters.
        *   `reward`:
            *   `function_path`: Full Python path to the adapted/new reward function.
            *   `params`: Any parameters the reward function takes via `**kwargs`.
        *   `evaluation_params`: `limit_samples` (e.g., to match the sample dataset size by default).
        *   `output.results_file`: e.g., `"new_example_results.jsonl"`.

6.  **Minimal Example Code File (Optional):**
    *   If the reward function is implemented within the example directory (e.g., `examples/new_example_name/custom_rewards.py`), ensure `reward.function_path` in the run config points to it correctly.
    *   A `main.py` in the example directory is generally not needed for the `eval-protocol run` flow unless it's defining the reward function itself.

7.  **Documentation (`examples/new_example_name/README.md`):**
    *   Briefly describe the task, dataset, and reward logic.
    *   Provide clear, copy-pasteable instructions:
        *   How to run data preparation (if applicable).
        *   How to execute the example using `eval-protocol run --config-dir examples/new_example_name/conf --config-name run_eval.yaml`.
        *   Common CLI overrides (e.g., changing dataset, model, number of samples).
    *   Explain the expected output and where to find results.

8.  **Testing:**
    *   Manually run the data conversion script.
    *   Manually run `eval-protocol run` with the new example configuration.
    *   Verify:
        *   Correct dataset loading.
        *   System prompt application.
        *   Model response generation (check API calls if it's the first run for new prompts).
        *   Caching (re-run to see if cache hits occur).
        *   Correct reward function execution and scoring for a few known cases if possible.
        *   Output file creation and format.

## II. Initial Example Candidates & Brief Plans

### A. MATH-lighteval Example (Relatively Straightforward)

*   **Verl Reference:** `DigitalLearningGmbH/MATH-lighteval` & `verl.utils.reward_score.math.compute_score`.
*   **Eval Protocol Target:**
    *   **Reward Function:** Use existing `eval_protocol.rewards.math.math_reward`. Verify its compatibility with `MATH-lighteval`'s answer format (often `\\boxed{}`).
    *   **Data Prep:** Use `examples/math_example/convert_dataset.py`.
        *   Source Config: `conf/dataset/math_lighteval_source.yaml` (maps `problem`->`query`, `solution`->`ground_truth`).
        *   Output: `development/math_lighteval_sample_prompts.jsonl`.
    *   **Prompt Config:** `conf/dataset/math_lighteval_prompts.yaml`.
    *   **Run Config:** `examples/math_lighteval_example/conf/run_eval.yaml`.
        *   `system_prompt`: "Solve... Output in `\\boxed{}`."
        *   `reward.function_path`: `"eval_protocol.rewards.math.math_reward"`.
    *   **README:** `examples/math_lighteval_example/README.md`.

### B. APPS Coding Example (More Complex)

*   **Verl Reference:** `codeparrot/apps` & `verl.utils.reward_score.prime_code.compute_score`.
*   **Eval Protocol Target:**
    *   **Reward Function (New/Adapted):** `eval_protocol.rewards.prime_code_adapted_reward.evaluate_coding_solution`.
        *   Input `ground_truth_for_eval`: JSON string of test cases from APPS' `input_output` field.
        *   Logic: Parse test cases. Execute the assistant's code solution against these test cases. This requires a secure code execution environment (big feature - for initial pass, might mock or simplify this part, e.g., by checking for specific keywords or simple execution if possible, or deferring full execution).
    *   **Data Prep:** New script `scripts/convert_apps_to_prompts.py`.
        *   Source Config: `conf/dataset/apps_source.yaml` (maps `question`->`query`, `input_output`->`ground_truth`).
        *   Output: `development/apps_sample_prompts.jsonl`.
    *   **Prompt Config:** `conf/dataset/apps_prompts.yaml`.
    *   **Run Config:** `examples/apps_coding_example/conf/run_eval.yaml`.
        *   `reward.function_path`: Path to the new adapted coding reward function.
        *   `system_prompt`: Coding-specific instructions (e.g., language, class/function structure).
    *   **README:** `examples/apps_coding_example/README.md`.

This playbook and the initial example outlines should provide a good foundation for parallelizing the work.
