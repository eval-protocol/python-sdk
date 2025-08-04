# Detailed Plan: APPS Coding Example for `eval-protocol run`

This plan details the steps to create a new example using the `codeparrot/apps` dataset and adapting logic similar to `verl.utils.reward_score.prime_code.compute_score`, leveraging the `eval-protocol run` CLI. This follows the general playbook outlined in `verl_replication_playbook.md`.

**Target `verl` Pairing:**
*   **Dataset:** `codeparrot/apps` (Hugging Face)
*   **`verl` Reward Logic:** `verl.utils.reward_score.prime_code.compute_score` (focus on non-sandbox execution initially).

**Phase 1: Reward Function (New/Adapted)**
1.  **Analyze `verl.utils.reward_score.prime_code`:**
    *   Understand its inputs: typically a code solution string and ground truth (likely test cases).
    *   How it executes code (if it does, this is the complex part) or checks correctness.
    *   How it parses test cases from the ground truth string.
2.  **Design `eval_protocol.rewards.apps_coding_reward.evaluate_apps_solution`:**
    *   Create a new file: `eval_protocol/rewards/apps_coding_reward.py`.
    *   Signature: `def evaluate_apps_solution(messages: List[Message], ground_truth: str, **kwargs) -> EvaluateResult:`
    *   `ground_truth`: Will be a JSON string from APPS' `input_output` field, containing lists of inputs and expected outputs for multiple test cases.
    *   **Core Logic (Initial Simplification - No Sandbox):**
        *   Extract the code solution from `messages[-1].content`. This might require parsing (e.g., if the code is in a specific markdown block).
        *   Parse the `ground_truth` JSON string into test cases.
        *   **Challenge:** Securely executing arbitrary code is complex and risky.
        *   **Initial Approach (No Execution):**
            *   Focus on static checks or simpler heuristics if possible.
            *   *Idea 1:* If canonical solutions are available in the dataset (APPS has `solutions`), perhaps do a similarity check (e.g., BLEU, ROUGE, or AST-based similarity if feasible) against one of the reference solutions. This is a proxy for correctness.
            *   *Idea 2:* If the `ground_truth` also contains expected outputs for specific inputs, and the model's code *also* generates these outputs as text (e.g. in comments or a specific format), we could parse and compare these. This is less likely.
            *   *Idea 3 (Placeholder for Execution):* For now, the reward function might simply check if the code is parsable by a Python AST parser (`ast.parse`). Score 1 if parsable, 0 otherwise, with a metric indicating "parsability." This is a very basic first step.
        *   **Future Goal (Full Execution):** Integrate a sandboxed code execution environment (e.g., E2B, Docker). This is a large feature. The reward function would then iterate through test cases, run the code with inputs, and compare outputs.
    *   Return `EvaluateResult` with score (e.g., fraction of test cases passed, or 0/1 for simpler checks) and detailed metrics.
    *   Add `@reward_function` decorator.

**Phase 2: Data Preparation**
1.  **Conversion Script:** Create a new script `scripts/convert_apps_to_prompts.py`.
    *   Input: `codeparrot/apps` dataset.
    *   Relevant source columns: `problem_id`, `question`, `input_output` (JSON string of test cases).
    *   Output JSONL format:
        ```json
        {
          "id": "problem_id_value",
          "user_query": "question_content",
          "ground_truth_for_eval": "input_output_json_string"
        }
        ```
2.  **Source Dataset Configuration:**
    *   Create `conf/dataset/apps_source.yaml`:
        ```yaml
        defaults: [- base_dataset, - _self_]
        source_type: huggingface
        path_or_name: "codeparrot/apps"
        # config_name: "main" # Or appropriate subset if available
        description: "Source configuration for codeparrot/apps dataset."
        column_mapping:
          # These are for convert_apps_to_prompts.py to find the right source data
          id_col: "problem_id"
          query_col: "question"
          ground_truth_col: "input_output"
        ```
3.  **Generate Sample Prompt JSONL:**
    *   Run `scripts/convert_apps_to_prompts.py` to create `development/apps_sample_prompts.jsonl` (e.g., 5 samples from a specific difficulty or subset).
    *   Command: (Example, assuming the script takes these args)
        ```bash
        .venv/bin/python scripts/convert_apps_to_prompts.py --dataset_name codeparrot/apps --split test --output_file development/apps_sample_prompts.jsonl --max_samples 5 --id_column problem_id --query_column question --ground_truth_column input_output
        ```
    *   This sample file **should be committed**.

**Phase 3: Hydra Prompt Dataset Configuration**
1.  Create `conf/dataset/apps_prompts.yaml`:
    ```yaml
    defaults: [- base_dataset, - _self_]
    source_type: "jsonl"
    path_or_name: "development/apps_sample_prompts.jsonl"
    split: "train"
    description: "Sample prompts from APPS dataset."
    column_mapping:
      id: "id"
      user_query: "user_query"
      ground_truth_for_eval: "ground_truth_for_eval"
      query: null
      ground_truth: null
      solution: null
    ```

**Phase 4: Example Run Configuration**
1.  Create directory: `examples/apps_coding_example/conf/`
2.  Create `examples/apps_coding_example/conf/run_eval.yaml`:
    ```yaml
    defaults: [- dataset: apps_prompts, - override hydra/job_logging: default, - override hydra/hydra_logging: default, - _self_]
    hydra: {searchpath: ['file://${oc.env:PWD}/conf']}

    system_prompt: "Please write a Python function to solve the following problem. Only output the function code, without any surrounding text or explanations."

    generation:
      enabled: true
      model_name: "accounts/fireworks/models/firefunction-v1" # A function-calling or coding model
      temperature: 0.0
      max_tokens: 2048
      cache_dir: "outputs/generated_responses_cache_apps"
      api_params: {rate_limit_qps: 1.0, max_retries: 3, max_concurrent_requests: 5}

    reward:
      function_path: "eval_protocol.rewards.apps_coding_reward.evaluate_apps_solution" # Path to the new reward function
      params: {} # Any params for the coding reward function

    evaluation_params: {limit_samples: null}
    output: {results_file: "apps_coding_example_results.jsonl"}
    logging_params: {batch_log_interval: 1}
    ```

**Phase 5: Minimal Example Code File (Optional)**
*   The main reward logic will be in `eval_protocol/rewards/apps_coding_reward.py`. No specific `main.py` needed in the example folder itself unless it defines a further wrapper.

**Phase 6: Documentation**
1.  Create `examples/apps_coding_example/README.md`:
    *   Describe APPS dataset, task, and the (initially simplified) reward logic.
    *   Instructions for data prep and running `eval-protocol run`.

**Phase 7: Testing**
1.  Implement the (simplified) `evaluate_apps_solution` reward function.
2.  Implement `scripts/convert_apps_to_prompts.py`.
3.  Generate `development/apps_sample_prompts.jsonl`.
4.  Run `eval-protocol run` with the APPS example config.
5.  Verify flow, generation (if enabled), caching, and that the reward function is called and returns a score.

**Key Challenge:** The `evaluate_apps_solution` reward function. Starting with a non-execution-based version is crucial for a quick first iteration. Full code execution is a separate, large feature.
