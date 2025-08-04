## IA. Cross-Cutting Refinements & Enhanced E2E Testing Strategy

This section outlines action items for improving code sharing and the rigor of end-to-end (E2E) testing, primarily based on learnings from the Math Example. These should be considered for all examples to ensure robustness and maintainability.

**Action Items:**

0.  **Update open-r1 dataset generation script to make sure we have samples properly covered** (This seems related to the new data prep step for Math Example, ensure alignment)

1.  **Refactor Math Example Scripts for Shared Logic:**
    *   **Goal:** Reduce code duplication within `examples/math_example/`.
    *   **Action:** Create common utility functions (e.g., in a new `examples/math_example/utils.py` or within existing scripts if changes are minor) for tasks like loading `dataset.jsonl`. Evaluate if message formatting for `math_reward` can also be centralized.
    *   **Benefit:** Improved maintainability and consistency.

2.  **Strengthen TRL E2E Test Verification (`test_e2e_trl_grpo_integration_script`):**
    *   **Goal:** Increase confidence in the TRL integration E2E test.
    *   **Action:** Modify `tests/test_readiness.py` to parse the stdout of `trl_grpo_integration.py`. Verify that the script logs the expected number of training steps (e.g., 1 step when `TEST_MODE_TRL` is true) and that these steps report reasonable metrics (e.g., non-null loss/reward).
    *   **Benefit:** More robust verification beyond just script completion.

3.  **Standardize Script-Internal Mocking for External API Calls:**
    *   **Goal:** Ensure consistent and clear mocking for E2E tests of scripts making external API calls.
    *   **Action:** Review the `TEST_MOCK_FIREWORKS_PREVIEW` and `TEST_MOCK_FIREWORKS_REGEN` environment variable flags used in the Fireworks example scripts. Document this pattern of enabling/disabling internal mocks via environment variables. Ensure this approach is easily adaptable for other examples (e.g., a coding example that might call a sandboxed execution API).
    *   **Benefit:** Clearer, maintainable E2E tests for API-dependent scripts.

4.  **Configuration Management for Example Scripts:**
    *   **Goal:** Improve configurability and reduce hardcoding in example scripts.
    *   **Action:** For parameters like model names or specific generation settings that might vary, consider using simple configuration files (e.g., a `config.json` within each example folder) or command-line arguments instead of hardcoding them directly in Python scripts.
    *   **Benefit:** Easier modification and experimentation with examples.

5.  **"Out-of-the-Box" Smoke Test Suite (Future Consideration):**
    *   **Goal:** Provide a higher level of assurance that examples work with minimal setup, potentially including real (but controlled) external interactions.
    *   **Action:** Plan for a separate test suite (e.g., `tests/test_smoke_examples.py`) that runs selected example scripts:
        *   Without `TEST_MODE_TRL` (i.e., more training steps).
        *   With API calls directed to actual test/free-tier endpoints if feasible, or to sophisticated local mock servers (e.g., a local Ollama instance for LLM calls, a mock API server for Fireworks).
        *   Using small, self-contained datasets that are known to work.
    *   **Benefit:** Increased confidence in true out-of-the-box functionality. This suite would likely run less frequently due to its nature.

6.  **Align `examples/math_example/trl_grpo_integration.py` with `examples/trl_integration/grpo_example.py`:**
    *   **Goal:** Ensure consistency and that the specific math example benefits from or informs the generic TRL GRPO example.
    *   **Action:** Review both `trl_grpo_integration.py` (math example) and `grpo_example.py` (generic TRL example). Propagate successful patterns (e.g., use of `trainer.train()`, `GRPOConfig` setup, dataset handling for reward functions) from the math example to the generic one if it serves as a better template, or vice-versa.
    *   **Benefit:** Canonical examples are up-to-date and reflect best practices found during specific example development.
7.  **Gitignore Example-Specific Outputs:**
    *   **Goal:** Prevent committing transient build artifacts and large output files from examples.
    *   **Action:** Ensure `.gitignore` includes patterns to exclude common example output directories (e.g., TRL training outputs like `*_output/`, `*output*/`). (Partially done by adding `examples/math_example/math_grpo_trainer_output*/` and general patterns). Verify coverage for all example types.
    *   **Benefit:** Cleaner repository, faster git operations.
8.  **Standardize Data Preparation as First Step:**
    *   **Goal:** Ensure all examples that rely on specific datasets clearly define their data sourcing and preparation.
    *   **Action:** For each example (Coding, Function Calling, Composite), explicitly include a "0. Data Preparation & Cleaning" step in its plan if it uses a custom or processed dataset. This step should detail how its `dataset.jsonl` is created.
    *   **Benefit:** Reproducibility and clarity on data origins for each example.
