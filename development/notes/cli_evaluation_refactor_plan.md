# Plan: Integrate Advanced Evaluation into Reward Kit CLI

**Objective:** Move the core logic from `examples/math_example/main.py` (dataset loading, system prompt handling, model response generation with caching & API client features, and evaluation orchestration) into the `eval-protocol` core library and expose it via an enhanced or new CLI command. The example `main.py` (and similar examples) should become minimal, primarily defining custom reward logic if necessary.

**[COMPLETED] Phase 1: Core Library Enhancements (`eval_protocol/`)**

1.  **[COMPLETED] New Orchestration Module (`eval_protocol/execution/pipeline.py`):**
    *   Houses `EvaluationPipeline` for main orchestration logic.
    *   Handles: Prompt Dataset loading, System prompt, Generation (via ModelClient), Caching, Dynamic Reward Function loading, Result collection.
2.  **[COMPLETED] Model Client Abstraction (`eval_protocol/generation/clients.py`):**
    *   `ModelClient` ABC created.
    *   `FireworksModelClient` implemented with `aiohttp`, basic retry, and auth error handling.
3.  **[COMPLETED] Caching Logic (`eval_protocol/generation/cache.py`):**
    *   `ResponseCache` class implemented for file-based JSON storage, keyed by sample ID, prompt, model, temp.
4.  **[COMPLETED] Configuration Models (`eval_protocol/config_models.py`):**
    *   Placeholder file created, then removed as Hydra structured configs are sufficient for now.
5.  **[COMPLETED] Dynamic Reward Function Loading Utility (`eval_protocol/utils/module_loader.py`):**
    *   `load_function` helper created.

**[COMPLETED] Phase 2: CLI Enhancements (`eval_protocol/cli.py` and `eval_protocol/cli_commands/`)**

1.  **[COMPLETED] New CLI Command (`eval-protocol run`):**
    *   Implemented as a Hydra application (`eval_protocol/cli_commands/run_eval_cmd.py`).
    *   Default configuration schema at `conf/cli/run_eval_config.yaml`.
    *   Supports CLI overrides for pipeline parameters.
2.  **[COMPLETED] Update `eval_protocol/cli.py`:** Added the `run` subcommand and delegation to the Hydra app.
3.  **[COMPLETED] Create `eval_protocol/cli_commands/run_eval_cmd.py`:** Houses the Hydra entry point and logic for the `run` command.

**[COMPLETED] Phase 3: Refactor `examples/math_example/`**

1.  **[COMPLETED] `examples/math_example/main.py` (Minimal):**
    *   Simplified to define a wrapper reward function `evaluate` (as per user feedback for consistency with potential preview API expectations).
2.  **[COMPLETED] New Hydra Config for CLI (`examples/math_example/conf/run_math_eval.yaml`):**
    *   Created to configure the `eval-protocol run` command for the math example.
3.  **[COMPLETED] Update `examples/math_example/README.md`:**
    *   Updated to focus on the new `eval-protocol run` CLI workflow.
4.  **[COMPLETED] Cleanup:** Removed old/redundant example configs and scripts (`local_eval.py`, `main_config.yaml`, old planning docs, `gsm8k_local_jsonl.yaml`). Updated `.gitignore`.

**Phase 4: Scale Testing & Client Robustness (Next Steps)**

1.  **Run on Full GSM8K Test Set:**
    *   Execute: `.venv/bin/python -m eval_protocol.cli run dataset=gsm8k_full_test_prompts evaluation_params.limit_samples=null` (or a large number like 1319).
    *   Monitor performance, API call volume, and caching effectiveness.
    *   Verify results for a larger set of diverse samples.
2.  **Enhance `FireworksModelClient`:**
    *   Implement more robust rate limiting (e.g., using an asynchronous token bucket algorithm).
    *   Integrate a library like `tenacity` for configurable retry strategies with exponential backoff and jitter for API calls.
    *   Refine concurrency management if the current `asyncio.Semaphore` approach shows limitations at scale.
    *   Define and use specific custom exceptions for API errors (e.g., `ModelAuthError`, `RateLimitError`, `ModelGenerationError`) to be raised by the client and handled by the pipeline.

**Phase 5: Formal Testing (Next Steps)**

1.  **Unit Tests:**
    *   `EvaluationPipeline`: Mock dependencies (dataset loader, model client, cache, reward function) to test orchestration logic, config handling, and control flow for different scenarios (e.g., generation enabled/disabled, cache hit/miss, errors from components).
    *   `FireworksModelClient`: Mock `aiohttp.ClientSession.post` to test API request construction, response parsing, retry logic, and error handling (429, 401/403, 5xx).
    *   `ResponseCache`: Test key generation, `put`/`get` operations, cache miss/hit, behavior with `temperature != 0.0`, and handling of corrupted cache files.
    *   `module_loader.load_function`: Test successful loading and error cases (module not found, function not found, not callable).
2.  **Integration Tests:**
    *   For `eval-protocol run` CLI command:
        *   Use `hydra.experimental.compose` and `hydra.experimental.initialize` to set up test configurations programmatically.
        *   Invoke `run_evaluation_command_logic` (or the `hydra_cli_entry_point`).
        *   Alternatively, run `eval-protocol run` as a subprocess and assert on output files or console logs.
        *   Test with small, self-contained mock datasets and a simple mock reward function.
        *   Verify end-to-end flow including (mocked) generation, caching, and evaluation.

**Phase 6: Replicate to Other Examples (Future)**

1.  Identify other examples (e.g., from `verl` project, other internal use cases) that fit the generate-then-evaluate pattern.
2.  For each identified example:
    *   Create necessary prompt dataset converters (if source format differs).
    *   Define Hydra dataset configurations for these prompt datasets in `conf/dataset/`.
    *   Implement custom reward functions in the example's directory if needed.
    *   Create an example-specific Hydra run configuration (like `run_math_eval.yaml`) for the `eval-protocol run` command.
    *   Update or create a README for the example, explaining how to use `eval-protocol run`.

**Phase 7: Broader Documentation (Future - was Phase 4)**

1.  Update main project documentation (e.g., in `docs/`) for the new `eval-protocol run` CLI command, its comprehensive configuration options, and typical usage patterns.
2.  Document the `EvaluationPipeline` architecture and its components for developers wanting to extend or understand the framework.
3.  Provide guidance on implementing and using custom `ModelClient` subclasses for different model providers.
4.  Explain the prompt dataset format and how to prepare data for the pipeline.
