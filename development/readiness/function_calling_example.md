## III. Function Calling Example (`examples/tool_calling_example/`)
*   **Core Reward Function(s):** `exact_tool_match_reward` (directly used by example scripts)
*   **Status:**
    *   [X] 0. Data Preparation & Cleaning:
        *   [X] Dataset `examples/tool_calling_example/dataset.jsonl` provided and reviewed.
    *   [X] 1. Dataset & Local Eval:
        *   [X] `examples/tool_calling_example/dataset.jsonl` is in place.
        *   [X] Create `examples/tool_calling_example/local_eval.py` (Uses `exact_tool_match_reward`). Script runs and dataset passes 100%.
    *   [X] 2. Fireworks Preview API:
        *   [X] Create `examples/tool_calling_example/fireworks_preview.py`. Script runs, uses metric folder wrapper pointing to `exact_tool_match_reward`. (Note: Live API preview may be blocked by server-side secret config issue, but fallback simulation works).
    *   [X] 3. Fireworks Regeneration (Qwen3):
        *   [X] Create `examples/tool_calling_example/fireworks_regenerate.py` (Uses `exact_tool_match_reward` via metric wrapper, with placeholder for Qwen3 API call; testable with mocks).
    *   [X] 4. TRL Integration (GRPO):
        *   [X] Create `examples/tool_calling_example/trl_grpo_integration.py` (Scaffold created, to use `exact_tool_match_reward`; testable with TRL mocks).
    *   [X] 5. End-to-End Integration Test:
        *   [X] Add end-to-end script execution tests to `tests/test_readiness.py` (Tests added for all above scripts, using mocks where appropriate, targeting `exact_tool_match_reward`).
