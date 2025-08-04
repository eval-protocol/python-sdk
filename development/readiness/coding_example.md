## II. Coding Example (`examples/coding_example/`)
*   **Core Reward Function(s):** `fractional_code_reward` (Python, with test cases)
*   **Dataset Source:** `development/CODING_DATASET.jsonl` (curate a small subset for the example)
*   **Status:**
    *   [ ] 0. Data Preparation & Cleaning:
        *   [ ] Define process to curate/generate `examples/coding_example/dataset.jsonl` from `development/CODING_DATASET.jsonl` or other sources.
    *   [ ] 1. Dataset & Local Eval:
        *   [ ] Create `examples/coding_example/dataset.jsonl` (curated subset)
        *   [ ] Create `examples/coding_example/local_eval.py`
    *   [ ] 2. Fireworks Preview API:
        *   [ ] Create `examples/coding_example/fireworks_preview.py`
    *   [ ] 3. Fireworks Regeneration (Qwen3):
        *   [ ] Create `examples/coding_example/fireworks_regenerate.py`
    *   [ ] 4. TRL Integration (GRPO):
        *   [ ] Create `examples/coding_example/trl_grpo_integration.py` (adapt from `examples/trl_integration/minimal_deepcoder_grpo_example.py`)
    *   [ ] 5. End-to-End Integration Test:
        *   [ ] Add end-to-end script execution tests to `tests/test_readiness.py`
