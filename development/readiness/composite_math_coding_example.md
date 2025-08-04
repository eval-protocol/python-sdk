## IV. Composite Example (Math + Coding) (`examples/composite_math_coding_example/`)
*   **Goal:** Evaluate both math and coding aspects in a single evaluation run.
*   **Approach:**
    *   Define a new dataset format where each sample has math and coding components.
    *   Develop a new composite reward function or an orchestration mechanism for `math_reward` and `fractional_code_reward`.
*   **Status:**
    *   [ ] 0. Data Preparation & Cleaning:
        *   [ ] Define process to create/curate `examples/composite_math_coding_example/dataset.jsonl`.
    *   [ ] 1. Dataset & Local Eval:
        *   [ ] Define composite problem structure.
        *   [ ] Create `examples/composite_math_coding_example/dataset.jsonl`
        *   [ ] Develop composite reward logic/function.
        *   [ ] Create `examples/composite_math_coding_example/local_eval.py`
    *   [ ] 2. Fireworks Preview API:
        *   [ ] Create `examples/composite_math_coding_example/fireworks_preview.py`
    *   [ ] 3. Fireworks Regeneration (Qwen3):
        *   [ ] Create `examples/composite_math_coding_example/fireworks_regenerate.py`
    *   [ ] 4. TRL Integration (GRPO):
        *   [ ] Create `examples/composite_math_coding_example/trl_grpo_integration.py`
    *   [ ] 5. End-to-End Integration Test:
        *   [ ] Add end-to-end script execution tests to `tests/test_readiness.py`
