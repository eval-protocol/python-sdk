# Running AIME/GPQA Evaluations in CI and Locally

This guide explains how to run the AIME2025 and GPQA evaluations using the
pytest-based `evaluation_test` decorator, how to control dataset size and
concurrency, how to select effort presets, and how to print/persist results
for CI dashboards/artifacts.

## Objectives
- Simple pass/fail: ensure evaluation configs don’t regress.
- Comparable metrics: capture aggregated accuracy across runs/rows.
- CI-friendly outputs: print summary lines to logs and save JSON artifacts.

## Prerequisites
- `FIREWORKS_API_KEY` set in the environment
- Install SDK: `pip install -e .[dev]`

## Controls
- Row limit
  - Default `max_dataset_rows=2` in each test decorator for quick CI.
  - Override centrally: `pytest --ep-max-rows=all` or `--ep-max-rows=50`.
- Concurrency
  - Set `max_concurrent_rollouts` in the decorator (recommend 4 for production Fireworks).
- Repeats
  - Set `num_runs` in the decorator (e.g., 4).
- Effort (Fireworks reasoning)
  - Provide `{"reasoning": {"effort": "low|medium|high"}}` in the test’s `rollout_input_params`.
  - The default rollout forwards it via LiteLLM `extra_body`.

## Printing & Persisting Results
- Flags:
  - `--ep-print-summary`: print concise summary lines at end of each eval
  - `--ep-summary-json=PATH`: write JSON with suite/model/agg_score/runs/rows/timestamp
- Example GitHub Actions snippet:
```yaml
- name: Run AIME low effort (full)
  run: |
    cd python-sdk
    pytest --ep-max-rows=all --ep-print-summary \
      --ep-summary-json=outputs/aime_low.json \
      -q examples/aime2025_chat_completion/tests/test_evaluation.py::test_aime2025_pointwise -q
- name: Upload AIME results
  uses: actions/upload-artifact@v4
  with:
    name: aime2025-low-summary
    path: python-sdk/outputs/aime_low.json
```

## Examples
### AIME (Low Effort, Full, Repeats=4, Concurrency=4)
```bash
cd python-sdk
pytest --ep-max-rows=all --ep-print-summary \
  --ep-summary-json=outputs/aime_low.json \
  -q examples/aime2025_chat_completion/tests/test_evaluation.py::test_aime2025_pointwise -q
```
Expected:
- Terminal summary: `EP Summary | suite=test_aime2025_pointwise model=... agg=0.530 runs=4 rows=...`
- JSON artifact at `outputs/aime_low.json`
- For `.../gpt-oss-120b`, low-effort pass rate should be ~≥ 0.50 when repeated

For medium/high effort, add `{"reasoning": {"effort": "medium|high"}}` to
`rollout_input_params` in the test decorator and rerun with a different JSON path.

### GPQA (Diamond, Low Effort)
```bash
cd python-sdk
pytest --ep-max-rows=all --ep-print-summary \
  --ep-summary-json=outputs/gpqa_low.json \
  -q examples/gpqa/tests/test_evaluation.py -q
```
Adjust repeats/concurrency/effort in the test decorator similarly to AIME.

## Pass/Fail Signals
- If `threshold_of_success` is set in a test, it will fail when aggregated score < threshold.
- Otherwise, printing and writing artifacts occur and the run succeeds for CI.

## Tips
- Use `--ep-max-rows` for toggling quick checks vs full evaluations without editing tests.
- Upload JSON artifacts for dashboards and historical comparisons.
- Keep concurrency conservative (e.g., 4) to avoid rate limiting.
