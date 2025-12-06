## GEPA-training Interface Design for Eval Protocol

### Goals

- **Tunable prompts for existing benchmarks**: Allow benchmarks like `test_aime25.py` and `test_gpqa.py` to expose parts of their configuration (e.g., system prompts) as training parameters, without changing their core evaluation logic.
- **Tight coupling with `@evaluation_test`**: Reuse the same rollout configuration, datasets, and metrics that are already defined via `evaluation_test`, instead of duplicating that configuration in a separate training API.
- **GEPA as one optimizer backend**: Provide a clean integration point for GEPA (and potentially other optimizers later) without requiring benchmarks to depend on DSPy or GEPA directly.

### High-Level Architecture

- **Benchmark file (e.g., `test_aime25.py`)**
  - Continues to define:
    - Dataset adapter (`aime2025_dataset_adapter`).
    - `@evaluation_test(...)`-decorated function (e.g., `test_aime25_pointwise`) that:
      - Uses `SingleTurnRolloutProcessor` (or another processor).
      - Computes per-row metrics and sets `row.evaluation_result`.
  - Adds *optional* training wiring at the bottom, under `if __name__ == "__main__":`, that:
    - Imports a training/core API from `eval_protocol.training`.
    - Specifies what is tunable (e.g., the system prompt) and how to adapt rows using a candidate.
    - Invokes a train routine (GEPA-based or otherwise).

- **Training core**
  - Provides a single central abstraction:
    - **`EPParameters`**: Encapsulates everything `evaluation_test` knows about the eval in a structured form:
      - One field for every parameter that `evaluation_test` accepts (dataset sources, adapters, completion params, rollout processor, aggregation, thresholds, etc.), after parsing/env overrides.
    - **Candidate representation**: Start with `dict[str, str]` (e.g., `{"system_prompt": "..."}`), anticipating future extensions (few-shot examples, tool docs, etc.).
  - Includes helper utilities to:
    - Build an `EPParameters` instance by introspecting an `@evaluation_test`-decorated function.
    - Run a single candidate or a batch of candidates through the full rollout + evaluation pipeline, returning aggregate scores (and optionally per-row scores).

- **GEPA adapter (e.g., `eval_protocol/training/gepa_adapter.py`)**
  - Wraps the training core and GEPA’s API:
    - Accepts:
      - An `EPConfig`.
      - A candidate space definition (for now, implicit via `dict[str, str]` keys).
      - GEPA configuration (budget, reflection model, seed, component selection strategy, etc.).
    - Provides:
      - A GEPA-compatible metric interface that:
        - Given a candidate, uses `EPConfig` (and benchmark-specific logic such as a custom `dataset_adapter`) to:
          - Construct or adapt rows for that candidate.
          - Run rollouts (reusing the same processors and params as the test).
          - Compute scalar scores (e.g., mean exact-match over a batch).
      - A training routine that returns:
        - A `best_candidate: dict[str, str]`.
        - Optional rich result object (e.g., mapping to `GEPAResult`, additional stats).

### Relationship to `evaluation_test` and `__ep_params__`

- Existing `evaluation_test` code will attach:

```python
ep_params: dict[str, Any] = {
    "rollout_processor": rollout_processor,
    "server_script_path": server_script_path,
    "mcp_config_path": mcp_config_path,
    "rollout_processor_kwargs": rollout_processor_kwargs,
    "mode": mode,
}
setattr(dual_mode_wrapper, "__ep_params__", ep_params)
```

- Design direction:
  - **Use `__ep_params__` as the single source of truth**.
  - **`__ep_params__` should contain all effective `evaluation_test` parameters**, including:
    - Parsed `completion_params` (after env overrides).
    - Dataset sources (`input_dataset`, `input_rows`, dataloaders, and `dataset_adapter`), after `parse_ep_*` transforms.
    - `aggregation_method`, `num_runs`, `max_dataset_rows`, etc.
    - Rollout and mode information (processor, kwargs, concurrency limits, mode).
  - The training core can then **directly convert `__ep_params__` into an `EPParameters` instance** without maintaining a separate training-only config.

- Training core will expose:
  - A factory like:

    ```python
    def build_ep_parameters_from_test(
        test_fn: TestFunction,
    ) -> EPParameters:
        ...
    ```

  - This function:
    - Reads `test_fn.__ep_params__`.
    - Reconstructs how to:
      - Load and preprocess the dataset.
      - Configure the rollout processor (`RolloutProcessorConfig`).
      - Run rollouts and then apply the row-level metric (by calling the decorated test function in a library mode).

- Training code (e.g., `python test_aime25.py`) then becomes:
  - Import the test function (e.g., `test_aime25_pointwise`).
  - Build an `EPParameters` from it.
  - Call into a GEPA-based trainer that uses the `EPParameters`.

### TODO for derek to figure out: how to store the changing system prompts.

- **Where tuned prompts live (storage format and location)**:
  - GEPA already supports a `run_dir` for logging and checkpoints.
  - We need to decide:
    - Whether EP should:
      - Treat `run_dir` as the canonical store and optionally add a small `best_candidate.json` there; or
      - Provide an additional EP-level artifact format.
  - For now, storage is left as an **explicit design TODO** and can be finalized once we have the core/adapter in place.

### Work Split: Person A vs Person B

#### Person A – training Core & `evaluation_test` Integration

- **1. Extend `evaluation_test` metadata (no behavior change)**
  - Populate a single `__ep_config__` dict on the decorated test function that includes:
    - Dataset specification (paths / input_rows / dataloaders, `dataset_adapter`, `max_dataset_rows`, etc.) after `parse_ep_*`.
    - Parsed `completion_params` (after env overrides like `parse_ep_completion_params_overwrite`).
    - Rollout settings (`rollout_processor`, `rollout_processor_kwargs`, `mode`, `max_concurrent_rollouts`, `max_concurrent_evaluations`).
    - Aggregation and threshold metadata.
  - Ensure:
    - Backwards compatibility for existing tests.
    - Clear typing and docstrings to guide future use.

- **2. Define core training abstractions in `eval_protocol/training/core.py`**
  - Define:
    - `EPConfig`:
      - A field for every parameter `evaluation_test` accepts (dataset, adapters, completion params, rollout processor, aggregation, thresholds, etc.).
      - Can be serialized/inspected for external tooling.
    - Candidate type alias (initially `Candidate = dict[str, str]`).
  - Implement:
    - `build_ep_config_from_test(test_fn: TestFunction) -> EPConfig`.
      - Reads `__ep_config__`.
      - Reuses the same dataset and rollout logic as pytest, but in a library-friendly way (no pytest invocation).
  - Helper(s) to:
    - Run a single candidate over the dataset, possibly with:
      - A subset of rows (train vs val split initially determined by the benchmark or EPConfig).
      - A configurable aggregation method (mean score to start).

- **3. Minimal tests and documentation for the core**
  - Add unit/integration tests that:
    - Use a tiny fake `@evaluation_test` function.
    - Confirm `build_ep_config_from_test` produces a config that can:
      - Load mock rows.
      - Run a dummy rollout processor.
      - Apply a simple metric to produce scores.
  - Document (in this design file or a short README) how benchmarks should think about exposing tunable pieces (e.g., via custom dataset adapters or other wiring).

#### Person B – GEPA Adapter & Benchmark Wiring

- **4. Implement GEPA integration in `eval_protocol/training/gepa_adapter.py`**
  - Define a small adapter API, e.g.:

```python
class GEPATrainer:
    def __init__(self, spec: trainingBenchmarkSpec, inject_fn: InjectFn, ...gepa_config...):
        ...

    def train(self) -> tuple[Candidate, Any]:
        """Run GEPA and return best candidate plus optional rich result."""
```

  - Inside, implement:
    - Conversion from `(spec, inject_fn)` into a GEPA metric:
      - For each candidate:
        - Clone or map the base dataset rows, applying `inject_fn(candidate, row)`.
        - Use the spec’s rollout runner + metric runner to compute per-example and aggregate scores.
        - Return the aggregate score (and optional textual feedback) to GEPA.
    - The call to `gepa.optimize(...)` with:
      - `seed_candidate` constructed from the baseline configuration (e.g., default system prompt).
      - Budget configuration (max metric calls / auto presets).
      - Reflection config (reflection LM or other knobs) passed in via constructor.
    - Mapping from `GEPAResult` (or equivalent) back into:
      - `best_candidate: Candidate`.
      - Optional rich result object (e.g., exposing Pareto-front stats).

- **5. Wire a first benchmark: AIME 2025**
  - In `eval_protocol/benchmarks/test_aime25.py`:
    - Factor the row-scoring logic inside `test_aime25_pointwise` into a **reusable metric function** (pure function that sets `row.evaluation_result` given a rolled-out row).
    - Decide how candidates should influence the evaluation:
      - For example, by making the dataset adapter or message-construction logic candidate-aware (e.g., changing the system prompt).
    - Add a `if __name__ == "__main__":` block that:
      - Imports `test_aime25_pointwise` and builds an `EPConfig` via `build_ep_config_from_test`.
      - Instantiates `GEPATrainer` with:
        - The `EPConfig`.
        - Initial GEPA config (budget, reflection model placeholder, seed).
      - Calls `trainer.train()` and prints/logs the resulting `best_candidate` for now.
    - Keep storage of tuned prompts as a TODO/extension point to be resolved later.

- **6. Optional second benchmark: GPQA**
  - Repeat step 5 for `test_gpqa.py`:
    - Identify what’s tunable (system prompt, possibly chain-of-thought instructions).
    - Extract metric logic into a reusable function.
    - Add candidate-aware wiring (e.g., via dataset adapters) and an optional `__main__` entrypoint calling the same GEPA trainer.
  - This will validate that:
    - The abstractions generalize across tasks.
    - No DSPy/GEPA-specific imports leak into benchmark files (other than a small, well-defined training API).

### Coordination Notes

- **Order of work**
  - Person A should go first (or in parallel up to the point where `EPConfig` and `build_ep_config_from_test` are usable).
  - Person B can stub against interfaces and adjust once Person A’s core is available.
- **Integration checkpoints**
  - After Person A lands the core + tests:
    - Person B wires AIME with a very simple “optimizer” (even random search) to smoke-test the path before hooking up real GEPA.
  - After GEPA integration works for AIME:
    - Decide on the canonical way to treat GEPA’s `run_dir` and/or additional artifacts for tuned prompts.
    - Optionally add a small helper that knows how to “run evaluation once with best GEPA candidate” for CI workflows.


future:

this is how gepa defines eval:

def metric(
    gold: Example,
    pred: Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | ScoreWithFeedback:
    """
    This function is called with the following arguments:
    - gold: The gold example.
    - pred: The predicted output.
    - trace: Optional. The trace of the program's execution.
    - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which
        the feedback is being requested.
    - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.

    Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
    feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
    and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
    If available at the predictor level, the metric should return {'score': float, 'feedback': str} corresponding
    to the predictor.
    If not available at the predictor level, the metric can also return a text feedback at the program level
    (using just the gold, pred and trace).
    If no feedback is returned, GEPA will use a simple text feedback consisting of just the score:
    f"This trajectory got a score of {score}."
    """
    ...

ideally generic way to turn evaluation_test into this.
