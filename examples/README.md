# Eval Protocol Examples: Developer Workflow

This guide provides a concise developer workflow for understanding, running, and creating examples within the Eval Protocol. For a comprehensive illustration of these principles, always refer to `examples/math_example/`.

## Core Workflow for an Example

The typical lifecycle of working with or developing an example involves these key stages:

**1. Dataset Configuration (`conf/dataset/`)**

*   **Goal**: Prepare data in the precise format your reward function expects.
*   **Key Pattern (`math_example`)**:
    *   **Base Dataset (`gsm8k.yaml`)**: Defines connection to raw data (e.g., HuggingFace) and initial column mappings.
    *   **Derived Dataset (`gsm8k_math_prompts.yaml`)**: References a base dataset. Critically, this is where you:
        *   Inject specific **system prompts**.
        *   Perform **transformations** to structure the input (e.g., creating a full user query from multiple fields).
        *   Ensure the output format matches what the reward function requires.
*   **Reproducibility**: This explicit, versioned dataset configuration (especially the derived dataset) is fundamental for reproducible evaluations. The reward function is tied to the output of this specific data preparation pipeline.
*   **Detailed Guide**: For a comprehensive explanation of all YAML fields used in dataset configurations, see the [Dataset Configuration Guide](../docs/dataset_configuration_guide.md).

**2. Reward Function (e.g., `main.py`)**

*   **Goal**: Implement the logic to evaluate model responses against the prepared dataset.
*   **Connection**: The reward function (often in an example's `main.py` or a dedicated metrics script) is designed to consume data precisely as formatted by its corresponding **derived dataset configuration**.
*   **Example (`math_example/main.py`)**: Contains an `evaluate()` function tailored to the GSM8K problems as prompted and formatted by `gsm8k_math_prompts.yaml`.

**3. Local Run (Evaluation or Training)**

*   **Goal**: Execute your example to generate model responses and evaluate them, or to run a training loop.
*   **Tools**:
    *   **CLI-based Evaluation**: `python -m eval_protocol.cli run --config-path examples/math_example/conf --config-name run_math_eval.yaml`
        *   Uses Hydra for configuration (see `run_math_eval.yaml`).
        *   Generates outputs like `math_example_results.jsonl` and `preview_input_output_pairs.jsonl`.
    *   **Custom Scripts (e.g., TRL Training)**: `.venv/bin/python examples/math_example/trl_grpo_integration.py`
        *   Also typically uses Hydra for configuration (see `math_example/conf/trl_grpo_config.yaml`).
*   **Configuration**: Parameters for model selection, sampling, and reward function behavior are managed via Hydra YAML files in the example's `conf/` directory.

**4. Previewing Results (`eval-protocol preview`)**

*   **Goal**: Inspect or re-evaluate generated prompt/response pairs, often using different or updated reward logic.
*   **Usage**:
    ```bash
    # Preview with local metric scripts or a deployed evaluator
    python -m eval_protocol.cli preview \
      --samples ./outputs/<timestamp_dir>/preview_input_output_pairs.jsonl \
      --metrics-folders custom_metric_name=./path/to/your_metrics_folder
    # or --remote-url <your_deployed_evaluator_url>
    ```
*   **Input**: Uses files like `preview_input_output_pairs.jsonl` generated during a local run.

**5. Deployment (Optional for Examples, see `CONTRIBUTING.md`)**

*   While individual examples focus on local execution and preview, the general process for deploying evaluators is covered in `development/CONTRIBUTING.md` using `eval-protocol deploy`. Example-specific READMEs might also touch on this if relevant.

## Key Principles for Examples

*   **Clarity & Conciseness**: The example-specific `README.md` should clearly explain its purpose and how to run it.
*   **Configuration-Driven**: Use Hydra (via `conf/` directory) for all parameters.
*   **Reproducibility**: Ensure the link between the derived dataset configuration (with its specific prompting/formatting) and the reward function is clear and robust.
*   **`math_example` as Gold Standard**: Refer to it for structure, documentation, and best practices.

## Finding More Details

*   **Example-Specific READMEs**: Each example directory (e.g., `examples/math_example/README.md`) contains detailed instructions for that specific example.
*   **Contribution Guidelines**: For broader development practices, coding standards, and deployment, see `development/CONTRIBUTING.md`.

## Contributing New Examples

1.  Follow the workflow and principles outlined above.
2.  Model your structure and documentation after `examples/math_example/`.
3.  Ensure your example has its own clear `README.md` and necessary `conf/` files.
4.  Test thoroughly.
