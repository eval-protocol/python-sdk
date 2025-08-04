# Design Doc: N-Variant Generation for Offline Rollouts

Based on my review of the code, I have come up with a design that I believe will achieve your goal. Here is a summary of my findings and the proposed design.

### Code Analysis Summary

1.  **Entry Point**: The command `eval-protocol run` is the entry point for the evaluation pipeline.
2.  **Configuration**: The pipeline is configured using Hydra, with the main configuration loaded from a YAML file.
3.  **Core Logic**: The core logic for the evaluation pipeline is in the `EvaluationPipeline` class in `eval_protocol/execution/pipeline.py`.
4.  **Single Sample Processing**: The `_process_single_sample` method in the `EvaluationPipeline` class is responsible for processing a single sample from the dataset. This includes generating a response from a model if `generation.enabled` is true.
5.  **Generation**: The `self.model_client.generate` method is called to generate a single response for a given sample.

### Proposed Design

I propose the following changes to implement the N-variant generation feature:

1.  **Introduce a New Configuration Parameter**: I'll add a new parameter, `n`, to the `generation` section of the Hydra configuration. This will allow you to specify how many variants to generate for each sample. For example:

    ```yaml
    generation:
      enabled: true
      model_name: "accounts/fireworks/models/firellava-13b"
      n: 5 # Generate 5 variants for each sample
    ```

2.  **Modify the Evaluation Pipeline**: I will modify the `EvaluationPipeline` to handle the `n` parameter.
    *   The `_process_single_sample` method will be updated to loop `n` times.
    *   In each iteration, it will call the model to generate a new response.
    *   Each generated response will be stored as a separate record with a shared `id` (from the original sample) and a new `response_index` field (from 0 to n-1).
    *   The `run` method will be updated to handle the multiple results generated for each sample and save them to the output file.

### Design Visualization

Here is a Mermaid diagram illustrating the proposed changes to the pipeline:

```mermaid
graph TD
    subgraph "eval_protocol/cli.py"
        A[eval-protocol run] --> B{parse_args};
        B --> C{hydra_cli_entry_point};
    end

    subgraph "eval_protocol/cli_commands/run_eval_cmd.py"
        C --> D[run_evaluation_command_logic];
        D --> E[pipeline = EvaluationPipeline(cfg)];
        E --> F[pipeline.run()];
    end

    subgraph "eval_protocol/execution/pipeline.py"
        F --> G[EvaluationPipeline.run];
        G -- for each sample --> H[EvaluationPipeline._process_single_sample];
        H -- reads cfg.generation.n --> I{Loop n times};
        I -- each iteration --> J[model_client.generate];
        J --> K[Create result dict with response_index];
        K --> L[Append to list of results];
        I -- after loop --> M[Return list of n results];
        M --> G;
        G -- flattens list of lists --> N[all_results];
        N --> O[Save results to file];
    end

    subgraph "Configuration (YAML)"
        P[your_config.yaml] -- defines --> Q[generation.n];
        Q -- loaded into --> E;
    end

    style F fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#ccf,stroke:#333,stroke-width:2px
    style M fill:#ccf,stroke:#333,stroke-width:2px
    style Q fill:#cfc,stroke:#333,stroke-width:2px
```

This design will allow you to generate multiple variants for each sample in your dataset by simply adding the `n` parameter to your configuration file.
