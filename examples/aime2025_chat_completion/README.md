## AIME2025 Chat Completion Example

This example reproduces gpt-oss's AIME2025 chat completion evaluation inside Eval Protocol.

### What it does
- Loads AIME2025 questions from Hugging Face
- Prompts a reasoning-capable chat-completions model
- Extracts the final integer answer from \boxed{...}
- Scores exact-match vs. the ground-truth integer

### Quick run (pytest, CI-friendly)
The evaluation is implemented as a pytest `evaluation_test` under `tests/`. Run it directly:

```bash
pytest -q examples/aime2025_chat_completion/tests/test_evaluation.py -q
```

Environment variables expected:
- `FIREWORKS_API_KEY`

To scale up, adjust parameters in the decorator (e.g., `threshold_of_success`, `max_dataset_rows`).



