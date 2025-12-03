# Tinker GSM8K Training Example

This example demonstrates how to use `eval_protocol` to fetch GSM8K data and train a model using `tinker`'s RL training loop.

## Prerequisites

1. **Tinker Cookbook**: Ensure `tinker-cookbook` is available. The script attempts to add `../../../tinker-cookbook` to `sys.path`.
2. **Eval Protocol**: Ensure `eval-protocol` is installed with HuggingFace support.
   ```bash
   pip install 'eval-protocol[huggingface]'
   ```
3. **Tinker API Key**: You need a Tinker API key.
   ```bash
   export TINKER_API_KEY=your_api_key_here
   ```

## Running the Training

Run the training script with python. We recommend using a small model for testing, such as `Qwen/Qwen3-4B-Instruct-2507`.

```bash
# Install dependencies
pip install 'eval-protocol[huggingface]' chz tinker

# Run training
export TINKER_API_KEY=your_api_key_here
python train.py model_name="Qwen/Qwen3-4B-Instruct-2507" groups_per_batch=4 train_limit=100 test_limit=10
```

### Configuration Options

- `model_name`: The model to train (e.g., `Qwen/Qwen3-4B-Instruct-2507`).
- `groups_per_batch`: Batch size (default: 100).
- `group_size`: Number of samples per problem (default: 4).
- `train_limit`: Number of training examples to fetch (default: 1000).
- `test_limit`: Number of test examples to fetch (default: 100).
- `log_path`: Path to save logs and checkpoints.

## How it Works

1. **Data Loading**: The script uses `eval_protocol.adapters.huggingface.create_gsm8k_adapter` to fetch GSM8K data.
2. **Dataset Adaptation**: `EvalProtocolGsm8kDataset` converts `EvaluationRow` objects from `eval_protocol` into `ProblemGroupBuilder` objects expected by `tinker`.
3. **Training**: The standard `tinker` training loop is used to optimize the model.
