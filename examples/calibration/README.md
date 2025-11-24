# Calibration Evaluator

This directory contains an evaluator for measuring the calibration of LLM classifiers.
It calculates:
- **Accuracy**: Fraction of correct predictions.
- **Brier Score**: Mean squared error of the probabilities. Lower is better.
- **ECE (Expected Calibration Error)**: Weighted average of the difference between confidence and accuracy in bins. Lower is better.

## Usage

1. Install dependencies:
   ```bash
   pip install datasets numpy openai python-dotenv
   ```

2. Set your Fireworks API key in `.env` or environment variables:
   ```bash
   export FIREWORKS_API_KEY=your_key
   ```

3. Run the evaluation script:
   ```bash
   python run_calibration.py
   ```

## Files

- `evaluator.py`: Contains the `calibration_evaluator` batch reward function.
- `run_calibration.py`: Script to load AG News dataset and run the evaluation on specified models.

## Configuration

You can modify `run_calibration.py` to:
- Change the models being evaluated (`MODELS` list).
- Change the dataset or number of samples.
- Adjust the class mapping if using a different dataset.

You can modify `evaluator.py` to:
- Change the class tokens (`CLASS_TOKENS`) if the model uses different tokenization.
- Adjust `top_logprobs` if needed (note that some models limit this to 5).
