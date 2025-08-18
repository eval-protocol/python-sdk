import json
import os

from datasets import load_dataset


def create_sample_jsonl():
    """
    Creates a sample JSONL file from the GSM8K dataset for testing purposes.
    The output file will be development/gsm8k_sample.jsonl.
    """
    try:
        # Load the GSM8K dataset, 'main' config, 'test' split
        dataset = load_dataset("gsm8k", name="main", split="test", trust_remote_code=True)
        print(f"Successfully loaded GSM8K test set. It has {len(dataset)} samples.")
    except Exception as e:
        print(f"Failed to load GSM8K dataset: {e}")
        return

    output_dir = "development"
    output_filename = "gsm8k_sample.jsonl"
    output_filepath = os.path.join(output_dir, output_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    num_samples_to_take = 5
    samples_written = 0

    print(f"Attempting to write {num_samples_to_take} samples to {output_filepath}...")

    with open(output_filepath, "w", encoding="utf-8") as f:
        for i in range(min(num_samples_to_take, len(dataset))):
            sample = dataset[i]
            question_content = sample.get("question")
            answer_content = sample.get("answer")  # This contains reasoning and final answer

            if question_content is None or answer_content is None:
                print(f"Skipping sample {i} due to missing 'question' or 'answer' field.")
                continue

            # For this sample, we use the 'answer' field (which includes reasoning)
            # Standardized prompt format
            record_id = f"gsm8k_test_{i}"
            record = {
                "id": record_id,
                "user_query": question_content,
                "ground_truth_for_eval": answer_content,  # Full answer string for math_reward
            }
            f.write(json.dumps(record) + "\n")
            samples_written += 1

    if samples_written > 0:
        print(f"Successfully wrote {samples_written} samples to {output_filepath}")
    else:
        print("No samples were written. Check dataset loading and content.")


if __name__ == "__main__":
    create_sample_jsonl()
