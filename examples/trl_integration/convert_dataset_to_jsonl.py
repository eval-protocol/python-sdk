import argparse
import json


def convert_raw_to_transformed_jsonl(input_file_path, output_file_path):
    """
    Converts a JSONL dataset from the "raw" format to the "transformed" format.

    The "raw" format is expected to have:
    - "prompt": a list of chat messages (e.g., [{"role": "user", "content": "..."}])
    - "reward_model": {"ground_truth": "[{\"input\": \"...\", \"expected_output\": \"...\"}, ...]"}

    The "transformed" format will have:
    - "prompt": a string (the user's content from the first user message)
    - "test_cases": a list of parsed {"input": "...", "expected_output": "..."} objects
    """
    try:
        with (
            open(input_file_path, "r", encoding="utf-8") as infile,
            open(output_file_path, "w", encoding="utf-8") as outfile,
        ):
            for line_number, line in enumerate(infile, 1):
                try:
                    raw_data = json.loads(line.strip())

                    # Extract user prompt
                    user_prompt_content = None
                    if isinstance(raw_data.get("prompt"), list):
                        for message in raw_data["prompt"]:
                            if message.get("role") == "user" and "content" in message:
                                user_prompt_content = message["content"]
                                break

                    if user_prompt_content is None:
                        print(f"Warning: Could not find user prompt in line {line_number}. Skipping.")
                        continue

                    # Extract and parse ground_truth
                    ground_truth_str = raw_data.get("reward_model", {}).get("ground_truth")
                    if ground_truth_str is None:
                        print(f"Warning: Could not find ground_truth in line {line_number}. Skipping.")
                        continue

                    try:
                        test_cases = json.loads(ground_truth_str)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse ground_truth JSON in line {line_number}: {e}. Skipping.")
                        continue

                    transformed_data = {
                        "prompt": user_prompt_content,
                        "test_cases": test_cases,
                    }

                    outfile.write(json.dumps(transformed_data) + "\n")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line {line_number}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_number}: {e}")

        print(f"Successfully converted {input_file_path} to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw TRL dataset to transformed JSONL format.")
    parser.add_argument("input_file", help="Path to the input raw JSONL file.")
    parser.add_argument("output_file", help="Path to the output transformed JSONL file.")

    args = parser.parse_args()

    convert_raw_to_transformed_jsonl(args.input_file, args.output_file)

# Example Usage:
# python examples/trl_integration/convert_dataset_to_jsonl.py \
#   examples/trl_integration/data/simulated_deepcoder_raw_sample.jsonl \
#   examples/trl_integration/data/simulated_deepcoder_transformed_sample.jsonl
