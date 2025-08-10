"""
HuggingFace Dataset Adapter Example

This example demonstrates how to use the HuggingFace adapter to load datasets
from HuggingFace Hub and convert them to EvaluationRow format for evaluation.
"""

import os
from typing import List

from eval_protocol.adapters.huggingface import (
    HuggingFaceAdapter,
    create_gsm8k_adapter,
    create_huggingface_adapter,
    create_math_adapter,
)
from eval_protocol.models import EvaluationRow


def gsm8k_example():
    """Example using the GSM8K dataset."""
    print("📚 Example 1: GSM8K Dataset")
    print("-" * 30)

    try:
        # Create GSM8K adapter using the convenience method
        adapter = create_gsm8k_adapter(
            split="test", system_prompt="You are a helpful assistant that solves math problems step by step."
        )

        print("✅ GSM8K adapter created successfully")
        print(f"📊 Dataset info: {adapter.get_dataset_info()}")

        # Get a few evaluation rows
        rows = list(
            adapter.get_evaluation_rows(
                limit=3,
                model_name="gpt-4",
                temperature=0.0,
            )
        )

        print(f"\nRetrieved {len(rows)} evaluation rows from GSM8K test set:")

        for i, row in enumerate(rows):
            print(f"\n  Row {i+1}:")
            print(f"    - ID: {row.input_metadata.row_id if row.input_metadata else 'N/A'}")
            print(f"    - Messages: {len(row.messages)}")

            # Show the math problem
            user_message = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_message:
                problem = (
                    user_message.content[:200] + "..." if len(user_message.content) > 200 else user_message.content
                )
                print(f"    - Problem: {problem}")

            # Show ground truth answer
            if row.ground_truth:
                answer_preview = row.ground_truth[:100] + "..." if len(row.ground_truth) > 100 else row.ground_truth
                print(f"    - Ground truth: {answer_preview}")

            print()

    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install HuggingFace dependencies: pip install 'eval-protocol[huggingface]'")
    except Exception as e:
        print(f"❌ Error loading GSM8K: {e}")


def math_dataset_example():
    """Example using the MATH competition dataset."""
    print("🧮 Example 2: MATH Competition Dataset")
    print("-" * 40)

    try:
        # Create MATH dataset adapter
        adapter = create_math_adapter(system_prompt="You are an expert mathematician. Solve this step by step.")

        print("✅ MATH dataset adapter created successfully")
        print(f"📊 Dataset info: {adapter.get_dataset_info()}")

        # Get a few examples
        rows = list(
            adapter.get_evaluation_rows(
                limit=2,
                model_name="gpt-4",
                temperature=0.1,
            )
        )

        print(f"\nRetrieved {len(rows)} evaluation rows from MATH test set:")

        for i, row in enumerate(rows):
            print(f"\n  Row {i+1}:")

            # Show the problem
            user_message = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_message:
                problem = (
                    user_message.content[:150] + "..." if len(user_message.content) > 150 else user_message.content
                )
                print(f"    - Problem: {problem}")

            # Show metadata
            if row.input_metadata and row.input_metadata.dataset_info:
                dataset_info = row.input_metadata.dataset_info
                if "original_type" in dataset_info:
                    print(f"    - Problem type: {dataset_info['original_type']}")
                if "original_level" in dataset_info:
                    print(f"    - Level: {dataset_info['original_level']}")

    except Exception as e:
        print(f"❌ Error with MATH dataset: {e}")


def custom_dataset_example():
    """Example using a custom dataset with transformation function."""
    print("🔧 Example 3: Custom Dataset with Transform Function")
    print("-" * 55)

    try:
        # Define transformation function for SQuAD dataset
        def squad_transform(row):
            """Transform SQuAD row to evaluation format."""
            context = row["context"]
            question = row["question"]
            answers = row["answers"]

            # Get first answer text
            answer_text = answers["text"][0] if answers["text"] else "No answer provided"

            return {
                "messages": [
                    {"role": "system", "content": "Answer the question based on the given context."},
                    {"role": "user", "content": f"Context: {context}\\n\\nQuestion: {question}"},
                ],
                "ground_truth": answer_text,
                "metadata": {
                    "dataset": "squad",
                    "context_length": len(context),
                    "question_length": len(question),
                    "num_possible_answers": len(answers["text"]),
                },
            }

        # Create adapter with transformation function
        adapter = create_huggingface_adapter(
            dataset_id="squad",
            transform_fn=squad_transform,
        )

        print("✅ Custom dataset adapter created successfully")

        # Get dataset info
        info = adapter.get_dataset_info()
        print(f"📊 Dataset info: {info}")

        # Get a few examples
        rows = list(
            adapter.get_evaluation_rows(
                split="validation",  # SQuAD has train/validation splits
                limit=2,
                model_name="gpt-3.5-turbo",
            )
        )

        print(f"\nRetrieved {len(rows)} evaluation rows:")

        for i, row in enumerate(rows):
            print(f"\n  Row {i+1}:")
            print(f"    - Messages: {len(row.messages)}")

            # Show question
            user_message = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_message:
                question = (
                    user_message.content[:100] + "..." if len(user_message.content) > 100 else user_message.content
                )
                print(f"    - Question: {question}")

            # SQuAD answers are complex, so just show if we have ground truth
            print(f"    - Has ground truth: {'Yes' if row.ground_truth else 'No'}")

    except Exception as e:
        print(f"❌ Error with custom dataset: {e}")


def local_file_example():
    """Example loading a local dataset file."""
    print("📁 Example 4: Local Dataset File")
    print("-" * 35)

    # Create a sample JSONL file for demonstration
    sample_file = "/tmp/sample_qa.jsonl"
    sample_data = [
        {"id": "q1", "question": "What is the capital of France?", "answer": "Paris", "category": "geography"},
        {"id": "q2", "question": "What is 2 + 2?", "answer": "4", "category": "math"},
        {
            "id": "q3",
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "category": "literature",
        },
    ]

    try:
        import json

        # Write sample data
        with open(sample_file, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        print(f"📝 Created sample file: {sample_file}")

        # Define transformation function for local data
        def local_qa_transform(row):
            """Transform local Q&A data to evaluation format."""
            return {
                "messages": [
                    {"role": "system", "content": "You are a knowledgeable assistant."},
                    {"role": "user", "content": row["question"]},
                ],
                "ground_truth": row["answer"],
                "metadata": {
                    "id": row.get("id"),
                    "category": row.get("category"),
                    "dataset": "local_qa_sample",
                },
            }

        # Load with adapter
        adapter = HuggingFaceAdapter.from_local(
            path=sample_file,
            transform_fn=local_qa_transform,
        )

        print("✅ Local file adapter created successfully")

        # Get all rows
        rows = list(
            adapter.get_evaluation_rows(
                model_name="gpt-3.5-turbo",
                temperature=0.0,
            )
        )

        print(f"\nLoaded {len(rows)} rows from local file:")

        for i, row in enumerate(rows):
            print(f"\n  Row {i+1}:")

            # Show question and answer
            user_msg = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_msg:
                print(f"    - Question: {user_msg.content}")

            if row.ground_truth:
                print(f"    - Answer: {row.ground_truth}")

            # Show original metadata
            if row.input_metadata and row.input_metadata.dataset_info:
                original_data = {k: v for k, v in row.input_metadata.dataset_info.items() if k.startswith("original_")}
                if original_data:
                    print(f"    - Original data: {original_data}")

        # Clean up
        os.remove(sample_file)
        print(f"\n🧹 Cleaned up sample file")

    except Exception as e:
        print(f"❌ Error with local file: {e}")


def evaluation_integration_example():
    """Show how to integrate with evaluation functions."""
    print("\n🧪 Example 5: Integration with Evaluation")
    print("-" * 45)

    try:
        # Import evaluation functions
        from eval_protocol.rewards.accuracy import accuracy_reward
        from eval_protocol.rewards.math import math_reward

        # Create GSM8K adapter
        adapter = create_gsm8k_adapter(split="test")

        # Get a few rows for evaluation
        rows = list(adapter.get_evaluation_rows(limit=2))

        print(f"Running evaluation on {len(rows)} GSM8K problems:")

        for i, row in enumerate(rows):
            print(f"\n  Problem {i+1}:")

            # Show the problem
            user_msg = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_msg:
                print(f"    Question: {user_msg.content[:100]}...")

            # For this example, we'll simulate an assistant response
            # In practice, this would come from your LLM
            row.messages.append(
                {"role": "assistant", "content": "Let me solve this step by step... The answer is 42."}
            )

            # Evaluate with math reward
            if row.ground_truth:
                try:
                    math_result = math_reward(
                        messages=row.messages,
                        ground_truth=row.ground_truth,
                    )
                    print(f"    Math score: {math_result.score:.2f}")
                    print(f"    Reason: {math_result.reason[:100]}...")

                    # Also try accuracy reward
                    acc_result = accuracy_reward(
                        messages=row.messages,
                        ground_truth=row.ground_truth,
                    )
                    print(f"    Accuracy score: {acc_result.score:.2f}")

                except Exception as e:
                    print(f"    ❌ Evaluation error: {e}")

    except ImportError:
        print("Evaluation functions not available")
    except Exception as e:
        print(f"❌ Error in evaluation integration: {e}")


def batch_processing_example():
    """Show how to process datasets in batches."""
    print("\n📦 Example 6: Batch Processing")
    print("-" * 35)

    try:
        adapter = create_gsm8k_adapter(split="test")

        batch_size = 5
        total_processed = 0

        print(f"Processing GSM8K test set in batches of {batch_size}:")

        # Process in batches
        for batch_start in range(0, 20, batch_size):  # Process first 20 items
            batch_rows = list(
                adapter.get_evaluation_rows(
                    limit=batch_size,
                    offset=batch_start,
                )
            )

            print(f"  Batch {batch_start//batch_size + 1}: {len(batch_rows)} rows")

            # Process each row in the batch
            for row in batch_rows:
                # Here you would typically:
                # 1. Generate a response with your LLM
                # 2. Evaluate the response
                # 3. Store results
                total_processed += 1

        print(f"✅ Processed {total_processed} rows total")

    except Exception as e:
        print(f"❌ Error in batch processing: {e}")


def main():
    """Run all examples."""
    print("🤗 HuggingFace Dataset Adapter Examples")
    print("=" * 50)

    # Run examples
    gsm8k_example()
    print("\n" + "=" * 50 + "\n")

    math_dataset_example()
    print("\n" + "=" * 50 + "\n")

    custom_dataset_example()
    print("\n" + "=" * 50 + "\n")

    local_file_example()
    print("\n" + "=" * 50 + "\n")

    evaluation_integration_example()
    print("\n" + "=" * 50 + "\n")

    batch_processing_example()


if __name__ == "__main__":
    try:
        main()

        print("\n✅ All examples completed!")
        print("\nNext steps:")
        print("1. Choose the dataset that fits your needs")
        print("2. Customize the system prompts for your use case")
        print("3. Integrate with your evaluation pipeline")
        print("4. Scale up to process full datasets")
        print("5. Use the EvaluationRow data for training or evaluation")

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install 'eval-protocol[huggingface]'")
    except Exception as e:
        print(f"❌ Error running examples: {e}")
