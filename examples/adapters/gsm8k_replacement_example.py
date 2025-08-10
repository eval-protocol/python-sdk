"""
GSM8K Replacement Example

This example shows how to replace the static GSM8K JSONL file
(development/gsm8k_sample.jsonl) with the dynamic HuggingFace adapter
to get fresh data from the GSM8K dataset.
"""

import json
from pathlib import Path
from typing import List

from eval_protocol.adapters.huggingface import create_gsm8k_adapter
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.rewards.math import math_reward


def load_original_gsm8k_sample() -> List[dict]:
    """Load the original GSM8K sample file for comparison."""
    sample_file = Path("development/gsm8k_sample.jsonl")

    if not sample_file.exists():
        print(f"⚠️ Original sample file not found: {sample_file}")
        return []

    data = []
    with open(sample_file, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def demonstrate_old_vs_new_approach():
    """Compare the old static file approach with the new adapter approach."""
    print("📊 Comparing Old vs New Approach")
    print("=" * 50)

    # OLD APPROACH: Static JSONL file
    print("🗂️  OLD APPROACH: Static JSONL File")
    print("-" * 35)

    original_data = load_original_gsm8k_sample()
    print(f"Loaded {len(original_data)} items from static file")

    if original_data:
        sample = original_data[0]
        print(f"Sample item fields: {list(sample.keys())}")
        print(f"Sample question: {sample.get('user_query', '')[:100]}...")
        print(f"Sample ground truth: {sample.get('ground_truth_for_eval', '')[:100]}...")

    print("\n" + "=" * 50 + "\n")

    # NEW APPROACH: HuggingFace Adapter
    print("🤗 NEW APPROACH: HuggingFace Adapter")
    print("-" * 38)

    try:
        # Create adapter
        adapter = create_gsm8k_adapter(
            system_prompt="You are a helpful assistant that solves math problems step by step."
        )

        print("✅ GSM8K adapter created successfully")

        # Get the same number of items as the original file
        num_items = len(original_data) if original_data else 6
        rows = list(adapter.get_evaluation_rows(limit=num_items))

        print(f"Retrieved {len(rows)} evaluation rows from HuggingFace")

        if rows:
            sample_row = rows[0]
            print(f"Sample EvaluationRow fields: messages, tools, input_metadata, ground_truth")

            # Show the question from messages
            user_msg = next((msg for msg in sample_row.messages if msg.role == "user"), None)
            if user_msg:
                print(f"Sample question: {user_msg.content[:100]}...")

            if sample_row.ground_truth:
                print(f"Sample ground truth: {sample_row.ground_truth[:100]}...")

    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install HuggingFace dependencies: pip install 'eval-protocol[huggingface]'")
        return
    except Exception as e:
        print(f"❌ Error with adapter: {e}")
        return

    print("\n" + "=" * 50 + "\n")

    # COMPARISON
    print("🔍 Key Differences")
    print("-" * 20)
    print("OLD APPROACH:")
    print("  ✅ Fast loading (local file)")
    print("  ❌ Static data (same 6 problems)")
    print("  ❌ Manual data preparation required")
    print("  ❌ Limited to pre-selected subset")
    print("  ❌ Requires manual format conversion")

    print("\nNEW APPROACH:")
    print("  ✅ Access to full GSM8K dataset (8,792 test problems)")
    print("  ✅ Automatic format conversion to EvaluationRow")
    print("  ✅ Built-in metadata handling")
    print("  ✅ Configurable system prompts")
    print("  ✅ Consistent with other adapters")
    print("  ⚠️  Requires internet connection and HuggingFace datasets")


def show_migration_example():
    """Show how to migrate existing code from JSONL to adapter."""
    print("\n🔄 Code Migration Example")
    print("=" * 30)

    print("OLD CODE:")
    print("-" * 10)
    print(
        """
# Old way with static JSONL file
input_dataset = ["development/gsm8k_sample.jsonl"]

# Manual loading and parsing
import json
data = []
with open("development/gsm8k_sample.jsonl", 'r') as f:
    for line in f:
        item = json.loads(line)
        # Manual conversion to expected format
        messages = [
            {"role": "user", "content": item["user_query"]}
        ]
        ground_truth = item["ground_truth_for_eval"]
        # ... more manual processing
"""
    )

    print("\nNEW CODE:")
    print("-" * 10)
    print(
        """
# New way with HuggingFace adapter
from eval_protocol.adapters.huggingface import create_gsm8k_adapter

# Create adapter with custom configuration
adapter = create_gsm8k_adapter(
    system_prompt="You are a helpful math tutor."
)

# Get evaluation rows (already in correct format)
evaluation_rows = list(adapter.get_evaluation_rows(
    split="test",  # or "train"
    limit=100,  # Can get much more data than static file
    model_name="gpt-4",
    temperature=0.0,
))

# evaluation_rows are already EvaluationRow objects!
# No manual conversion needed

# For complete control, use the generic adapter:
def custom_gsm8k_transform(row):
    return {
        'messages': [
            {'role': 'system', 'content': 'Custom system prompt here'},
            {'role': 'user', 'content': row['question']}
        ],
        'ground_truth': row['answer'],
        'metadata': {'custom_field': 'custom_value'}
    }

from eval_protocol.adapters.huggingface import create_huggingface_adapter
custom_adapter = create_huggingface_adapter(
    dataset_id="gsm8k",
    config_name="main",
    transform_fn=custom_gsm8k_transform
)
"""
    )

    print("\n✅ Benefits of Migration:")
    print("  - More data available (6 → 8,792 problems)")
    print("  - Automatic format handling")
    print("  - Better metadata preservation")
    print("  - Consistent API across all adapters")
    print("  - Easy to customize system prompts")


def practical_migration_demo():
    """Show a practical example of using the adapter in evaluation."""
    print("\n🧪 Practical Evaluation Example")
    print("=" * 35)

    try:
        # Create adapter
        adapter = create_gsm8k_adapter()

        # Get a few problems for evaluation
        print("Loading GSM8K problems...")
        rows = list(adapter.get_evaluation_rows(limit=3))
        print(f"✅ Loaded {len(rows)} problems from GSM8K test set")

        # Simulate evaluation workflow
        for i, row in enumerate(rows):
            print(f"\n📝 Problem {i+1}:")

            # Show the problem
            user_msg = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_msg:
                print(f"   Question: {user_msg.content[:150]}...")

            # In a real scenario, you'd generate a response with your LLM
            # For this demo, we'll add a dummy response
            dummy_response = "Let me solve this step by step. After working through the math, the answer is 42."
            row.messages.append(Message(role="assistant", content=dummy_response))

            # Evaluate with math reward function
            if row.ground_truth:
                try:
                    result = math_reward(
                        messages=row.messages,
                        ground_truth=row.ground_truth,
                    )
                    print(f"   📊 Math evaluation score: {result.score:.2f}")
                    print(f"   💭 Evaluation reason: {result.reason[:100]}...")

                    # Show metadata
                    if row.input_metadata:
                        print(f"   🏷️  Row ID: {row.input_metadata.row_id}")
                        if row.input_metadata.dataset_info:
                            dataset_info = row.input_metadata.dataset_info
                            print(f"   📚 Dataset: {dataset_info.get('dataset_name', 'N/A')}")
                            print(f"   📍 Row index: {dataset_info.get('row_index', 'N/A')}")

                except Exception as e:
                    print(f"   ❌ Evaluation error: {e}")

        print(f"\n✅ Successfully processed {len(rows)} problems using the new adapter approach!")

    except Exception as e:
        print(f"❌ Error in practical demo: {e}")


def performance_comparison():
    """Compare performance characteristics of both approaches."""
    print("\n⚡ Performance Considerations")
    print("=" * 35)

    import time

    # Time the old approach (if file exists)
    original_data = load_original_gsm8k_sample()
    if original_data:
        start_time = time.time()
        # Simulate processing the original data
        processed_old = len(original_data)
        old_time = time.time() - start_time
        print(f"📁 Static file approach: {processed_old} items in {old_time:.4f}s")
    else:
        print("📁 Static file not available for timing")
        old_time = 0
        processed_old = 0

    # Time the new approach
    try:
        start_time = time.time()
        adapter = create_gsm8k_adapter()
        rows = list(adapter.get_evaluation_rows(split="test", limit=max(6, processed_old)))
        new_time = time.time() - start_time
        processed_new = len(rows)

        print(f"🤗 HuggingFace adapter: {processed_new} items in {new_time:.4f}s")

        if old_time > 0:
            if new_time > old_time:
                factor = new_time / old_time
                print(f"   📊 Adapter is {factor:.1f}x slower (but loads fresh data)")
            else:
                factor = old_time / new_time
                print(f"   📊 Adapter is {factor:.1f}x faster!")

        print(f"\n💡 Trade-offs:")
        print(f"   Static file: Fast ({old_time:.4f}s) but limited data ({processed_old} items)")
        print(f"   Adapter: Slower ({new_time:.4f}s) but access to full dataset ({processed_new}+ items)")

    except Exception as e:
        print(f"❌ Error timing adapter: {e}")


def main():
    """Run the complete GSM8K replacement demonstration."""
    print("🚀 GSM8K Replacement Example")
    print("=" * 50)
    print("This example shows how to replace the static GSM8K JSONL file")
    print("with the dynamic HuggingFace adapter for better data access.")
    print()

    # Run all demonstrations
    demonstrate_old_vs_new_approach()
    show_migration_example()
    practical_migration_demo()
    performance_comparison()

    print("\n" + "=" * 50)
    print("🎯 MIGRATION SUMMARY")
    print("=" * 50)
    print("1. ✅ Replace static JSONL with HuggingFace adapter")
    print("2. ✅ Get access to full GSM8K dataset (8,792 test problems)")
    print("3. ✅ Automatic conversion to EvaluationRow format")
    print("4. ✅ Built-in metadata and system prompt support")
    print("5. ✅ Consistent API with other data sources")
    print()
    print("📝 Next Steps:")
    print("- Update your evaluation scripts to use the adapter")
    print("- Experiment with different system prompts")
    print("- Scale up to use more than 6 problems")
    print("- Consider using train split for different use cases")
    print("- Integrate with your existing evaluation pipeline")


if __name__ == "__main__":
    main()
