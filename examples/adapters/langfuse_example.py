"""
Langfuse Adapter Example

This example demonstrates how to use the Langfuse adapter to pull data from
a Langfuse deployment and convert it to EvaluationRow format for evaluation.
"""

import os
from datetime import datetime, timedelta
from typing import List

from eval_protocol.adapters.langfuse import create_langfuse_adapter
from eval_protocol.models import EvaluationRow


def main():
    """Example usage of the Langfuse adapter."""

    # Configuration - you can set these as environment variables
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "your_public_key_here")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "your_secret_key_here")
    host = os.getenv("LANGFUSE_HOST", "https://langfuse-web-prod-zfdbl7ykrq-uc.a.run.app")
    project_id = os.getenv("LANGFUSE_PROJECT_ID", "cmdj5yxhk0006s6022cyi0prv")

    print(f"Connecting to Langfuse at: {host}")
    print(f"Project ID: {project_id}\n")

    # Create the adapter
    try:
        adapter = create_langfuse_adapter(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            project_id=project_id,
        )
        print("✅ Langfuse adapter created successfully")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install Langfuse dependencies: pip install 'eval-protocol[langfuse]'")
        return
    except Exception as e:
        print(f"❌ Failed to create adapter: {e}")
        return

    # Example 1: Get recent evaluation rows
    print("\n📊 Example 1: Get recent evaluation rows")
    try:
        rows = list(
            adapter.get_evaluation_rows(
                limit=5,
                from_timestamp=datetime.now() - timedelta(days=7),
                include_tool_calls=True,
            )
        )

        print(f"Retrieved {len(rows)} evaluation rows")
        for i, row in enumerate(rows):
            print(f"  Row {i+1}:")
            print(f"    - ID: {row.input_metadata.row_id if row.input_metadata else 'N/A'}")
            print(f"    - Messages: {len(row.messages)}")
            print(f"    - Has tools: {'Yes' if row.tools else 'No'}")
            print(f"    - Ground truth: {'Yes' if row.ground_truth else 'No'}")

            # Show first message content (truncated)
            if row.messages:
                content = row.messages[0].content or ""
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"    - First message: {preview}")
            print()

    except Exception as e:
        print(f"❌ Error retrieving rows: {e}")

    # Example 2: Filter by specific criteria
    print("\n🔍 Example 2: Filter by specific criteria")
    try:
        rows = list(
            adapter.get_evaluation_rows(
                limit=3,
                tags=["production"],  # Filter by tags if available
                include_tool_calls=True,
            )
        )

        print(f"Retrieved {len(rows)} rows with 'production' tag")

    except Exception as e:
        print(f"❌ Error with filtered query: {e}")

    # Example 3: Get specific traces by ID
    print("\n🎯 Example 3: Get specific traces by ID")
    try:
        # Replace with actual trace IDs from your Langfuse deployment
        trace_ids = ["trace_id_1", "trace_id_2"]  # These would be real IDs

        rows = list(
            adapter.get_evaluation_rows_by_ids(
                trace_ids=trace_ids,
                include_tool_calls=True,
            )
        )

        print(f"Retrieved {len(rows)} rows by specific IDs")

    except Exception as e:
        print(f"❌ Error retrieving specific traces: {e}")

    # Example 4: Extract different types of conversations
    print("\n💬 Example 4: Analyze conversation types")
    try:
        rows = list(adapter.get_evaluation_rows(limit=10, include_tool_calls=True))

        chat_only = []
        tool_calling = []

        for row in rows:
            if row.tools and any(
                msg.tool_calls for msg in row.messages if hasattr(msg, "tool_calls") and msg.tool_calls
            ):
                tool_calling.append(row)
            else:
                chat_only.append(row)

        print(f"Chat-only conversations: {len(chat_only)}")
        print(f"Tool calling conversations: {len(tool_calling)}")

        # Show example of tool calling conversation
        if tool_calling:
            row = tool_calling[0]
            print(f"\n🔧 Example tool calling conversation:")
            for i, msg in enumerate(row.messages):
                print(f"  {i+1}. {msg.role}: {msg.content[:50] if msg.content else '[No content]'}...")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(f"     🛠 Tool call: {tool_call}")

    except Exception as e:
        print(f"❌ Error analyzing conversation types: {e}")


def demonstrate_evaluation_integration():
    """Show how to use Langfuse data with evaluation functions."""
    print("\n🧪 Integration with Evaluation Functions")

    # This would typically be in a separate evaluation script
    try:
        from eval_protocol.rewards.math import math_reward

        # Create adapter (reuse configuration from main example)
        adapter = create_langfuse_adapter(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "your_public_key_here"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", "your_secret_key_here"),
            host=os.getenv("LANGFUSE_HOST", "https://langfuse-web-prod-zfdbl7ykrq-uc.a.run.app"),
            project_id=os.getenv("LANGFUSE_PROJECT_ID", "cmdj5yxhk0006s6022cyi0prv"),
        )

        # Get data and evaluate
        rows = list(adapter.get_evaluation_rows(limit=3))

        for i, row in enumerate(rows):
            print(f"\nEvaluating row {i+1}:")

            # Only evaluate if we have ground truth
            if row.ground_truth:
                try:
                    result = math_reward(
                        messages=row.messages,
                        ground_truth=row.ground_truth,
                    )
                    print(f"  Math evaluation score: {result.score:.2f}")
                    print(f"  Reason: {result.reason}")
                except Exception as e:
                    print(f"  ❌ Evaluation failed: {e}")
            else:
                print(f"  ⚠️ No ground truth available for evaluation")

    except ImportError:
        print("Math reward function not available")
    except Exception as e:
        print(f"❌ Error in evaluation integration: {e}")


if __name__ == "__main__":
    print("🚀 Langfuse Adapter Example")
    print("=" * 50)

    # Check if credentials are set
    if not all(
        [
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
        ]
    ):
        print("⚠️  To run this example with real data, set environment variables:")
        print("   export LANGFUSE_PUBLIC_KEY='your_public_key'")
        print("   export LANGFUSE_SECRET_KEY='your_secret_key'")
        print("   export LANGFUSE_HOST='your_langfuse_host'  # optional")
        print("   export LANGFUSE_PROJECT_ID='your_project_id'  # optional")
        print()

    main()
    demonstrate_evaluation_integration()

    print("\n✅ Example completed!")
    print("\nNext steps:")
    print("1. Set up your Langfuse credentials")
    print("2. Modify the filters and parameters to match your data")
    print("3. Integrate with your evaluation pipeline")
    print("4. Use the converted EvaluationRow data for training or evaluation")
