"""
BigQuery Adapter Example

This example demonstrates how to use the BigQuery adapter to query data from
Google BigQuery and convert it to EvaluationRow format for evaluation.
"""

import os
from typing import Any, Dict

from eval_protocol.adapters.bigquery import create_bigquery_adapter


def react_documentation_example():
    """Example usage of the BigQuery adapter with react documentation data."""

    # Configuration - you can set these as environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Optional

    table_id = ...
    input_column = "question"
    output_column = "response"

    print(f"📊 BigQuery Project: {project_id}")
    print(f"📋 Table: {table_id}")
    print(f"🔧 Authentication: {'Service Account' if credentials_path else 'Default credentials'}")
    print()

    # Transform function for react documentation data
    def react_transform(row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform react documentation data to evaluation format."""
        question = str(row.get(input_column, ""))
        response = str(row.get(output_column, ""))

        system_prompt = """You are an expert developer assistant specializing in frontend coding."""

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "ground_truth": response,
        }

    # Create the adapter
    try:
        adapter = create_bigquery_adapter(
            transform_fn=react_transform,
            dataset_id=project_id,
            credentials_path=credentials_path,
        )
        print("✅ BigQuery adapter created successfully")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install BigQuery dependencies: pip install 'eval-protocol[bigquery]'")
        return
    except Exception as e:
        print(f"❌ Failed to create adapter: {e}")
        print("Make sure you have valid Google Cloud credentials")
        return

    # Example 1: Get recent evaluation rows
    print("\n📚 Example 1: Get react documentation Q&A pairs")
    try:
        query = f"""
        SELECT {input_column}, {output_column}
        FROM `{table_id}`
        ORDER BY RAND()
        LIMIT 5
        """

        rows = list(
            adapter.get_evaluation_rows(
                query=query,
                limit=3,
                model_name="gpt-4",
                temperature=0.0,
            )
        )

        print(f"Retrieved {len(rows)} evaluation rows")
        for i, row in enumerate(rows):
            print(f"\n  📄 Example {i+1}:")
            print(f"    - ID: {row.input_metadata.row_id if row.input_metadata else 'N/A'}")
            print(f"    - Messages: {len(row.messages)}")

            # Show question
            user_msg = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_msg:
                question = user_msg.content[:120] + "..." if len(user_msg.content) > 120 else user_msg.content
                print(f"    - Question: {question}")

            # Show answer
            if row.ground_truth:
                answer = row.ground_truth[:120] + "..." if len(row.ground_truth) > 120 else row.ground_truth
                print(f"    - Answer: {answer}")

    except Exception as e:
        print(f"❌ Error retrieving rows: {e}")
        print("Make sure the table exists and you have access to it")

    # Example 2: Filtered query with specific criteria
    print("\n🔍 Example 2: Filter by content length")
    try:
        query = f"""
        SELECT {input_column}, {output_column}
        FROM `{table_id}`
        WHERE LENGTH({input_column}) > 50
          AND LENGTH({output_column}) > 100
        ORDER BY RAND()
        LIMIT 10
        """

        rows = list(
            adapter.get_evaluation_rows(
                query=query,
                limit=2,
                model_name="gpt-4",
                temperature=0.1,
            )
        )

        print(f"Retrieved {len(rows)} filtered rows with content length > 50 and > 100")

        for i, row in enumerate(rows):
            print(f"\n  🔍 Filtered Row {i+1}:")
            print(f"    - ID: {row.input_metadata.row_id if row.input_metadata else 'N/A'}")
            print(f"    - Messages: {len(row.messages)}")

            # Show question
            user_msg = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_msg:
                question = user_msg.content[:120] + "..." if len(user_msg.content) > 120 else user_msg.content
                print(f"    - Question: {question}")
                print(f"    - Question length: {len(user_msg.content)} chars")

            # Show answer
            if row.ground_truth:
                answer = row.ground_truth[:120] + "..." if len(row.ground_truth) > 120 else row.ground_truth
                print(f"    - Answer: {answer}")
                print(f"    - Answer length: {len(row.ground_truth)} chars")

    except Exception as e:
        print(f"❌ Error with filtered query: {e}")

    # Example 3: Parameterized query for security
    print("\n🎯 Example 3: Secure parameterized query")
    try:
        from google.cloud import bigquery

        query = f"""
        SELECT {input_column}, {output_column}
        FROM `{table_id}`
        WHERE LENGTH({input_column}) >= @min_question_length
          AND LENGTH({output_column}) >= @min_answer_length
        ORDER BY RAND()
        LIMIT @max_rows
        """

        query_params = [
            bigquery.ScalarQueryParameter("min_question_length", "INT64", 30),
            bigquery.ScalarQueryParameter("min_answer_length", "INT64", 50),
            bigquery.ScalarQueryParameter("max_rows", "INT64", 3),
        ]

        rows = list(
            adapter.get_evaluation_rows(
                query=query,
                query_params=query_params,
                limit=2,
            )
        )

        print(f"Retrieved {len(rows)} rows using parameterized query")
        print("🔒 Query used secure parameters to prevent injection attacks")

        for i, row in enumerate(rows):
            print(f"\n  🎯 Parameterized Row {i+1}:")
            print(f"    - ID: {row.input_metadata.row_id if row.input_metadata else 'N/A'}")
            print(f"    - Messages: {len(row.messages)}")

            # Show question
            user_msg = next((msg for msg in row.messages if msg.role == "user"), None)
            if user_msg:
                question = user_msg.content[:120] + "..." if len(user_msg.content) > 120 else user_msg.content
                print(f"    - Question: {question}")

            # Show answer
            if row.ground_truth:
                answer = row.ground_truth[:120] + "..." if len(row.ground_truth) > 120 else row.ground_truth
                print(f"    - Answer: {answer}")

    except ImportError:
        print("❌ BigQuery client not available for parameterized queries")
    except Exception as e:
        print(f"❌ Error with parameterized query: {e}")


if __name__ == "__main__":
    """Run the BigQuery adapter example."""
    print("📊 BigQuery Adapter Example")
    print("=" * 50)

    # Check if credentials are configured
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        print("⚠️  To run this example with real data, set environment variables:")
        print("   export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account.json'  # optional")
        print()
        print("   Or use: gcloud auth application-default login")
        print()

    react_documentation_example()
