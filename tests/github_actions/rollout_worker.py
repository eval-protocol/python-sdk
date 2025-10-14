#!/usr/bin/env python3
"""
GitHub Actions rollout worker script.

This script is called by the GitHub Actions workflow to perform the actual rollout.
It makes an OpenAI completion call that gets automatically traced via the tracing proxy.
"""

import argparse
import json
import os

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="GitHub Actions rollout worker")

    # Required arguments from workflow inputs
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--metadata", required=True, help="JSON serialized metadata object")
    parser.add_argument("--model-base-url", required=True, help="Base URL for the model API")

    args = parser.parse_args()

    # Parse the metadata
    try:
        metadata = json.loads(args.metadata)
    except Exception as e:
        print(f"❌ Failed to parse metadata: {e}")
        exit(1)

    rollout_id = metadata["rollout_id"]
    row_id = metadata["row_id"]

    print(f"🚀 Starting rollout {rollout_id}")
    print(f"   Model: {args.model}")
    print(f"   Row ID: {row_id}")

    dataset = [  # In this example, worker has access to the dataset and we use index to associate rows.
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ]

    user_content = dataset[int(row_id)]
    messages = [{"role": "user", "content": user_content}]

    print(f"   Messages: {len(messages)} messages")

    try:
        completion_kwargs = {"model": args.model, "messages": messages}

        client = OpenAI(base_url=args.model_base_url, api_key=os.environ.get("FIREWORKS_API_KEY"))

        print("📡 Calling OpenAI completion...")
        completion = client.chat.completions.create(**completion_kwargs)

        print(f"✅ Rollout {rollout_id} completed successfully")

    except Exception as e:
        print(f"❌ Error in rollout {rollout_id}: {e}")


if __name__ == "__main__":
    main()
