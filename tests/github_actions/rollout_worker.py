#!/usr/bin/env python3
"""
GitHub Actions rollout worker script.

This script is called by the GitHub Actions workflow to perform the actual rollout.
It makes an OpenAI completion call and saves the full conversation trace as JSON.
"""

import argparse
import json
import os

from openai import OpenAI
from eval_protocol.types.remote_rollout_processor import InitRequest


def main():
    parser = argparse.ArgumentParser(description="GitHub Actions rollout worker")

    # Required arguments from workflow inputs
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--metadata", required=True, help="JSON serialized metadata object")
    parser.add_argument("--messages", required=True, help="JSON serialized messages array")
    parser.add_argument("--tools", required=False, help="JSON serialized tools array")
    parser.add_argument("--model-base-url", required=True, help="Base URL for the model API")

    args = parser.parse_args()

    # Parse the JSON inputs
    try:
        metadata = json.loads(args.metadata)
        messages = json.loads(args.messages)
        tools = json.loads(args.tools) if args.tools else None
    except Exception as e:
        print(f"❌ Failed to parse JSON inputs: {e}")
        exit(1)

    rollout_id = metadata["rollout_id"]
    print(f"🚀 Starting rollout {rollout_id}")
    print(f"   Model: {args.model}")
    print(f"   Messages: {len(messages)} messages")

    # Perform the rollout
    conversation = messages.copy()

    try:
        completion_kwargs = {"model": args.model, "messages": messages}
        if tools:
            completion_kwargs["tools"] = tools

        client = OpenAI(base_url=args.model_base_url, api_key=os.environ.get("FIREWORKS_API_KEY"))

        print("📡 Calling OpenAI completion...")
        completion = client.chat.completions.create(**completion_kwargs)
        print("✅ Received response")

        # Add assistant response to conversation
        if completion.choices and completion.choices[0].message:
            assistant_message = completion.choices[0].message.model_dump()
            conversation.append(assistant_message)

        print(f"✅ Rollout {rollout_id} completed successfully")

    except Exception as e:
        print(f"❌ Error in rollout {rollout_id}: {e}")


if __name__ == "__main__":
    main()
