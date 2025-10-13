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


def main():
    parser = argparse.ArgumentParser(description="GitHub Actions rollout worker")

    # Required arguments from workflow inputs
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--rollout-id", required=True, help="Rollout ID for tracking")
    parser.add_argument("--prompt", required=True, help="User prompt for the rollout")

    args = parser.parse_args()

    print(f"🚀 Starting rollout {args.rollout_id}")
    print(f"   Model: {args.model}")
    print(f"   Prompt: {args.prompt}")

    # Build messages array
    messages = [{"role": "user", "content": args.prompt}]

    print(f"   Messages: {len(messages)} messages")

    # Perform the rollout
    conversation = messages.copy()

    try:
        completion_kwargs = {"model": args.model, "messages": messages}

        client = OpenAI(api_key=os.environ.get("FIREWORKS_API_KEY"))

        print("📡 Calling OpenAI completion...")
        completion = client.chat.completions.create(**completion_kwargs)
        print("✅ Received response")

        # Add assistant response to conversation
        if completion.choices and completion.choices[0].message:
            assistant_message = completion.choices[0].message.model_dump()
            conversation.append(assistant_message)

        # Save successful trace
        trace_data = {
            "status": "success",
            "rollout_id": args.rollout_id,
            "model": args.model,
            "messages": conversation,
            "usage": completion.usage.model_dump() if completion.usage else None,
        }

        print(f"✅ Rollout {args.rollout_id} completed successfully")

    except Exception as e:
        print(f"❌ Error in rollout {args.rollout_id}: {e}")

        # Save error trace
        trace_data = {
            "status": "error",
            "rollout_id": args.rollout_id,
            "model": args.model,
            "messages": conversation,
            "error": str(e),
        }

    # Save trace to file
    output_file = f"rollout_trace_{args.rollout_id}.json"
    with open(output_file, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"💾 Saved trace to {output_file}")


if __name__ == "__main__":
    main()
