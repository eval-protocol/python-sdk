#!/usr/bin/env python3
"""
GitHub Actions rollout worker script.

This script is called by the GitHub Actions workflow to perform the actual rollout.
It makes an OpenAI completion call and saves the full conversation trace as JSON.
"""

import argparse
import base64
import json
import os

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="GitHub Actions rollout worker")

    # Required arguments from workflow inputs
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--rollout-id", required=True, help="Rollout ID for tracking")
    parser.add_argument("--messages-b64", required=True, help="Base64 encoded JSON messages")
    parser.add_argument("--tools-b64", required=False, help="Base64 encoded JSON tools (optional)")

    args = parser.parse_args()

    print(f"🚀 Starting rollout {args.rollout_id}")
    print(f"   Model: {args.model}")

    # Decode messages and tools
    try:
        messages = json.loads(base64.b64decode(args.messages_b64).decode("utf-8"))
        tools = None
        if args.tools_b64:
            tools = json.loads(base64.b64decode(args.tools_b64).decode("utf-8"))
    except Exception as e:
        print(f"❌ Failed to decode inputs: {e}")
        # Save error trace
        error_data = {
            "status": "error",
            "rollout_id": args.rollout_id,
            "model": args.model,
            "messages": [],
            "error": f"Failed to decode inputs: {e}",
        }
        with open(f"rollout_trace_{args.rollout_id}.json", "w") as f:
            json.dump(error_data, f, indent=2)
        exit(1)

    print(f"   Messages: {len(messages)} messages")
    print(f"   Tools: {len(tools) if tools else 0} tools")

    # Perform the rollout
    conversation = messages.copy()

    try:
        completion_kwargs = {"model": args.model, "messages": messages}
        if tools:
            completion_kwargs["tools"] = tools

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
            "tools": tools,
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
            "tools": tools,
            "error": str(e),
        }

    # Save trace to file
    output_file = f"rollout_trace_{args.rollout_id}.json"
    with open(output_file, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"💾 Saved trace to {output_file}")


if __name__ == "__main__":
    main()
