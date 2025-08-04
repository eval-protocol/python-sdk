#!/usr/bin/env python3
"""
Simple test to capture lunar lander trajectory with images.

This directly tests the MCP server to capture tool responses with rendered frames.
"""

import asyncio
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def test_lunar_lander_direct():
    """Test lunar lander MCP server directly to capture frames."""

    print("🚀 Testing LunarLander MCP Server Direct Communication")

    # Start the lunar lander server directly
    cmd = [
        sys.executable,
        "mcp_server/lunar_lander_mcp_server.py",
        "--port",
        "8006",
        "--seed",
        "42",
    ]

    print(f"🚀 Starting server: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Wait for server to start
    time.sleep(3)

    if process.poll() is not None:
        print("❌ Server failed to start")
        stdout, stderr = process.communicate()
        print(f"Output: {stdout}")
        return False

    print("✅ Server started")

    # Create output directory
    output_dir = Path("trajectory_output_direct")
    output_dir.mkdir(exist_ok=True)

    try:
        # Connect to the MCP server
        async with streamablehttp_client("http://localhost:8006") as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("🔗 Connected to MCP server")

                # Get initial state
                resources = await session.list_resources()
                print(f"📋 Available resources: {[r.name for r in resources]}")

                # Get initial state resource
                initial_state = await session.read_resource("game://initial_state")
                print(f"🎮 Initial state: {initial_state.contents[0].text[:100]}...")

                # Parse initial state to get rendered frame
                try:
                    initial_data = json.loads(initial_state.contents[0].text)
                    if "rendered_frame" in initial_data:
                        # Save initial frame
                        frame_data = initial_data["rendered_frame"]
                        if frame_data.startswith("data:image/png;base64,"):
                            image_data = frame_data.split(",")[1]
                            image_bytes = base64.b64decode(image_data)
                            with open(output_dir / "initial_frame.png", "wb") as f:
                                f.write(image_bytes)
                            print("💾 Saved initial frame")
                except Exception as e:
                    print(f"⚠️  Could not save initial frame: {e}")

                # Get available tools
                tools = await session.list_tools()
                print(f"🔧 Available tools: {[t.name for t in tools]}")

                # Perform actions and capture frames
                actions = ["NOTHING", "FIRE_MAIN", "FIRE_LEFT", "FIRE_RIGHT", "NOTHING"]

                for i, action in enumerate(actions):
                    print(f"🎮 Step {i+1}: {action}")

                    # Call lander_action tool
                    result = await session.call_tool("lander_action", {"action": action})

                    print(f"📊 Tool result type: {type(result.content)}")

                    if result.content:
                        for content in result.content:
                            if hasattr(content, "text"):
                                try:
                                    response_data = json.loads(content.text)
                                    print(f"📊 Response keys: {list(response_data.keys())}")

                                    # Save step summary
                                    step_summary = {
                                        "step": i + 1,
                                        "action": action,
                                        "reward": response_data.get("reward", 0),
                                        "terminated": response_data.get("terminated", False),
                                        "status": response_data.get("status", "Unknown"),
                                    }

                                    with open(output_dir / f"step_{i+1:03d}_summary.json", "w") as f:
                                        json.dump(step_summary, f, indent=2)

                                    # Save rendered frame if available
                                    if "rendered_frame" in response_data:
                                        frame_data = response_data["rendered_frame"]
                                        if frame_data and frame_data.startswith("data:image/png;base64,"):
                                            image_data = frame_data.split(",")[1]
                                            image_bytes = base64.b64decode(image_data)

                                            frame_path = output_dir / f"step_{i+1:03d}_{action.lower()}.png"
                                            with open(frame_path, "wb") as f:
                                                f.write(image_bytes)
                                            print(f"  💾 Saved frame: {frame_path}")
                                        else:
                                            print(f"  ⚠️  No rendered frame in response")
                                    else:
                                        print(f"  ⚠️  No rendered_frame field in response")

                                except json.JSONDecodeError as e:
                                    print(f"  ❌ Could not parse response as JSON: {e}")
                                    print(f"  Raw content: {content.text[:200]}...")

                print(f"📁 All data saved to {output_dir}")
                return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up server
        print("🧹 Cleaning up server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    success = asyncio.run(test_lunar_lander_direct())

    if success:
        print("✅ Direct test passed!")
        print("📁 Check trajectory_output_direct/ for images and data")
    else:
        print("❌ Direct test failed!")

    sys.exit(0 if success else 1)
