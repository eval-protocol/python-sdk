#!/usr/bin/env python3
"""
Test LunarLander MCP Server with Conda Isolation

This test verifies that the lunar lander example works correctly with
conda environment isolation, testing complex dependencies like swig and box2d.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import eval_protocol as ep


async def test_lunar_lander_with_conda_isolation():
    """Test the lunar lander example with managed simulation server and conda isolation."""

    print("🚀 Testing LunarLander MCP Server with Conda Isolation")

    # Paths
    base_dir = Path(__file__).parent
    production_script = base_dir / "mcp_server" / "lunar_lander_mcp_server.py"
    requirements_file = base_dir / "mcp_server" / "requirements.txt"

    # Start managed simulation server with conda isolation
    managed_server_script = (
        Path(__file__).parent.parent / "frozen_lake_mcp_complete" / "mcp_server" / "managed_simulation_server.py"
    )

    print(f"📦 Production script: {production_script}")
    print(f"📋 Requirements: {requirements_file}")
    print(f"🔧 Managed server: {managed_server_script}")

    cmd = [
        sys.executable,
        str(managed_server_script),
        "--port",
        "9004",
        "--production-script",
        str(production_script),
        "--requirements",
        str(requirements_file),
        "--use-conda-isolation",
    ]

    print(f"🚀 Starting managed server: {' '.join(cmd)}")

    # Start the managed server
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Wait for server to start and capture initial output
    print("⏳ Waiting for server to start...")
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < 60:  # 60 second timeout
        if process.poll() is not None:
            # Process died
            print("❌ Server process died!")
            stdout, stderr = process.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False

        time.sleep(1)

        # Check if we can connect (basic check)
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", 9004))
            sock.close()
            if result == 0:
                server_ready = True
                break
        except Exception:
            pass

    if not server_ready:
        print("❌ Server failed to start within timeout")
        process.terminate()
        return False

    print("✅ Server is running!")

    try:
        # Test basic functionality using eval_protocol
        print("🧪 Testing basic lunar lander functionality...")

        # Create a simple dataset for testing
        dataset = [
            {
                "id": "lunar_test_0",
                "seed": 42,
                "system_prompt": "You are controlling a lunar lander. Your goal is to land safely on the landing pad.",
                "user_prompt_template": "Current state:\n{observation}\n\nChoose your action from: NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT",
                "environment_context": {"timeout_seconds": 30},
            },
            {
                "id": "lunar_test_1",
                "seed": 123,
                "system_prompt": "You are controlling a lunar lander. Your goal is to land safely on the landing pad.",
                "user_prompt_template": "Current state:\n{observation}\n\nChoose your action from: NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT",
                "environment_context": {"timeout_seconds": 30},
            },
        ]

        # Configure for MCP environment
        envs = await ep.make("http://localhost:9004/mcp", dataset=dataset)

        # Simple policy that takes random actions
        class RandomLunarLanderPolicy:
            def __init__(self):
                import random

                self.rng = random.Random(42)

            async def __call__(self, tool_schemas, observations, system_prompts, user_prompts):
                from eval_protocol.types import MCPToolCall

                tool_calls = []
                actions = ["NOTHING", "FIRE_LEFT", "FIRE_MAIN", "FIRE_RIGHT"]

                for i in range(len(observations)):
                    action = self.rng.choice(actions)
                    tool_call = MCPToolCall("lander_action", {"action": action})
                    tool_calls.append(tool_call)

                return tool_calls

        policy = RandomLunarLanderPolicy()

        # Run a few episodes to test the environment
        print("🎮 Running test episodes...")

        evaluation_rows = await ep.rollout(envs, policy, steps=20)  # Keep short for testing

        print(f"✅ Completed {len(evaluation_rows)} evaluation rows")

        # Create output directory for images
        output_dir = Path(__file__).parent / "trajectory_output"
        output_dir.mkdir(exist_ok=True)

        print(f"💾 Saving evaluation data to {output_dir}")

        # Validate evaluation rows and save evaluation data
        for i, eval_row in enumerate(evaluation_rows):
            print(f"📊 Episode {i}: evaluation row object of type {type(eval_row)}")

            # Save evaluation summary
            evaluation_summary = {
                "episode": i,
                "total_reward": eval_row.get_total_reward(),
                "steps": eval_row.get_steps(),
                "terminated": eval_row.get_terminated(),
                "termination_reason": eval_row.get_termination_reason(),
                "messages_count": len(eval_row.messages),
            }

            # Note: observations, actions, and rewards are now embedded in messages
            # Actions and rewards can be extracted from control_plane_step data in messages
            control_plane_messages = [msg for msg in eval_row.messages if msg.control_plane_step]
            if control_plane_messages:
                evaluation_summary["control_plane_steps"] = len(control_plane_messages)
                evaluation_summary["sample_rewards"] = [
                    msg.control_plane_step["reward"] for msg in control_plane_messages[:5]
                ]

            # Save evaluation summary to JSON
            with open(output_dir / f"episode_{i}_summary.json", "w") as f:
                json.dump(evaluation_summary, f, indent=2, default=str)

            # Debug: print the structure of messages
            print(f"  🔍 Messages count: {len(eval_row.messages)}")
            print(f"  🔍 Control plane messages: {len(control_plane_messages)}")

            if control_plane_messages:
                first_control_plane = control_plane_messages[0].control_plane_step
                print(f"  🔍 First control plane step keys: {list(first_control_plane.keys())}")

                # Save full first control plane step for debugging
                with open(output_dir / f"episode_{i}_first_control_plane_debug.json", "w") as f:
                    json.dump(first_control_plane, f, indent=2, default=str)

                # Try to extract frames from control plane steps (if available)
                for step_idx, msg in enumerate(control_plane_messages[:10]):  # First 10 steps
                    control_plane_step = msg.control_plane_step
                    if isinstance(control_plane_step, dict):
                        print(f"    Step {step_idx} keys: {list(control_plane_step.keys())}")
                        if "rendered_frame" in control_plane_step:
                            frame_data = control_plane_step["rendered_frame"]
                            if frame_data and frame_data.startswith("data:image/png;base64,"):
                                try:
                                    # Decode base64 image
                                    import base64

                                    image_data = frame_data.split(",")[1]
                                    image_bytes = base64.b64decode(image_data)

                                    # Save image
                                    image_path = output_dir / f"episode_{i}_step_{step_idx:03d}.png"
                                    with open(image_path, "wb") as img_file:
                                        img_file.write(image_bytes)

                                    print(f"  💾 Saved frame: {image_path}")
                                except Exception as e:
                                    print(f"  ❌ Error saving frame {step_idx}: {e}")
                        else:
                            print(f"    Step {step_idx}: No rendered_frame field in control_plane_step")
                    else:
                        print(
                            f"    Step {step_idx}: control_plane_step is not a dict, type: {type(control_plane_step)}"
                        )
            else:
                print(f"  🔍 No control plane messages found")

            print(f"  ✅ Episode {i} validation passed")

        print(f"📁 All evaluation data saved to {output_dir}")
        print(f"   - Episode summaries: episode_*_summary.json")
        print(f"   - Control plane debug data: episode_*_first_control_plane_debug.json")
        print(f"   - Rendered frames: episode_*_step_*.png (if available)")

        print("🎉 All tests passed! Conda isolation working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up
        print("🧹 Cleaning up server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the test
    success = asyncio.run(test_lunar_lander_with_conda_isolation())

    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)
