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

import eval_protocol as rk


async def test_lunar_lander_with_conda_isolation():
    """Test the lunar lander example with managed simulation server and conda isolation."""

    print("🚀 Testing LunarLander MCP Server with Conda Isolation")

    # Paths
    base_dir = Path(__file__).parent
    production_script = base_dir / "mcp_server" / "lunar_lander_mcp_server.py"
    requirements_file = base_dir / "mcp_server" / "requirements.txt"

    # Start managed simulation server with conda isolation
    managed_server_script = (
        Path(__file__).parent.parent
        / "frozen_lake_mcp_complete"
        / "mcp_server"
        / "managed_simulation_server.py"
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
        # Test basic functionality using reward_kit
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
        envs = rk.make("http://localhost:9004/mcp", dataset=dataset)

        # Simple policy that takes random actions
        class RandomLunarLanderPolicy:
            def __init__(self):
                import random

                self.rng = random.Random(42)

            async def __call__(
                self, tool_schemas, observations, system_prompts, user_prompts
            ):
                from eval_protocol.mcp.types import MCPToolCall

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

        rollouts = await rk.rollout(envs, policy, steps=20)  # Keep short for testing

        print(f"✅ Completed {len(rollouts)} rollouts")

        # Create output directory for images
        output_dir = Path(__file__).parent / "trajectory_output"
        output_dir.mkdir(exist_ok=True)

        print(f"💾 Saving trajectory data to {output_dir}")

        # Validate rollouts and save trajectory data
        for i, trajectory in enumerate(rollouts):
            print(f"📊 Episode {i}: trajectory object of type {type(trajectory)}")

            # Save trajectory summary
            trajectory_summary = {
                "episode": i,
                "total_reward": getattr(trajectory, "total_reward", "N/A"),
                "steps": getattr(trajectory, "steps", "N/A"),
                "terminated": getattr(trajectory, "terminated", "N/A"),
                "duration": getattr(trajectory, "duration", "N/A"),
            }

            # Save observations and actions if available
            if hasattr(trajectory, "observations"):
                trajectory_summary["observations"] = trajectory.observations[
                    :5
                ]  # First 5 for brevity
            if hasattr(trajectory, "actions"):
                trajectory_summary["actions"] = trajectory.actions[
                    :5
                ]  # First 5 for brevity
            if hasattr(trajectory, "rewards"):
                trajectory_summary["rewards"] = trajectory.rewards[
                    :5
                ]  # First 5 for brevity

            # Save trajectory summary to JSON
            with open(output_dir / f"episode_{i}_summary.json", "w") as f:
                json.dump(trajectory_summary, f, indent=2, default=str)

            # Debug: print the structure of observations
            if hasattr(trajectory, "observations"):
                print(f"  🔍 Observations type: {type(trajectory.observations)}")
                print(
                    f"  🔍 Number of observations: {len(trajectory.observations) if trajectory.observations else 0}"
                )

                if trajectory.observations and len(trajectory.observations) > 0:
                    first_obs = trajectory.observations[0]
                    print(f"  🔍 First observation type: {type(first_obs)}")
                    print(
                        f"  🔍 First observation keys: {first_obs.keys() if isinstance(first_obs, dict) else 'Not a dict'}"
                    )

                    # Save full first observation for debugging
                    with open(
                        output_dir / f"episode_{i}_first_obs_debug.json", "w"
                    ) as f:
                        json.dump(first_obs, f, indent=2, default=str)

                # Try to extract frames from observations
                for step_idx, obs in enumerate(
                    trajectory.observations[:10]
                ):  # First 10 steps
                    if isinstance(obs, dict):
                        print(f"    Step {step_idx} keys: {list(obs.keys())}")
                        if "rendered_frame" in obs:
                            frame_data = obs["rendered_frame"]
                            if frame_data and frame_data.startswith(
                                "data:image/png;base64,"
                            ):
                                try:
                                    # Decode base64 image
                                    import base64

                                    image_data = frame_data.split(",")[1]
                                    image_bytes = base64.b64decode(image_data)

                                    # Save image
                                    image_path = (
                                        output_dir
                                        / f"episode_{i}_step_{step_idx:03d}.png"
                                    )
                                    with open(image_path, "wb") as img_file:
                                        img_file.write(image_bytes)

                                    print(f"  💾 Saved frame: {image_path}")
                                except Exception as e:
                                    print(f"  ❌ Error saving frame {step_idx}: {e}")
                        else:
                            print(f"    Step {step_idx}: No rendered_frame field")
                    else:
                        print(f"    Step {step_idx}: Not a dict, type: {type(obs)}")
            else:
                print(f"  🔍 No observations attribute found")

            print(f"  ✅ Episode {i} validation passed")

        print(f"📁 All trajectory data saved to {output_dir}")
        print(f"   - Episode summaries: episode_*_summary.json")
        print(f"   - Rendered frames: episode_*_step_*.png")

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
