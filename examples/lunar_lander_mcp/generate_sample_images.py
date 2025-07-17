#!/usr/bin/env python3
"""
Generate sample lunar lander trajectory images for demonstration.

This creates a simple trajectory showing the lunar lander in action,
saving each frame as a PNG image to visualize the environment.
"""

import base64
import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "mcp_server"))

from lunar_lander_adapter import LunarLanderAdapter


def generate_sample_trajectory():
    """Generate a sample trajectory with saved frames."""

    print("🚀 Generating sample lunar lander trajectory")

    # Create output directory
    output_dir = Path("examples/lunar_lander_mcp/sample_trajectory")
    output_dir.mkdir(exist_ok=True)

    # Create adapter and environment
    adapter = LunarLanderAdapter()
    env = adapter.create_environment({})

    # Reset environment with seed for reproducibility
    obs, info = adapter.reset_environment(env, seed=42)

    print(f"🎮 Initial observation: {obs}")
    print(f"📊 Environment info: {info}")

    # Save initial frame
    initial_frame = adapter.render_frame(env)
    if initial_frame:
        save_frame(initial_frame, output_dir / "step_000_initial.png", 0, "INITIAL")

    # Define a sequence of actions to demonstrate different behaviors
    actions = [
        "NOTHING",  # Let it fall naturally
        "NOTHING",  # Continue falling
        "FIRE_MAIN",  # Fire main engine to slow descent
        "FIRE_LEFT",  # Adjust orientation
        "FIRE_MAIN",  # More main engine
        "FIRE_RIGHT",  # Adjust other direction
        "FIRE_MAIN",  # Try to slow down
        "NOTHING",  # Coast
        "FIRE_MAIN",  # Final attempt
        "NOTHING",  # See what happens
    ]

    trajectory_data = []

    for step, action in enumerate(actions, 1):
        print(f"\n🎮 Step {step}: {action}")

        # Parse and execute action
        action_int = adapter.parse_action(action)
        obs, reward, terminated, truncated, info = adapter.step_environment(
            env, action_int
        )

        # Format observation
        formatted_obs = adapter.format_observation(obs)

        # Get status
        status = adapter.get_landing_status(obs, reward, terminated, truncated)

        # Render frame
        frame = adapter.render_frame(env)

        # Save frame
        if frame:
            filename = f"step_{step:03d}_{action.lower()}.png"
            save_frame(frame, output_dir / filename, step, action)

        # Save step data
        step_data = {
            "step": step,
            "action": action,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "status": status,
            "observation": formatted_obs,
            "has_frame": frame is not None,
        }

        trajectory_data.append(step_data)

        print(f"  📊 Reward: {reward:.3f}")
        print(f"  📊 Status: {status}")
        print(f"  📊 Position: x={obs[0]:.3f}, y={obs[1]:.3f}")
        print(f"  📊 Velocity: vx={obs[2]:.3f}, vy={obs[3]:.3f}")
        print(f"  🖼️  Frame saved: {frame is not None}")

        if terminated or truncated:
            print(f"🏁 Episode ended: {status}")
            break

    # Save trajectory summary
    import json

    with open(output_dir / "trajectory_summary.json", "w") as f:
        json.dump(
            {
                "seed": 42,
                "total_steps": len(trajectory_data),
                "final_status": (
                    trajectory_data[-1]["status"] if trajectory_data else "Unknown"
                ),
                "total_reward": sum(step["reward"] for step in trajectory_data),
                "steps": trajectory_data,
            },
            f,
            indent=2,
        )

    print(f"\n📁 Trajectory saved to {output_dir}")
    print(f"   📊 {len(trajectory_data)} steps recorded")
    print(f"   🖼️  {len(list(output_dir.glob('*.png')))} images saved")
    print(f"   📋 Summary: trajectory_summary.json")

    env.close()
    return output_dir


def save_frame(frame_data: str, output_path: Path, step: int, action: str):
    """Save a base64 frame to a PNG file."""
    try:
        if not frame_data.startswith("data:image/png;base64,"):
            print(f"  ❌ Invalid frame format for step {step}")
            return False

        # Extract base64 data
        image_data = frame_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Save to file
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        print(f"  💾 Saved: {output_path}")
        return True

    except Exception as e:
        print(f"  ❌ Error saving frame {step}: {e}")
        return False


if __name__ == "__main__":
    try:
        output_dir = generate_sample_trajectory()
        print(f"\n✅ Sample trajectory generated successfully!")
        print(f"📁 View images in: {output_dir.absolute()}")

    except Exception as e:
        print(f"❌ Error generating trajectory: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
