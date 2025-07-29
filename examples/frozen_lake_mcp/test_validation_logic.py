#!/usr/bin/env python3
"""
Test our hardened validation logic against the existing trajectory file.
This should catch the bugs we've identified.
"""

import json
import sys
from pathlib import Path

from _pytest.outcomes import Failed

# Add the test directory to the path so we can import our validation functions
sys.path.insert(0, str(Path(__file__).parent / "tests"))

import importlib.util

# Import from the local test file explicitly
import sys

spec = importlib.util.spec_from_file_location(
    "test_frozen_lake_e2e_local",
    Path(__file__).parent / "tests" / "test_frozen_lake_e2e.py",
)
local_test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_test_module)

_validate_control_plane_sync = local_test_module._validate_control_plane_sync
_validate_no_repeated_states = local_test_module._validate_no_repeated_states
_validate_trajectory_termination = local_test_module._validate_trajectory_termination


def test_validation_with_existing_data():
    """Test our validation logic with the existing multi_env_trajectory.jsonl file."""

    recording_file = Path(__file__).parent / "tests/recordings/multi_env_trajectory.jsonl"

    if not recording_file.exists():
        print(f"❌ Recording file not found: {recording_file}")
        return False

    print(f"🔍 Testing validation logic with: {recording_file}")

    # Load the recorded data
    recorded_entries = []
    with open(recording_file, "r") as f:
        for line in f:
            if line.strip():
                recorded_entries.append(json.loads(line))

    print(f"📊 Loaded {len(recorded_entries)} recorded entries")

    # Group by environment
    env_recordings = {}
    for entry in recorded_entries:
        env_idx = entry["env_index"]
        if env_idx not in env_recordings:
            env_recordings[env_idx] = []
        env_recordings[env_idx].append(entry)

    print(f"📊 Found recordings for environments: {list(env_recordings.keys())}")

    # Test our validation logic
    dataset = [{"id": f"test_{i}"} for i in range(len(env_recordings))]  # Dummy dataset

    print("\n🔍 Testing repeated states validation...")
    try:
        _validate_no_repeated_states(env_recordings, dataset)
        print("✅ Repeated states validation passed - no issues detected")
        repeated_states_ok = True
    except (Exception, Failed) as e:
        print(f"❌ Repeated states validation failed (as expected): {str(e).split(':')[0]}...")
        repeated_states_ok = False

    print("\n🔍 Testing control plane sync validation...")
    try:
        _validate_control_plane_sync(env_recordings, dataset)
        print("✅ Control plane sync validation passed - no issues detected")
        control_plane_ok = True
    except (Exception, Failed) as e:
        print(f"❌ Control plane sync validation failed (as expected): {str(e).split(':')[0]}...")
        control_plane_ok = False

    print("\n🔍 Testing trajectory termination validation...")
    try:
        _validate_trajectory_termination(env_recordings, dataset)
        print("✅ Trajectory termination validation passed - no issues detected")
        trajectory_termination_ok = True
    except (Exception, Failed) as e:
        print(f"❌ Trajectory termination validation failed (as expected): {str(e).split(':')[0]}...")
        trajectory_termination_ok = False

    # Summary
    if repeated_states_ok and control_plane_ok and trajectory_termination_ok:
        print("\n✅ All validations passed - no bugs detected")
        return True
    else:
        print(f"\n❌ Validation caught bugs (as expected):")
        print(f"  - Repeated states bug: {'No' if repeated_states_ok else 'Yes'}")
        print(f"  - Control plane sync bug: {'No' if control_plane_ok else 'Yes'}")
        print(f"  - Trajectory termination bug: {'No' if trajectory_termination_ok else 'Yes'}")
        return False


if __name__ == "__main__":
    test_validation_with_existing_data()
