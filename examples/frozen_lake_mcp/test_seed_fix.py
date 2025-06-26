#!/usr/bin/env python3
"""
Test script to verify that FrozenLake seeding works correctly.

This tests both the dataset seed extraction and the adapter's seeded environment creation.
"""

import json
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from frozen_lake_adapter import FrozenLakeAdapter

from reward_kit.datasets.loader import load_dataset_from_config


def test_dataset_loading():
    """Test that seeds are properly loaded from the dataset."""
    print("🧪 Testing dataset loading...")

    # Load rollouts dataset
    config = {"source": "rollouts.jsonl"}
    dataset = load_dataset_from_config(config)

    print(f"📊 Dataset loaded: {len(dataset)} entries")

    # Check first few entries for seeds
    for i, entry in enumerate(dataset[:3]):
        seed = entry.get("seed")
        prompt = entry.get("prompt", "No prompt")[:50] + "..."
        print(f"  Entry {i}: seed={seed}, prompt='{prompt}'")

    seeds = [entry.get("seed") for entry in dataset if entry.get("seed") is not None]
    print(f"✅ Found {len(seeds)} entries with seeds: {seeds}")
    return seeds


def test_adapter_seeded_creation():
    """Test that the adapter's create_environment_with_seed produces deterministic results."""
    print("\n🧪 Testing adapter seeded environment creation...")

    adapter = FrozenLakeAdapter()

    # Test that the same seed produces the same environment
    seed = 42
    print(f"🎯 Testing deterministic creation with seed={seed}")

    # Create environment twice with same seed
    env1, obs1, info1 = adapter.create_environment_with_seed(seed=seed)
    env2, obs2, info2 = adapter.create_environment_with_seed(seed=seed)

    # Compare the generated maps
    map1 = env1.desc
    map2 = env2.desc

    print(f"  Environment 1 map shape: {map1.shape}")
    print(f"  Environment 2 map shape: {map2.shape}")
    print(f"  Initial observations: {obs1}, {obs2}")

    # Check if maps are identical
    maps_identical = (map1 == map2).all()
    obs_identical = obs1 == obs2

    print(f"  Maps identical: {maps_identical}")
    print(f"  Initial observations identical: {obs_identical}")

    if maps_identical and obs_identical:
        print("✅ Seeded environment creation is deterministic!")
    else:
        print("❌ Seeded environment creation is not deterministic!")

        # Show first few rows of each map for debugging
        print("  Map 1 (first 3 rows):")
        for row in map1[:3]:
            print(f"    {row}")
        print("  Map 2 (first 3 rows):")
        for row in map2[:3]:
            print(f"    {row}")

    # Clean up
    adapter.close_environment(env1)
    adapter.close_environment(env2)

    return maps_identical and obs_identical


def test_different_seeds():
    """Test that different seeds produce different environments."""
    print("\n🧪 Testing that different seeds produce different environments...")

    adapter = FrozenLakeAdapter()

    # Create environments with different seeds
    env1, obs1, info1 = adapter.create_environment_with_seed(seed=42)
    env2, obs2, info2 = adapter.create_environment_with_seed(seed=123)

    map1 = env1.desc
    map2 = env2.desc

    maps_different = not (map1 == map2).all()

    print(f"  Seed 42 map shape: {map1.shape}")
    print(f"  Seed 123 map shape: {map2.shape}")
    print(f"  Maps are different: {maps_different}")

    if maps_different:
        print("✅ Different seeds produce different environments!")
    else:
        print("❌ Different seeds produce identical environments!")
        print("  This might indicate a seeding issue.")

    # Clean up
    adapter.close_environment(env1)
    adapter.close_environment(env2)

    return maps_different


def main():
    """Run all tests."""
    print("🚀 FrozenLake Seed Fix Verification")
    print("=" * 50)

    try:
        # Test 1: Dataset loading
        seeds_from_dataset = test_dataset_loading()

        # Test 2: Adapter deterministic creation
        deterministic_creation = test_adapter_seeded_creation()

        # Test 3: Different seeds produce different environments
        different_seeds_work = test_different_seeds()

        print("\n📋 Test Summary:")
        print(f"  Dataset loading: {'✅ PASS' if seeds_from_dataset else '❌ FAIL'}")
        print(
            f"  Deterministic creation: {'✅ PASS' if deterministic_creation else '❌ FAIL'}"
        )
        print(
            f"  Different seeds work: {'✅ PASS' if different_seeds_work else '❌ FAIL'}"
        )

        if all([seeds_from_dataset, deterministic_creation, different_seeds_work]):
            print("\n🎉 All tests passed! Seed handling should work correctly.")
            return 0
        else:
            print("\n⚠️ Some tests failed. Check the implementation.")
            return 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
