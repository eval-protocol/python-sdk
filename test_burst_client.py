#!/usr/bin/env python3
"""
Burst Client Test - Simulates 50 threads calling envs.reset() -> get_initial_state
Exact pattern: _execute_rollout() -> envs.reset() -> get_initial_state -> client.get()
"""

import asyncio
import threading
import time
from typing import Any, Dict, List

import httpx


class EnvResetClient:
    """
    Simulates the exact pattern from your code:
    50 threads -> _execute_rollout() -> envs.reset() -> get_initial_state -> client.get()
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.initial_state_url = f"{base_url}/control/initial_state"

    async def get_initial_state(self, thread_id: int) -> Dict[str, Any]:
        """
        Simulates the get_initial_state call from your McpGym code.
        This is the slow HTTP call that happens during envs.reset().
        """
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # This is the exact pattern from your code
                initial_state_response = await client.get(
                    self.initial_state_url,
                    headers=headers,
                    timeout=30.0,
                )
                initial_state_response.raise_for_status()
                result = initial_state_response.json()

                end_time = time.time()
                duration = end_time - start_time

                return {"thread_id": thread_id, "success": True, "duration": duration, "initial_state": result}

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            return {"thread_id": thread_id, "success": False, "duration": duration, "error": str(e)}

    async def envs_reset(self, thread_id: int) -> Dict[str, Any]:
        """
        Simulates envs.reset() which internally calls get_initial_state.
        This is what gets called from _execute_rollout().
        """
        print(f"🔄 Thread {thread_id}: envs.reset() called")

        # This simulates the envs.reset() -> get_initial_state call chain
        return await self.get_initial_state(thread_id)


async def _execute_rollout(thread_id: int, client: EnvResetClient) -> Dict[str, Any]:
    """
    Simulates _execute_rollout() function that calls envs.reset().
    This runs concurrently using asyncio, matching your actual pattern.
    """
    print(f"🚀 Rollout {thread_id}: _execute_rollout() started")

    # This is where envs.reset() gets called
    result = await client.envs_reset(thread_id)
    return result


async def run_burst_test(num_clients: int = 50, server_url: str = "http://localhost:8000"):
    """
    Run burst test simulating 50 concurrent _execute_rollout() calls.
    Each one calls envs.reset() -> get_initial_state -> client.get()
    """
    print(f"🚀 Starting burst test with {num_clients} concurrent rollouts")
    print(f"🎯 Target server: {server_url}")
    print(f"📋 Pattern: _execute_rollout() -> envs.reset() -> get_initial_state -> client.get()")

    client = EnvResetClient(server_url)

    # Create tasks for concurrent rollouts (simulating your threading pattern)
    start_time = time.time()
    tasks = [_execute_rollout(i, client) for i in range(num_clients)]

    # Run all rollouts concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time()
    total_duration = end_time - start_time

    # Analyze results
    successful = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
    exceptions = [r for r in results if not isinstance(r, dict)]

    print(f"\n📊 BURST TEST RESULTS:")
    print(f"   Total rollouts: {num_clients}")
    print(f"   Total time: {total_duration:.3f}s")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Exceptions: {len(exceptions)}")

    if successful:
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        min_duration = min(r["duration"] for r in successful)
        max_duration = max(r["duration"] for r in successful)

        print(f"   Average rollout duration: {avg_duration:.3f}s")
        print(f"   Min rollout duration: {min_duration:.3f}s")
        print(f"   Max rollout duration: {max_duration:.3f}s")

        # Show sample successful result
        sample = successful[0]
        print(f"\n✅ Sample successful rollout:")
        print(f"   Thread ID: {sample['thread_id']}")
        print(f"   Initial state: {sample['initial_state']['observation']}")
        print(f"   Timestamp: {sample['initial_state']['timestamp']}")

    if failed:
        print(f"\n❌ Sample failed rollouts:")
        for fail in failed[:3]:  # Show first 3 failures
            print(f"   Thread {fail['thread_id']}: {fail['error']}")

    if exceptions:
        print(f"\n💥 Sample exceptions:")
        for exc in exceptions[:3]:  # Show first 3 exceptions
            print(f"   {type(exc).__name__}: {exc}")

    # Key test: If concurrent, should take ~1 second. If sequential, ~50 seconds.
    if total_duration < 5:  # Allow some overhead
        print(f"\n🎉 CONCURRENCY WORKING! Total time {total_duration:.3f}s (expected ~1s for concurrent)")
    else:
        print(f"\n⚠️  POSSIBLE SEQUENTIAL EXECUTION! Total time {total_duration:.3f}s (expected ~1s for concurrent)")

    return len(successful) == num_clients


def main():
    """Run the burst test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Envs Reset Burst Test - Simulates 50 rollouts calling get_initial_state"
    )
    parser.add_argument("--rollouts", type=int, default=50, help="Number of concurrent rollouts")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")

    args = parser.parse_args()

    success = asyncio.run(run_burst_test(args.rollouts, args.server))

    if success:
        print(f"\n🎉 ALL {args.rollouts} ROLLOUTS SUCCESSFUL!")
        exit(0)
    else:
        print(f"\n💥 SOME ROLLOUTS FAILED!")
        exit(1)


if __name__ == "__main__":
    main()
