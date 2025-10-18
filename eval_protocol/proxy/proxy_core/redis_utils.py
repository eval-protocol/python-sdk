"""
Redis utilities for tracking chat completions via insertion IDs.
"""

import logging
from typing import Set, cast
import redis

logger = logging.getLogger(__name__)


def register_insertion_id(redis_client: redis.Redis, rollout_id: str, insertion_id: str) -> bool:
    """Register an insertion_id for a rollout_id in Redis.

    Tracks all expected completion insertion_ids for this rollout.

    Args:
        rollout_id: The rollout ID
        insertion_id: Unique identifier for this specific completion

    Returns:
        True if successful, False otherwise
    """
    try:
        redis_client.sadd(rollout_id, insertion_id)
        logger.info(f"Registered insertion_id {insertion_id} for rollout {rollout_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to register insertion_id for {rollout_id}: {e}")
        return False


def get_insertion_ids(redis_client: redis.Redis, rollout_id: str) -> Set[str]:
    """Get all expected insertion_ids for a rollout_id from Redis.

    Args:
        rollout_id: The rollout ID to get insertion_ids for

    Returns:
        Set of insertion_id strings, empty set if none found or on error
    """
    try:
        raw = redis_client.smembers(rollout_id)
        # Typing in redis stubs may be Awaitable[Set[Any]] | Set[Any]; at runtime this is a Set[bytes]
        raw_ids = cast(Set[object], raw)
        # Normalize to set[str]
        insertion_ids: Set[str] = set()
        for b in raw_ids:
            try:
                insertion_ids.add(b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else cast(str, b))
            except Exception:
                continue
        logger.debug(f"Found {len(insertion_ids)} expected insertion_ids for rollout {rollout_id}")
        return insertion_ids
    except Exception as e:
        logger.error(f"Failed to get insertion_ids for {rollout_id}: {e}")
        return set()
