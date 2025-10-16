"""
Traces fetching handler for Langfuse integration.
"""

import time
import random
import logging
import asyncio
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
import redis
from .redis_utils import get_insertion_ids
from .models import ProxyConfig, LangfuseTracesResponse, TraceResponse, TracesParams

logger = logging.getLogger(__name__)


def _extract_tag_value(tags: Optional[List[str]], prefix: str) -> Optional[str]:
    """Extract value from a tag with the given prefix (e.g., 'rollout_id:' or 'insertion_id:')."""
    if not tags:
        return None
    for tag in tags:
        if tag.startswith(prefix):
            return tag.split(":", 1)[1]
    return None


def _serialize_trace_to_dict(trace_full: Any) -> Dict[str, Any]:
    """Convert Langfuse trace object to dict format."""
    timestamp = getattr(trace_full, "timestamp", None)

    return {
        "id": trace_full.id,
        "name": getattr(trace_full, "name", None),
        "user_id": getattr(trace_full, "user_id", None),
        "session_id": getattr(trace_full, "session_id", None),
        "tags": getattr(trace_full, "tags", []),
        "timestamp": str(timestamp) if timestamp else None,
        "input": getattr(trace_full, "input", None),
        "output": getattr(trace_full, "output", None),
        "metadata": getattr(trace_full, "metadata", None),
        "observations": [
            {
                "id": obs.id,
                "type": getattr(obs, "type", None),
                "name": getattr(obs, "name", None),
                "start_time": str(getattr(obs, "start_time", None)) if getattr(obs, "start_time", None) else None,
                "end_time": str(getattr(obs, "end_time", None)) if getattr(obs, "end_time", None) else None,
                "input": getattr(obs, "input", None),
                "output": getattr(obs, "output", None),
                "parent_observation_id": getattr(obs, "parent_observation_id", None),
            }
            for obs in getattr(trace_full, "observations", [])
        ]
        if hasattr(trace_full, "observations")
        else [],
    }


async def _fetch_trace_list_with_retry(
    langfuse_client: Any,
    page: int,
    limit: int,
    tags: Optional[List[str]],
    user_id: Optional[str],
    session_id: Optional[str],
    name: Optional[str],
    environment: Optional[str],
    version: Optional[str],
    release: Optional[str],
    fields: Optional[str],
    from_ts: Optional[datetime],
    to_ts: Optional[datetime],
    max_retries: int,
) -> Any:
    """Fetch trace list with rate limit retry logic."""
    list_retries = 0
    while list_retries < max_retries:
        try:
            traces = langfuse_client.api.trace.list(
                page=page,
                limit=limit,
                tags=tags,
                user_id=user_id,
                session_id=session_id,
                name=name,
                environment=environment,
                version=version,
                release=release,
                fields=fields,
                from_timestamp=from_ts,
                to_timestamp=to_ts,
                order_by="timestamp.desc",
            )

            # If no results, possible due to indexing delay--remote rollout processor just finished pushing rows to Langfuse
            if traces and traces.meta and traces.meta.total_items == 0 and page == 1:
                raise Exception("Empty results")

            return traces
        except Exception as e:
            list_retries += 1
            if list_retries < max_retries and ("429" in str(e) or "Empty results" in str(e)):
                sleep_time = 2**list_retries  # Exponential backoff for rate limits
                logger.warning(
                    "Retrying trace.list in %ds (attempt %d/%d): %s", sleep_time, list_retries, max_retries, str(e)
                )
                await asyncio.sleep(sleep_time)
            elif list_retries == max_retries:
                # Return 404 if we've retried max_retries
                # TODO: write some tests around proxy exception handling
                logger.error("Failed to fetch trace list after %d retries: %s", max_retries, e)
                raise HTTPException(
                    status_code=404, detail=f"Failed to fetch traces after {max_retries} retries: {str(e)}"
                )
            else:
                # Catch all other exceptions
                logger.error("Failed to fetch trace list: %s", e)
                raise HTTPException(status_code=500, detail=f"Failed to fetch traces: {str(e)}")


async def _fetch_trace_detail_with_retry(
    langfuse_client: Any,
    trace_id: str,
    max_retries: int,
) -> Optional[Any]:
    """Fetch full trace details with rate limit retry logic."""
    detail_retries = 0
    while detail_retries < max_retries:
        try:
            trace_full = langfuse_client.api.trace.get(trace_id)
            return trace_full
        except Exception as e:
            detail_retries += 1
            if "429" in str(e) and detail_retries < max_retries:
                sleep_time = 2**detail_retries  # Exponential backoff for rate limits
                logger.warning(
                    "Rate limit hit on trace.get(%s), retrying in %ds (attempt %d/%d)",
                    trace_id,
                    sleep_time,
                    detail_retries,
                    max_retries,
                )
                await asyncio.sleep(sleep_time)
            elif "Not Found" in str(e) or "404" in str(e):
                logger.debug("Trace %s not found, skipping", trace_id)
                return None
            else:
                logger.warning("Failed to fetch trace %s after %d retries: %s", trace_id, max_retries, e)
                return None


async def fetch_langfuse_traces(
    config: ProxyConfig,
    redis_client: redis.Redis,
    request: Request,
    params: TracesParams,
):
    """
    Fetch full traces from Langfuse for the specified project.

    This endpoint uses the stored Langfuse keys for the project and polls
    traces based on the provided filters.

    If project_id is not provided, uses the default project.

    Returns a list of full trace objects (including observations) in JSON format.
    """

    # Preprocess traces request
    if config.preprocess_traces_request:
        params = config.preprocess_traces_request(request, params)

    tags = params.tags
    project_id = params.project_id
    limit = params.limit
    sample_size = params.sample_size
    user_id = params.user_id
    session_id = params.session_id
    name = params.name
    environment = params.environment
    version = params.version
    release = params.release
    fields = params.fields
    hours_back = params.hours_back
    from_timestamp = params.from_timestamp
    to_timestamp = params.to_timestamp
    sleep_between_gets = params.sleep_between_gets
    max_retries = params.max_retries

    # Use default project if not specified
    if project_id is None:
        project_id = config.default_project_id

    # Validate project_id
    if project_id not in config.langfuse_keys:
        raise HTTPException(
            status_code=404,
            detail=f"Project ID '{project_id}' not found. Available projects: {list(config.langfuse_keys.keys())}",
        )

    # Extract rollout_id from tags for Redis lookup
    rollout_id = _extract_tag_value(tags, "rollout_id:")

    try:
        # Import the Langfuse adapter
        from langfuse import Langfuse

        # Create Langfuse client with the project's keys
        langfuse_client = Langfuse(
            public_key=config.langfuse_keys[project_id]["public_key"],
            secret_key=config.langfuse_keys[project_id]["secret_key"],
            host=config.langfuse_host,
        )

        # Parse datetime strings if provided
        from_ts = None
        to_ts = None
        if from_timestamp:
            from_ts = datetime.fromisoformat(from_timestamp.replace("Z", "+00:00"))
        if to_timestamp:
            to_ts = datetime.fromisoformat(to_timestamp.replace("Z", "+00:00"))

        # Determine time window: explicit from/to takes precedence over hours_back
        if from_ts is None and to_ts is None and hours_back:
            to_ts = datetime.now()
            from_ts = to_ts - timedelta(hours=hours_back)

        # Get expected insertion_ids from Redis for completeness checking
        expected_ids: Set[str] = set()
        if rollout_id:
            expected_ids = get_insertion_ids(redis_client, rollout_id)
            logger.info(f"Fetching traces for rollout_id '{rollout_id}', expecting {len(expected_ids)} insertion_ids")
            if not expected_ids:
                logger.warning(
                    f"No expected insertion_ids found in Redis for rollout '{rollout_id}'. Returning empty traces."
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"No expected insertion_ids found in Redis for rollout '{rollout_id}'. Returning empty traces.",
                )

        # Track all traces we've collected across retry attempts
        trace_ids: Set[str] = set()  # Langfuse trace IDs (for deduplication)
        all_traces: List[Dict[str, Any]] = []  # Full trace data
        insertion_ids: Set[str] = set()  # Insertion IDs extracted from traces (for completeness check)

        for retry in range(max_retries):
            # On first attempt, use rollout_id tag. On retries, target missing insertion_ids
            if retry == 0:
                fetch_tags = tags
            else:
                # Build targeted tags for missing insertion_ids
                missing_ids = expected_ids - insertion_ids
                fetch_tags = [f"insertion_id:{id}" for id in missing_ids]
                logger.info(
                    f"Retry {retry}: Targeting {len(fetch_tags)} missing insertion_ids for rollout '{rollout_id}' (last5): {[id[-5:] for id in sorted(missing_ids)[:10]]}{'...' if len(missing_ids) > 10 else ''}"
                )

            current_page = 1
            collected = 0

            while collected < limit:
                current_page_limit = min(100, limit - collected)  # Langfuse API max is 100

                # Fetch trace list with rate limit retry logic
                traces = await _fetch_trace_list_with_retry(
                    langfuse_client,
                    current_page,
                    current_page_limit,
                    fetch_tags,
                    user_id,
                    session_id,
                    name,
                    environment,
                    version,
                    release,
                    fields,
                    from_ts,
                    to_ts,
                    max_retries,
                )

                if not traces or not traces.data:
                    logger.debug("No more traces found on page %d", current_page)
                    break

                # For traces we find not in our current list of traces, do trace.get
                for trace_info in traces.data:
                    if trace_info.id in trace_ids:
                        continue  # Skip already processed traces

                    if sleep_between_gets > 0:
                        await asyncio.sleep(sleep_between_gets)

                    # Fetch full trace with rate limit retry logic
                    trace_full = await _fetch_trace_detail_with_retry(
                        langfuse_client,
                        trace_info.id,
                        max_retries,
                    )

                    if trace_full:
                        try:
                            trace_dict = _serialize_trace_to_dict(trace_full)
                            all_traces.append(trace_dict)
                            trace_ids.add(trace_info.id)

                            # Extract insertion_id for completeness checking
                            insertion_id = _extract_tag_value(trace_dict.get("tags", []), "insertion_id:")
                            if insertion_id:
                                insertion_ids.add(insertion_id)
                                logger.debug(f"Found insertion_id '{insertion_id}' for rollout '{rollout_id}'")

                        except Exception as e:
                            logger.warning("Failed to serialize trace %s: %s", trace_info.id, e)
                            continue

                collected += len(traces.data)

                # Check if we have more pages
                if hasattr(traces.meta, "page") and hasattr(traces.meta, "total_pages"):
                    if traces.meta.page >= traces.meta.total_pages:
                        break
                elif len(traces.data) < current_page_limit:
                    break

                current_page += 1

            # If we have all expected completions or more, return traces. At least once is ok.
            if expected_ids <= insertion_ids:
                logger.info(
                    f"Traces complete for rollout '{rollout_id}': {len(insertion_ids)}/{len(expected_ids)} insertion_ids found, returning {len(all_traces)} traces"
                )
                if sample_size is not None and len(all_traces) > sample_size:
                    all_traces = random.sample(all_traces, sample_size)
                    logger.info(f"Sampled down to {sample_size} traces")

                return LangfuseTracesResponse(
                    project_id=project_id,
                    total_traces=len(all_traces),
                    traces=[TraceResponse(**trace) for trace in all_traces],
                )

            # If it doesn't match, wait and do loop again (exponential backoff)
            if retry < max_retries - 1:
                wait_time = 2**retry
                still_missing = expected_ids - insertion_ids
                logger.info(
                    f"Attempt {retry + 1}/{max_retries}. Found {len(insertion_ids)}/{len(expected_ids)} for rollout '{rollout_id}'. Still missing (last5): {[id[-5:] for id in sorted(still_missing)[:10]]}{'...' if len(still_missing) > 10 else ''}. Waiting {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

        logger.error(
            f"Incomplete traces for rollout_id '{rollout_id}': Found {len(insertion_ids)}/{len(expected_ids)} completions."
        )
        raise HTTPException(
            status_code=404,
            detail=f"Incomplete traces for rollout_id '{rollout_id}': Found {len(insertion_ids)}/{len(expected_ids)} completions.",
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="Langfuse SDK not installed. Install with: pip install langfuse")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching traces from Langfuse: {str(e)}")


async def pointwise_fetch_langfuse_trace(
    config: ProxyConfig,
    redis_client: redis.Redis,
    request: Request,
    params: TracesParams,
):
    """
    Fetch the latest trace from Langfuse for the specified project.

    Since insertion_ids are UUID v7 (time-ordered), we only fetch the last one
    as it contains all accumulated information from the pointwise evaluation.

    Returns a single trace object or raises if not found.
    """

    # Preprocess traces request
    if config.preprocess_traces_request:
        params = config.preprocess_traces_request(request, params)

    tags = params.tags
    project_id = params.project_id
    user_id = params.user_id
    session_id = params.session_id
    name = params.name
    environment = params.environment
    version = params.version
    release = params.release
    fields = params.fields
    hours_back = params.hours_back
    from_timestamp = params.from_timestamp
    to_timestamp = params.to_timestamp
    sleep_between_gets = params.sleep_between_gets
    max_retries = params.max_retries

    # Use default project if not specified
    if project_id is None:
        project_id = config.default_project_id

    # Validate project_id
    if project_id not in config.langfuse_keys:
        raise HTTPException(
            status_code=404,
            detail=f"Project ID '{project_id}' not found. Available projects: {list(config.langfuse_keys.keys())}",
        )

    # Extract rollout_id from tags for Redis lookup
    rollout_id = _extract_tag_value(tags, "rollout_id:")

    try:
        # Import the Langfuse adapter
        from langfuse import Langfuse

        # Create Langfuse client with the project's keys
        logger.debug(f"Connecting to Langfuse at {config.langfuse_host} for project '{project_id}'")
        langfuse_client = Langfuse(
            public_key=config.langfuse_keys[project_id]["public_key"],
            secret_key=config.langfuse_keys[project_id]["secret_key"],
            host=config.langfuse_host,
        )

        # Parse datetime strings if provided
        from_ts = None
        to_ts = None
        if from_timestamp:
            from_ts = datetime.fromisoformat(from_timestamp.replace("Z", "+00:00"))
        if to_timestamp:
            to_ts = datetime.fromisoformat(to_timestamp.replace("Z", "+00:00"))

        # Determine time window: explicit from/to takes precedence over hours_back
        if from_ts is None and to_ts is None and hours_back:
            to_ts = datetime.now()
            from_ts = to_ts - timedelta(hours=hours_back)

        # Get insertion_ids from Redis to find the latest one
        expected_ids: Set[str] = set()
        if rollout_id:
            expected_ids = get_insertion_ids(redis_client, rollout_id)
            logger.info(
                f"Pointwise fetch for rollout_id '{rollout_id}', found {len(expected_ids)} insertion_ids in Redis"
            )
            if not expected_ids:
                logger.warning(
                    f"No insertion_ids found in Redis for rollout '{rollout_id}'. Cannot determine latest trace."
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"No insertion_ids found in Redis for rollout '{rollout_id}'. Cannot determine latest trace.",
                )

        # Get the latest (last) insertion_id since UUID v7 is time-ordered
        latest_insertion_id = max(expected_ids)  # UUID v7 max = newest
        logger.info(f"Targeting latest insertion_id: {latest_insertion_id} for rollout '{rollout_id}'")

        for retry in range(max_retries):
            # Fetch trace list targeting the latest insertion_id
            traces = await _fetch_trace_list_with_retry(
                langfuse_client,
                page=1,
                limit=1,  # Only need the one trace
                tags=[f"insertion_id:{latest_insertion_id}"],
                user_id=user_id,
                session_id=session_id,
                name=name,
                environment=environment,
                version=version,
                release=release,
                fields=fields,
                from_ts=from_ts,
                to_ts=to_ts,
                max_retries=max_retries,
            )

            if traces and traces.data:
                # Get the trace info
                trace_info = traces.data[0]
                logger.debug(f"Found trace {trace_info.id} for latest insertion_id {latest_insertion_id}")

                # Fetch full trace details
                trace_full = await _fetch_trace_detail_with_retry(
                    langfuse_client,
                    trace_info.id,
                    max_retries,
                )

                if trace_full:
                    trace_dict = _serialize_trace_to_dict(trace_full)
                    logger.info(
                        f"Successfully fetched latest trace for rollout '{rollout_id}', insertion_id: {latest_insertion_id}"
                    )
                    return LangfuseTracesResponse(
                        project_id=project_id,
                        total_traces=1,
                        traces=[TraceResponse(**trace_dict)],
                    )

            # If not successful and not last retry, sleep and continue
            if retry < max_retries - 1:
                wait_time = 2**retry
                logger.info(
                    f"Pointwise fetch attempt {retry + 1}/{max_retries} failed for rollout '{rollout_id}', insertion_id: {latest_insertion_id}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

        # After all retries failed
        logger.error(
            f"Failed to fetch latest trace for rollout '{rollout_id}', insertion_id: {latest_insertion_id} after {max_retries} retries"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Failed to fetch latest trace for rollout '{rollout_id}' after {max_retries} retries",
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="Langfuse SDK not installed. Install with: pip install langfuse")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching latest trace from Langfuse: {str(e)}")
