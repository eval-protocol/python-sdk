import asyncio


async def allow_litellm_logging_to_start() -> None:
    """Let LiteLLM claim queued logging callbacks before the event loop can close.

    LiteLLM 1.80+ queues coroutine objects in a process-global logging worker.
    When pytest replaces a function-scoped event loop before the worker claims a
    callback, LiteLLM drops that coroutine while rebinding the queue. Two event
    loop turns let the worker wrap the callback in a task so normal loop
    shutdown can cancel it cleanly.
    """
    await asyncio.sleep(0)
    await asyncio.sleep(0)
