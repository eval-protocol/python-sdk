from typing import TypedDict
from pytest import StashKey


class ResultsUrl(TypedDict):
    invocation_id: str
    pivot_url: str
    table_url: str


RESULTS_URLS_STASH_KEY = StashKey[dict[str, ResultsUrl]]()


def _store_local_ui_url_in_stash(invocation_id: str, pivot_url: str, table_url: str):
    """Store results URL in pytest session stash."""
    try:
        import sys

        # Walk up the call stack to find the pytest session
        session = None
        frame = sys._getframe()  # pyright: ignore[reportPrivateUsage]
        while frame:
            if "session" in frame.f_locals and hasattr(frame.f_locals["session"], "stash"):  # pyright: ignore[reportAny]
                session = frame.f_locals["session"]  # pyright: ignore[reportAny]
                break
            frame = frame.f_back

        if session is not None:
            global RESULTS_URLS_STASH_KEY

            if RESULTS_URLS_STASH_KEY not in session.stash:  # pyright: ignore[reportAny]
                session.stash[RESULTS_URLS_STASH_KEY] = {}  # pyright: ignore[reportAny]

            # Store by invocation_id as key - automatically handles deduplication
            session.stash[RESULTS_URLS_STASH_KEY][invocation_id] = {  # pyright: ignore[reportAny]
                "invocation_id": invocation_id,
                "pivot_url": pivot_url,
                "table_url": table_url,
            }
        else:
            pass

    except Exception as e:  # pyright: ignore[reportUnusedVariable]
        pass


def store_local_ui_url(invocation_id: str, pivot_url: str, table_url: str):
    """Public function to store results URL in pytest session stash."""
    _store_local_ui_url_in_stash(invocation_id, pivot_url, table_url)
