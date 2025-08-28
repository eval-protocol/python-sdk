from typing import Literal, TypedDict
from pytest import StashKey


class ExperimentLink(TypedDict):
    experiment_id: str
    job_link: str
    status: Literal["success", "failure"]


EXPERIMENT_LINKS_STASH_KEY = StashKey[list[ExperimentLink]]()


def store_experiment_link(experiment_id: str, job_link: str, status: Literal["success", "failure"]):
    """Store experiment link in pytest session stash."""
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
            global EXPERIMENT_LINKS_STASH_KEY

            if EXPERIMENT_LINKS_STASH_KEY not in session.stash:  # pyright: ignore[reportAny]
                session.stash[EXPERIMENT_LINKS_STASH_KEY] = []  # pyright: ignore[reportAny]

            session.stash[EXPERIMENT_LINKS_STASH_KEY].append(  # pyright: ignore[reportAny]
                {"experiment_id": experiment_id, "job_link": job_link, "status": status}
            )
        else:
            pass

    except Exception as e:  # pyright: ignore[reportUnusedVariable]
        pass
