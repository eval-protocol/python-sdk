import logging
import os

logger = logging.getLogger(__name__)

# do not inherit root logger since we are a handler ourselves
logger.propagate = False

logger.addHandler(logging.StreamHandler())

if os.environ.get("EP_DEBUG") == "true":
    logger.setLevel(logging.DEBUG)
    logger.debug("EP_DEBUG=true detected, set log level to DEBUG")


class RolloutIdFilter(logging.Filter):
    """
    A filter that simply adds the rollout_id to the record so that you don't
    have to pass it as extra data every time you log.
    """

    def __init__(self, rollout_id: str):
        self.rollout_id = rollout_id

    def filter(self, record):
        logger.debug(f"Filtering record with rollout_id: {self.rollout_id}")
        record.rollout_id = self.rollout_id
        return True
