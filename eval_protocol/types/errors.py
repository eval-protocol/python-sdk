class RolloutTerminationException(Exception):
    """
    Exception raised during rollout. This error means the rollout needs to be terminated and its not retriable,
    and the trajectory need to be perserved for future evalution or analysis.

    For example, if the policy (llm) returns 400 User error, we need to end the rollout but keep the trajectory.
    It differs from other exceptions such as network error, which are retriable and the trajectory should be discarded if
    it fails eventually after retries.

    This will cause trajectory.termination_reason to be set to TerminationReason.INTERRUPTED.
    """

    pass
