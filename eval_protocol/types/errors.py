class NonSkippableException(Exception):
    """
    A type of custom exception raised during rollout or evaluation. This error means the rollout/evaluation result is not skippable and need to be
    processed explicitly.

    For example, if the policy (llm) returns 400 User error, we need to end the rollout but keep the trajectory.
    It differs from other exceptions such as network error, which are retriable and the trajectory should be discarded if
    it fails eventually after retries.
    """

    pass
