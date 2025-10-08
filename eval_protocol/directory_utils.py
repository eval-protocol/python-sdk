import os
from typing import Optional

# Shared constants for directory discovery
EVAL_PROTOCOL_DIR = ".eval_protocol"
PYTHON_FILES = ["pyproject.toml", "requirements.txt"]
DATASETS_DIR = "datasets"


def find_eval_protocol_dir() -> str:
    """
    Find the .eval_protocol directory in the user's home folder.

    Returns:
        Path to the .eval_protocol directory in the user's home folder
    """
    # Always use the home folder for .eval_protocol directory
    log_dir = os.path.expanduser(os.path.join("~", EVAL_PROTOCOL_DIR))

    # create the .eval_protocol directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def find_eval_protocol_datasets_dir() -> str:
    """
    Find the .eval_protocol/datasets directory in the user's home folder.

    Returns:
        Path to the .eval_protocol/datasets directory in the user's home folder
    """
    log_dir = find_eval_protocol_dir()

    # create the datasets subdirectory
    datasets_dir = os.path.join(log_dir, DATASETS_DIR)
    os.makedirs(datasets_dir, exist_ok=True)

    return datasets_dir
