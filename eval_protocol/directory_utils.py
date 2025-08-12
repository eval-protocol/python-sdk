import os
from typing import Optional

# Shared constants for directory discovery
EVAL_PROTOCOL_DIR = ".eval_protocol"
PYTHON_FILES = ["pyproject.toml", "requirements.txt"]
DATASETS_DIR = "datasets"


def find_eval_protocol_dir() -> str:
    """
    Find the .eval_protocol directory by looking up the directory tree.

    Returns:
        Path to the .eval_protocol directory
    """
    # recursively look up for a .eval_protocol directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != "/":
        if os.path.exists(os.path.join(current_dir, EVAL_PROTOCOL_DIR)):
            log_dir = os.path.join(current_dir, EVAL_PROTOCOL_DIR)
            break
        current_dir = os.path.dirname(current_dir)
    else:
        # if not found, recursively look up until a pyproject.toml or requirements.txt is found
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != "/":
            if any(os.path.exists(os.path.join(current_dir, f)) for f in PYTHON_FILES):
                log_dir = os.path.join(current_dir, EVAL_PROTOCOL_DIR)
                break
            current_dir = os.path.dirname(current_dir)
        else:
            # get the PWD that this python process is running in
            log_dir = os.path.join(os.getcwd(), EVAL_PROTOCOL_DIR)

    # create the .eval_protocol directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def find_eval_protocol_datasets_dir() -> str:
    """
    Find the .eval_protocol/datasets directory by looking up the directory tree.

    Returns:
        Path to the .eval_protocol/datasets directory
    """
    log_dir = find_eval_protocol_dir()

    # create the datasets subdirectory
    datasets_dir = os.path.join(log_dir, DATASETS_DIR)
    os.makedirs(datasets_dir, exist_ok=True)

    return datasets_dir
