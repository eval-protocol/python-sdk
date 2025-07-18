import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import toml
import yaml

# from deepdiff import DeepDiff
from dotenv import load_dotenv
# from loguru import logger

res = load_dotenv()
# if not res:
#     logger.warning("No .env file found")

SOURCE_DIR = Path(__file__).parent
AIRLINE_DATA_DIR = SOURCE_DIR
AIRLINE_DB_PATH = AIRLINE_DATA_DIR / "db.json"
AIRLINE_POLICY_PATH = AIRLINE_DATA_DIR / "policy.md"
AIRLINE_TASK_SET_PATH = AIRLINE_DATA_DIR / "tasks.json"


def get_dict_hash(obj: dict) -> str:
    """
    Generate a unique hash for dict.
    Returns a hex string representation of the hash.
    """
    hash_string = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(hash_string.encode()).hexdigest()


# def show_dict_diff(dict1: dict, dict2: dict) -> str:
#     """
#     Show the difference between two dictionaries.
#     """
#     diff = DeepDiff(dict1, dict2)
#     return diff


def get_now() -> str:
    """
    Returns the current date and time in the format YYYYMMDD_HHMMSS.
    """
    now = datetime.now()
    return format_time(now)


def format_time(time: datetime) -> str:
    """
    Format the time in the format YYYYMMDD_HHMMSS.
    """
    return time.isoformat()


def get_commit_hash() -> str:
    """
    Get the commit hash of the current directory.
    """
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
            .split("\n")[0]
        )
    except Exception as e:
        # logger.error(f"Failed to get git hash: {e}")
        print(f"Failed to get git hash: {e}")
        commit_hash = "unknown"
    return commit_hash

# NOTE: When using the results of load_file(), we need to pay attention to the case
# where the value is None when loading from json or yaml, the key will be missing in
# toml since there is no "null" in toml.


def load_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Load the content of a file from a path based on the file extension.

    Args:
        path: The path to the file to load.
        **kwargs: Additional keyword arguments to pass to the file reader.

    Returns:
        The data dictionary loaded from the file.
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r") as fp:
            data = json.load(fp, **kwargs)
    elif path.suffix == ".yaml" or path.suffix == ".yml":
        with open(path, "r") as fp:
            data = yaml.load(fp, Loader=yaml.SafeLoader, **kwargs)
    elif path.suffix == ".toml":
        with open(path, "r") as fp:
            data = toml.load(fp, **kwargs)
    elif path.suffix == ".txt" or path.suffix == ".md":
        encoding = kwargs.pop("encoding", None)
        if len(kwargs) > 0:
            raise ValueError(f"Unsupported keyword arguments: {kwargs}")
        with open(path, "r", encoding=encoding) as fp:
            data = fp.read()
    else:
        raise ValueError(f"Unsupported file extension: {path}")
    return data


def dump_file(path: str | Path, data: dict[str, Any], **kwargs: Any) -> None:
    """Dump data content to a file based on the file extension.

    Args:
        path: The path to the file to dump the data to.
        data: The data dictionary to dump to the file.
        **kwargs: Additional keyword arguments to pass to the file writer.
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)  # make dir if not exists

    if path.suffix == ".json":
        with open(path, "w") as fp:
            json.dump(data, fp, **kwargs)
    elif path.suffix == ".yaml" or path.suffix == ".yml":
        with open(path, "w") as fp:
            yaml.dump(data, fp, **kwargs)
    elif path.suffix == ".toml":
        # toml cannot dump the Enum values, so we need to convert them to strings
        data_str = json.dumps(data)
        new_data = json.loads(data_str)
        with open(path, "w") as fp:
            toml.dump(new_data, fp, **kwargs)
    elif path.suffix == ".txt" or path.suffix == ".md":
        encoding = kwargs.pop("encoding", None)
        if len(kwargs) > 0:
            raise ValueError(f"Unsupported keyword arguments: {kwargs}")
        with open(path, "w", encoding=encoding) as fp:
            fp.write(data)
    else:
        raise ValueError(f"Unsupported file extension: {path}")
