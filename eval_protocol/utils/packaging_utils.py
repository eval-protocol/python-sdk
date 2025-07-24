import logging
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_pip_executable(venv_pip_path: Optional[str] = None) -> List[str]:
    """Determines the pip executable command parts."""
    if venv_pip_path and os.path.exists(venv_pip_path) and os.access(venv_pip_path, os.X_OK):
        logger.info(f"Using specified pip executable: {venv_pip_path}")
        return [venv_pip_path]

    # Try to find pip in the current virtual environment's scripts/bin directory
    # sys.executable should be /path/to/.venv/bin/python
    # So, pip should be /path/to/.venv/bin/pip
    # On Windows, it might be /path/to/.venv/Scripts/pip.exe
    potential_pip_path = os.path.join(os.path.dirname(sys.executable), "pip")
    if os.name == "nt":  # Windows check
        potential_pip_path += ".exe"

    if os.path.exists(potential_pip_path) and os.access(potential_pip_path, os.X_OK):
        logger.info(f"Using auto-detected pip executable: {potential_pip_path}")
        return [potential_pip_path]

    # Fallback to sys.executable -m pip (should generally work if python is from the venv)
    logger.info(f"Using pip via: {sys.executable} -m pip")
    return [sys.executable, "-m", "pip"]


def install_requirements(
    requirements_list: List[str],
    venv_pip_path: Optional[str] = None,  # User can specify e.g. ".venv/bin/pip"
    log_output: bool = True,
) -> None:
    """
    Installs a list of Python package requirements using pip.

    Args:
        requirements_list: A list of requirement strings (e.g., ["package_a==1.0", "package_b>=2.0"]).
        venv_pip_path: Optional path to the specific pip executable to use.
        log_output: If True, logs the stdout and stderr of the pip command.
    """
    if not requirements_list:
        logger.debug("No requirements provided to install.")
        return

    unique_requirements = sorted(list(set(req.strip() for req in requirements_list if req.strip())))
    if not unique_requirements:
        logger.debug("No unique, non-empty requirements to install after stripping.")
        return

    pip_command_parts = get_pip_executable(venv_pip_path)

    # Create a temporary requirements file
    # delete=False is used because on Windows, a file opened for writing cannot be opened by another process.
    # We will manually delete it in the finally block.
    tmp_req_fd, tmp_req_file_path = tempfile.mkstemp(suffix=".txt", prefix="rk_reqs_")

    try:
        with os.fdopen(tmp_req_fd, "w") as tmp_req_file:
            for req in unique_requirements:
                tmp_req_file.write(req + "\n")

        logger.info(
            f"Attempting to install requirements: {unique_requirements} using pip command: {' '.join(pip_command_parts)} -r {tmp_req_file_path}"
        )

        command = pip_command_parts + ["install", "-r", tmp_req_file_path]

        process = subprocess.run(
            command,
            check=True,  # Raise CalledProcessError on non-zero exit
            capture_output=True,
            text=True,  # Decodes stdout/stderr as text
            encoding="utf-8",  # Explicit encoding
            errors="replace",  # Handle potential encoding errors in pip output
        )
        if log_output and process.stdout:
            logger.info(f"Pip install stdout:\n{process.stdout.strip()}")
        # pip often uses stderr for progress/warnings even on success
        if log_output and process.stderr:
            logger.info(f"Pip install stderr:\n{process.stderr.strip()}")
        logger.info(f"Successfully installed requirements: {unique_requirements}")

    except subprocess.CalledProcessError as e:
        error_message = f"Error installing requirements from {tmp_req_file_path}.\n"
        error_message += f"Command: {' '.join(e.cmd)}\n"
        if e.stdout:
            error_message += f"Pip stdout:\n{e.stdout.strip()}\n"
        if e.stderr:
            error_message += f"Pip stderr:\n{e.stderr.strip()}\n"
        logger.error(error_message)
        raise RuntimeError(
            f"Failed to install requirements: {unique_requirements}. Details:\n{e.stderr or e.stdout or str(e)}"
        )
    except FileNotFoundError:
        logger.error(
            f"Pip executable not found: {' '.join(pip_command_parts)}. Please ensure pip is installed and in PATH, or venv_pip_path is correct."
        )
        raise
    finally:
        if os.path.exists(tmp_req_file_path):
            os.remove(tmp_req_file_path)
            logger.debug(f"Removed temporary requirements file: {tmp_req_file_path}")
