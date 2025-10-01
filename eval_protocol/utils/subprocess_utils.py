"""Cross-platform subprocess utilities for running scripts and commands."""

import os
import platform
import subprocess
from typing import Optional


def run_script_cross_platform(
    script_name: str,
    working_directory: str,
    capture_output: bool = True,
    print_output: bool = False,
    inherit_stdout: bool = False,
) -> subprocess.Popen:
    """
    Run a script in a cross-platform manner.

    Args:
        script_name: Name of the script to run (e.g., "start.sh")
        working_directory: Directory to run the script in
        capture_output: Whether to capture stdout/stderr
        print_output: Whether to print output in real-time
        inherit_stdout: Whether to inherit stdout from parent process

    Returns:
        subprocess.Popen object for the running process

    Raises:
        RuntimeError: If the script fails to start or execute
    """
    script_path = os.path.join(working_directory, script_name)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Determine stdout handling
    if inherit_stdout:
        stdout = None  # Inherit from parent process
        stderr = subprocess.STDOUT  # Still capture stderr
    elif capture_output:
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
    else:
        stdout = None
        stderr = None

    if platform.system() == "Windows":
        # On Windows, use cmd.exe to run the script
        cmd = ["cmd.exe", "/c", script_name]
        process = subprocess.Popen(
            cmd,
            cwd=working_directory,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    else:
        # On Unix-like systems, make executable and run with proper shebang
        os.chmod(script_path, 0o755)

        # Use the full path to the script with shell=True
        process = subprocess.Popen(
            script_path,
            stdout=stdout,
            stderr=stderr,
            text=True,
            shell=True,
        )

    # Print output in real-time if requested
    if print_output and capture_output and process.stdout:
        for line in process.stdout:
            print(line, end="")

    return process


def run_script_and_wait(
    script_name: str,
    working_directory: str,
    print_output: bool = False,
    inherit_stdout: bool = False,
    timeout: Optional[int] = None,
) -> int:
    """
    Run a script and wait for it to complete.

    Args:
        script_name: Name of the script to run
        working_directory: Directory to run the script in
        print_output: Whether to print output in real-time
        inherit_stdout: Whether to inherit stdout from parent process
        timeout: Maximum time to wait for the script to complete

    Returns:
        Return code of the script

    Raises:
        RuntimeError: If the script fails to execute
        subprocess.TimeoutExpired: If the script times out
    """
    process = run_script_cross_platform(
        script_name=script_name,
        working_directory=working_directory,
        capture_output=print_output and not inherit_stdout,
        print_output=print_output,
        inherit_stdout=inherit_stdout,
    )

    try:
        returncode = process.wait(timeout=timeout)
        if returncode != 0:
            raise RuntimeError(f"Script '{script_name}' failed with return code {returncode}")
        return returncode
    except subprocess.TimeoutExpired:
        process.kill()
        raise
