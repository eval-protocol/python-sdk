import json
import os
import re  # Added for Serveo URL parsing
import shutil  # Added for checking ssh availability
import signal
import subprocess
import time
from typing import IO, Any, Dict, List, Optional  # Added IO, Any, List, Dict, Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Store PIDs of started processes
managed_processes: Dict[int, Dict[str, Any]] = {}  # pid -> {process, command, log_file, log_file_path, env}

NGROK_API_URL = "http://127.0.0.1:4040/api/tunnels"


def start_process(
    command: List[str],  # Changed to List[str]
    log_file_path: str,
    cwd: Optional[str] = None,  # Changed to Optional[str]
    new_process_group: bool = True,
    env: Optional[Dict[str, str]] = None,  # Changed to Optional[Dict[str, str]]
) -> Optional[subprocess.Popen]:  # Changed to Optional[subprocess.Popen]
    """
    Starts a process in the background and logs its output.
    Stores the process information for later management.

    Args:
        command: A list representing the command and its arguments.
        log_file_path: Path to the file where stdout and stderr will be logged.
        cwd: The working directory for the command. Defaults to current directory.
        new_process_group: Whether to start the process in a new group (for Unix-like systems).
        env: Optional dictionary of environment variables for the new process.

    Returns:
        The Popen object for the started process.
    """
    print(f"Starting process: {' '.join(str(c) for c in command)}")  # Ensure all parts of command are str for join
    print(f"Logging output to: {log_file_path}")

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    log_file: IO[Any] = open(log_file_path, "w")

    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    try:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=log_file,
            cwd=cwd if cwd else os.getcwd(),
            preexec_fn=os.setsid if (os.name != "nt" and new_process_group) else None,
            env=process_env,
        )

        managed_processes[process.pid] = {
            "process": process,  # Type: subprocess.Popen
            "command": command,  # Type: List[str]
            "log_file": log_file,  # Type: IO[Any] (TextIOWrapper)
            "log_file_path": log_file_path,  # Type: str
            "is_ngrok": "ngrok" in command[0] if command else False,
            "env": env,  # Type: Optional[Dict[str, str]]
        }
        print(f"Process started with PID: {process.pid}")
        return process
    except Exception as e:
        print(f"Failed to start process: {e}")
        log_file.close()  # Ensure log file is closed on error
        return None


# --- Ngrok functions ---
def get_ngrok_public_url(retries: int = 5, delay: int = 3) -> Optional[str]:
    """
    Queries the local ngrok API to get the public HTTPS URL.
    """
    if not REQUESTS_AVAILABLE:
        print("ERROR: 'requests' library is not installed. Cannot fetch ngrok URL automatically.")
        print("Please install it, e.g., pip install requests")
        return None

    # NGROK_API_URL is now a module-level constant
    for attempt in range(retries):
        try:
            response = requests.get(NGROK_API_URL, timeout=5)
            response.raise_for_status()
            tunnels_data = response.json()
            for tunnel in tunnels_data.get("tunnels", []):
                if tunnel.get("proto") == "https" and tunnel.get("public_url", "").startswith("https://"):
                    print(f"Found ngrok public URL: {tunnel['public_url']}")
                    return tunnel["public_url"]
            print(
                f"Attempt {attempt + 1}/{retries}: HTTPS tunnel not found yet in ngrok API response. Retrying in {delay}s..."
            )
        except requests.exceptions.ConnectionError:
            print(
                f"Attempt {attempt + 1}/{retries}: ngrok API not yet available at {NGROK_API_URL}. Retrying in {delay}s..."
            )
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries}: Error fetching ngrok URL: {e}. Retrying in {delay}s...")
        time.sleep(delay)
    print("ERROR: Failed to get ngrok public URL after multiple retries.")
    return None


def start_ngrok_and_get_url(
    local_port: int, ngrok_log_file: str, authtoken: Optional[str] = None
) -> tuple[Optional[subprocess.Popen], Optional[str]]:
    """
    Starts ngrok to expose a local port and retrieves its public HTTPS URL.
    """
    ngrok_command = [
        "ngrok",
        "http",
        str(local_port),
        "--log=stdout",
    ]  # Using --log=stdout to simplify log capture by start_process

    try:
        # Check if ngrok command is available
        ngrok_version_process = subprocess.run(["ngrok", "--version"], capture_output=True, text=True, check=True)
        print(f"Found ngrok version: {ngrok_version_process.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ngrok command not found or not executable. Please ensure ngrok is installed and in your PATH.")
        return None, None

    if authtoken:
        # This is usually done by the user beforehand with `ngrok config add-authtoken <token>`
        # Or by setting NGROK_AUTHTOKEN environment variable.
        # Forcing it via command line is also an option but less common for persistent setup.
        print(
            "Note: Ngrok authtoken should be pre-configured by the user (e.g., 'ngrok config add-authtoken <token>') or via NGROK_AUTHTOKEN env var."
        )
        # Example if passing via env for the subprocess:
        # ngrok_env = os.environ.copy()
        # ngrok_env["NGROK_AUTHTOKEN"] = authtoken
        # ngrok_process = start_process(ngrok_command, ngrok_log_file, new_process_group=False, env=ngrok_env)

    print(f"Attempting to start ngrok for port {local_port}...")
    # ngrok typically doesn't need to be in a new process group for simple start/stop.
    # Its logs will go to ngrok_log_file via start_process.
    ngrok_process = start_process(ngrok_command, ngrok_log_file, new_process_group=False)

    if not ngrok_process or ngrok_process.poll() is not None:
        print(f"ERROR: Failed to start ngrok. Check log: {ngrok_log_file}")
        # managed_processes should handle cleanup if start_process added it and it failed.
        # However, if ngrok_process is None, it wasn't added.
        # If it's not None but poll() is not None, it means it started and exited quickly.
        if ngrok_process and ngrok_process.pid in managed_processes:
            # This might be redundant if start_process failed and didn't add it,
            # or if it was added and then stop_process is called later.
            # For safety, ensure it's stopped if it was ever in managed_processes.
            pass  # stop_process will be called by the caller if needed
        return None, None

    print(f"ngrok process started with PID {ngrok_process.pid}. Waiting for tunnel URL...")
    # Increased sleep time as ngrok can take a moment to establish tunnel and API to update
    time.sleep(8)

    public_url = get_ngrok_public_url()

    if not public_url:
        print(f"ERROR: Could not retrieve public URL from ngrok API. Check log: {ngrok_log_file}")
        # If URL fetch fails, stop the ngrok process we started.
        stop_process(ngrok_process.pid)  # stop_process will remove it from managed_processes
        return None, None

    print(f"Successfully started ngrok and retrieved public URL: {public_url}")
    return ngrok_process, public_url


# --- End of Ngrok functions ---


def start_serveo_and_get_url(
    local_port: int, log_file_path: str, timeout_seconds: int = 20
) -> tuple[subprocess.Popen | None, str | None]:
    """
    Starts Serveo.net SSH tunnel to expose a local port and retrieves its public HTTPS URL.
    The SSH process is added to managed_processes.

    Args:
        local_port: The local port number to expose (e.g., 8001).
        log_file_path: Path to the file where Serveo SSH client output will be logged.
        timeout_seconds: How long to wait for Serveo to provide a URL.

    Returns:
        A tuple (ssh_process, public_url). (None, None) on failure.
    """
    if not shutil.which("ssh"):
        print("ERROR: 'ssh' command not found. Please ensure OpenSSH client is installed and in your PATH.")
        return None, None

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Using a temporary file for UserKnownHostsFile might be more robust on some systems than /dev/null
    # For simplicity, /dev/null is used here as specified in the plan.
    # On Windows, /dev/null equivalent is NUL.
    known_hosts_file = "/dev/null" if os.name != "nt" else "NUL"

    serveo_command = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"UserKnownHostsFile={known_hosts_file}",
        "-R",
        f"80:localhost:{local_port}",
        "serveo.net",
    ]

    print(f"Attempting to start Serveo.net tunnel for localhost:{local_port}...")
    print(f"Command: {' '.join(serveo_command)}")
    print(f"Logging Serveo SSH client output to: {log_file_path}")

    public_url = None
    ssh_process = None

    try:
        # We need to capture stdout to parse the URL.
        # stderr will also be captured by the same pipe.
        log_file = open(log_file_path, "w")
        ssh_process = subprocess.Popen(
            serveo_command,
            stdout=subprocess.PIPE,  # Capture stdout for parsing
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout pipe
            text=True,  # Decode output as text
            bufsize=1,  # Line-buffered
            universal_newlines=True,  # Ensure consistent line endings
            preexec_fn=(os.setsid if os.name != "nt" else None),  # New process group for proper termination
        )

        # Add to managed_processes early so it can be cleaned up if something goes wrong
        managed_processes[ssh_process.pid] = {
            "process": ssh_process,
            "command": serveo_command,
            "log_file": log_file,  # This log_file will store what we read from the pipe
            "log_file_path": log_file_path,
            "is_ngrok": False,
        }
        print(f"Serveo SSH process started with PID: {ssh_process.pid}. Waiting for URL...")

        url_pattern = re.compile(r"Forwarding HTTP traffic from (https://\S+\.serveo\.net)")

        start_time = time.time()
        if ssh_process.stdout:
            for line in iter(ssh_process.stdout.readline, ""):
                log_file.write(line)  # Write to the main log file
                log_file.flush()
                print(f"[Serveo PID {ssh_process.pid}]: {line.strip()}")  # Also print to console for live feedback

                match = url_pattern.search(line)
                if match:
                    public_url = match.group(1)
                    print(f"Found Serveo public URL: {public_url}")
                    break  # URL found

                if time.time() - start_time > timeout_seconds:
                    print(f"ERROR: Timeout ({timeout_seconds}s) waiting for Serveo URL.")
                    break  # Timeout
                if ssh_process.poll() is not None:  # Process terminated unexpectedly
                    print(f"ERROR: Serveo SSH process terminated unexpectedly. Check log: {log_file_path}")
                    break

            # If loop exited because readline returned '', process ended.
            if ssh_process.poll() is not None and not public_url:
                print(f"ERROR: Serveo SSH process ended before URL was found. Check log: {log_file_path}")

        else:  # Should not happen if Popen was successful
            print("ERROR: SSH process stdout stream not available.")

    except FileNotFoundError:
        print("ERROR: 'ssh' command not found. Please ensure OpenSSH client is installed and in your PATH.")
        if ssh_process and ssh_process.pid in managed_processes:
            stop_process(ssh_process.pid)  # Clean up if partially started
        return None, None
    except Exception as e:
        print(f"ERROR: An exception occurred while starting or monitoring Serveo: {e}")
        if ssh_process and ssh_process.pid in managed_processes:
            stop_process(ssh_process.pid)  # Clean up
        return None, None

    if not public_url:
        print("ERROR: Could not retrieve public URL from Serveo.net.")
        if ssh_process:  # If process was started, try to stop it
            stop_process(ssh_process.pid)
        return None, None

    # The ssh_process is kept running in the background by Popen.
    # It's up to the caller to manage its lifecycle (e.g., via stop_process or stop_all_processes).
    return ssh_process, public_url


def stop_process(pid: int):
    """
    Stops a managed process and its process group.
    Closes its log file.

    Args:
        pid: The PID of the process to stop.
    """
    if pid in managed_processes:
        proc_info = managed_processes[pid]
        process: subprocess.Popen = proc_info["process"]
        log_file: IO[Any] = proc_info["log_file"]
        command_list: List[str] = proc_info["command"]
        command_str = " ".join(str(c) for c in command_list)

        print(f"Stopping process PID {pid} ({command_str})...")
        try:
            if os.name == "nt":
                # For Windows, taskkill is more reliable for process trees
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    check=True,
                    capture_output=True,
                )
            else:
                # Send SIGTERM to the entire process group
                os.killpg(os.getpgid(pid), signal.SIGTERM)

            if hasattr(process, "stdout") and process.stdout and not process.stdout.closed:
                process.stdout.close()  # This might not be necessary if Popen handles it on terminate/kill
            process.wait(timeout=5)  # Wait for graceful termination
            print(f"Process PID {pid} terminated gracefully.")
        except subprocess.TimeoutExpired:
            print(f"Process PID {pid} did not terminate gracefully, sending SIGKILL...")
            if os.name == "nt":
                # On Windows, Popen.kill() is often sufficient for direct children.
                # taskkill /T is for tree. If os.killpg was used via setsid, this might be complex.
                # For simplicity, let's try process.kill() first.
                process.kill()
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)  # Kill the whole group
            process.wait(timeout=5)  # Wait for kill
            print(f"Process PID {pid} killed.")
        except ProcessLookupError:  # Process might have already exited
            print(f"Process PID {pid} already exited.")
        except Exception as e:
            print(f"Error stopping process PID {pid}: {e}")
        finally:
            if log_file and not log_file.closed:  # Check if log_file is TextIOWrapper and not closed
                log_file.close()
            if pid in managed_processes:  # Check if pid still exists before deleting
                del managed_processes[pid]
    else:
        print(f"Process with PID {pid} not found in managed list.")


def stop_all_processes():
    """
    Stops all currently managed processes.
    """
    print("Stopping all managed processes...")
    # Iterate over a copy of keys as `stop_process` modifies the dictionary
    for pid in list(managed_processes.keys()):
        stop_process(pid)
    print("All managed processes have been requested to stop.")


if __name__ == "__main__":
    import sys  # Import sys here for sys.executable

    # Example Usage:
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("Starting a sample sleep process...")
    sleep_log = os.path.join(log_dir, "sleep_test.log")
    # Using sys.executable to ensure we use the same python interpreter
    # For a real server, this would be [sys.executable, 'path/to/server_script.py']
    # Note: sys.executable is not defined in this scope if this file is run directly without importing sys first.
    # For the example, let's assume python is in path or use a simple command.

    # Test basic process start/stop
    print("Starting a sample sleep process...")
    sleep_log = os.path.join(log_dir, "sleep_test.log")
    # Using a simple platform-independent sleep command for the example
    sleep_command = ["timeout", "10"] if os.name == "nt" else ["sleep", "10"]
    proc = start_process(sleep_command, sleep_log)

    if proc and proc.pid:  # Check if proc is not None and has a pid
        print(f"Sleep process PID: {proc.pid}")
        print("Waiting for a few seconds before stopping...")
        time.sleep(3)
        stop_process(proc.pid)
    else:
        print("Failed to start sleep process.")

    print("\nStarting another process to test stop_all...")
    another_log = os.path.join(log_dir, "another_test.log")
    # Using a simple platform-independent sleep command for the example
    another_sleep_command = ["timeout", "10"] if os.name == "nt" else ["sleep", "10"]
    proc2 = start_process(another_sleep_command, another_log)
    if proc2 and proc2.pid:
        print(f"Another process PID: {proc2.pid}")
        time.sleep(1)
        # stop_all_processes() # This would be called by atexit or explicitly
    else:
        print("Failed to start another process.")

    # Test ngrok (manual execution needed if you want to see this run)
    # print("\nTesting ngrok start (requires ngrok in PATH and a service on port 8888)...")
    # ngrok_test_log = os.path.join(log_dir, "ngrok_test.log")
    # ngrok_proc, ngrok_url = start_ngrok_and_get_url(8888, ngrok_test_log)
    # if ngrok_proc and ngrok_url:
    #     print(f"Ngrok started: PID {ngrok_proc.pid}, URL {ngrok_url}")
    #     time.sleep(5) # Keep it running for 5s
    #     # stop_process(ngrok_proc.pid) # stop_all_processes will handle it
    # else:
    #     print("Failed to start ngrok for testing.")

    # stop_all_processes() will be called by atexit if this script is run
    # or can be called explicitly if needed.
    # For this example, we'll let atexit handle it if processes were started.
    # If running this __main__ block, ensure atexit is registered or call stop_all_processes()
    import atexit

    atexit.register(stop_all_processes)
    print("Subprocess manager example finished. Check logs in 'logs' directory.")
    print(
        "Remaining managed processes (should be empty if all stopped):",
        managed_processes.keys(),
    )
