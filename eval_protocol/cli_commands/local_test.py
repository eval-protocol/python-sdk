import argparse
import os
import shlex
import subprocess
import sys
from typing import List

from .utils import _build_entry_point, _discover_and_select_tests


def _find_dockerfiles(root: str) -> List[str]:
    """Return Dockerfiles in the project root only (no recursive search)."""
    dockerfiles: List[str] = []
    root_dockerfile = os.path.join(root, "Dockerfile")
    if os.path.isfile(root_dockerfile):
        dockerfiles.append(root_dockerfile)
    return dockerfiles


def _build_docker_image(dockerfile_path: str, image_tag: str, build_extras: List[str] | None = None) -> bool:
    context_dir = os.path.dirname(dockerfile_path)
    print(f"Building Docker image '{image_tag}' from {dockerfile_path} ...")
    try:
        base_cmd = ["docker", "build"]
        if build_extras:
            base_cmd += build_extras
        base_cmd += ["-t", image_tag, "-f", dockerfile_path, context_dir]
        proc = subprocess.run(base_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)
        return proc.returncode == 0
    except FileNotFoundError:
        print("Error: docker not found in PATH. Install Docker or use --ignore-docker.")
        return False


def _run_pytest_host(pytest_target: str) -> int:
    """Run pytest against a target on the host and return its exit code."""
    # Always enforce a small success threshold for evaluation_test-based suites so that runs with all-zero scores fail.
    cmd = [sys.executable, "-m", "pytest", "--ep-success-threshold", "0.001", pytest_target, "-vs"]
    # Print the exact command being executed for easier debugging.
    print("Running locally:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def _run_pytest_in_docker(
    project_root: str, image_tag: str, pytest_target: str, run_extras: List[str] | None = None
) -> int:
    workdir = "/workspace"
    # Host HOME logs directory to map into container
    host_home = os.path.expanduser("~")
    host_logs_dir = os.path.join(host_home, ".eval_protocol")
    try:
        os.makedirs(host_logs_dir, exist_ok=True)
    except Exception:
        pass
    # Mount read-only is safer; but tests may write artifacts. Use read-write.
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{project_root}:{workdir}",
        "-v",
        f"{host_logs_dir}:/container_home/.eval_protocol",
        "-e",
        "HOME=/container_home",
        "-e",
        "EVAL_PROTOCOL_DIR=/container_home/.eval_protocol",
        "-w",
        workdir,
    ]

    # If EP_SUMMARY_JSON is set on the host, mirror it into the container so that
    # pytest evaluation tests can write summary artifacts that are visible to the
    # host. We map paths under the host logs directory (~/.eval_protocol) into the
    # mounted container home directory.
    host_summary_path = os.environ.get("EP_SUMMARY_JSON")
    if host_summary_path:
        try:
            rel_path = os.path.relpath(host_summary_path, host_logs_dir)
            # Only forward the variable when the summary path is inside the logs dir.
            if not rel_path.startswith(os.pardir):
                container_summary_path = os.path.join("/container_home/.eval_protocol", rel_path)
                cmd += ["-e", f"EP_SUMMARY_JSON={container_summary_path}"]
        except Exception:
            # Best-effort only; do not fail docker execution if we can't map the path.
            pass
    # Try to match host user to avoid permission problems on mounted volume
    try:
        uid = os.getuid()  # type: ignore[attr-defined]
        gid = os.getgid()  # type: ignore[attr-defined]
        cmd += ["--user", f"{uid}:{gid}"]
    except Exception:
        pass
    if run_extras:
        cmd += run_extras

    # Build pytest command, always enforcing the same small success threshold as
    # the host runner so that all-zero score runs fail consistently.
    pytest_cmd: list[str] = ["pytest", "--ep-success-threshold", "0.001", pytest_target, "-vs"]

    cmd += [image_tag] + pytest_cmd
    print("Running in Docker:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except FileNotFoundError:
        print("Error: docker not found in PATH. Install Docker or use --ignore-docker.")
        return 1


def run_evaluator_test(
    project_root: str,
    pytest_target: str,
    ignore_docker: bool,
    docker_build_extra: str = "",
    docker_run_extra: str = "",
) -> int:
    """Run an evaluator test either on host or in Docker, reusing local-test logic."""
    build_extras = shlex.split(docker_build_extra) if docker_build_extra else []
    run_extras = shlex.split(docker_run_extra) if docker_run_extra else []

    if ignore_docker:
        if not pytest_target:
            print("Error: Failed to resolve a pytest target to run.")
            return 1
        return _run_pytest_host(pytest_target)

    dockerfiles = _find_dockerfiles(project_root)
    if len(dockerfiles) > 1:
        print("Error: Multiple Dockerfiles found. Only one Dockerfile is allowed for evaluator validation/local-test.")
        for df in dockerfiles:
            print(f" - {df}")
        print("Hint: or use --ignore-docker to bypass Docker and use local pytest.")
        return 1
    if len(dockerfiles) == 1:
        # Ensure host home logs directory exists so container writes are visible to host ep logs
        try:
            os.makedirs(os.path.join(os.path.expanduser("~"), ".eval_protocol"), exist_ok=True)
        except Exception:
            pass
        image_tag = "ep-evaluator:local"
        ok = _build_docker_image(dockerfiles[0], image_tag, build_extras=build_extras)
        if not ok:
            print("Docker build failed. See logs above.")
            return 1
        if not pytest_target:
            print("Error: Failed to resolve a pytest target to run.")
            return 1
        return _run_pytest_in_docker(project_root, image_tag, pytest_target, run_extras=run_extras)

    # No Dockerfile: run on host
    if not pytest_target:
        print("Error: Failed to resolve a pytest target to run.")
        return 1
    return _run_pytest_host(pytest_target)


def local_test_command(args: argparse.Namespace) -> int:
    project_root = os.getcwd()

    # Selection and pytest target resolution
    pytest_target: str = ""
    entry = getattr(args, "entry", None)
    if entry:
        if "::" in entry:
            file_part, func_part = entry.split("::", 1)
            file_path = (
                file_part if os.path.isabs(file_part) else os.path.abspath(os.path.join(project_root, file_part))
            )
            pytest_target = _build_entry_point(project_root, file_path, func_part)
        else:
            file_path = entry if os.path.isabs(entry) else os.path.abspath(os.path.join(project_root, entry))
            # Use path relative to project_root when possible
            try:
                rel = os.path.relpath(file_path, project_root)
            except Exception:
                rel = file_path
            pytest_target = rel
    else:
        non_interactive = bool(getattr(args, "yes", False))
        selected = _discover_and_select_tests(project_root, non_interactive=non_interactive)
        if not selected:
            return 1
        if len(selected) != 1:
            print("Error: Please select exactly one evaluation test for 'local-test'.")
            return 1
        chosen = selected[0]
        func_name = chosen.qualname.split(".")[-1]
        pytest_target = _build_entry_point(project_root, chosen.file_path, func_name)

    ignore_docker = bool(getattr(args, "ignore_docker", False))
    build_extras_str = getattr(args, "docker_build_extra", "") or ""
    run_extras_str = getattr(args, "docker_run_extra", "") or ""
    return run_evaluator_test(
        project_root,
        pytest_target,
        ignore_docker=ignore_docker,
        docker_build_extra=build_extras_str,
        docker_run_extra=run_extras_str,
    )
