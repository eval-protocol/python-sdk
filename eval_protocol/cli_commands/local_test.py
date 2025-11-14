import argparse
import os
import subprocess
import sys
import shlex
from typing import List

from .upload import _discover_tests, _prompt_select


def _find_dockerfiles(root: str) -> List[str]:
    skip_dirs = {".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".git", "vendor"}
    dockerfiles: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for name in filenames:
            if name == "Dockerfile":
                dockerfiles.append(os.path.join(dirpath, name))
    return dockerfiles


def _run_pytest_host(pytest_target: str) -> int:
    print(f"Running locally: pytest {pytest_target} -vs")
    proc = subprocess.run([sys.executable, "-m", "pytest", pytest_target, "-vs"])
    return proc.returncode


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
    # Try to match host user to avoid permission problems on mounted volume
    try:
        uid = os.getuid()  # type: ignore[attr-defined]
        gid = os.getgid()  # type: ignore[attr-defined]
        cmd += ["--user", f"{uid}:{gid}"]
    except Exception:
        pass
    if run_extras:
        cmd += run_extras
    cmd += [image_tag, "pytest", pytest_target, "-vs"]
    print("Running in Docker:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except FileNotFoundError:
        print("Error: docker not found in PATH. Install Docker or use --ignore-docker.")
        return 1


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
            # Convert to project-relative like the non-:: path
            try:
                rel = os.path.relpath(file_path, project_root)
            except Exception:
                rel = file_path
            pytest_target = f"{rel}::{func_part}"
        else:
            file_path = entry if os.path.isabs(entry) else os.path.abspath(os.path.join(project_root, entry))
            # Use path relative to project_root when possible
            try:
                rel = os.path.relpath(file_path, project_root)
            except Exception:
                rel = file_path
            pytest_target = rel
    else:
        tests = _discover_tests(project_root)
        if not tests:
            print("No evaluation tests found.\nHint: Ensure @evaluation_test is applied.")
            return 1
        non_interactive = bool(getattr(args, "yes", False))
        selected = _prompt_select(tests, non_interactive=non_interactive)
        if not selected:
            print("No tests selected.")
            return 1
        if len(selected) != 1:
            print("Error: Please select exactly one evaluation test for 'local-test'.")
            return 1
        chosen = selected[0]
        abs_path = os.path.abspath(chosen.file_path)
        try:
            rel = os.path.relpath(abs_path, project_root)
        except Exception:
            rel = abs_path
        pytest_target = rel

    ignore_docker = bool(getattr(args, "ignore_docker", False))
    build_extras_str = getattr(args, "docker_build_extra", "") or ""
    run_extras_str = getattr(args, "docker_run_extra", "") or ""
    build_extras = shlex.split(build_extras_str) if build_extras_str else []
    run_extras = shlex.split(run_extras_str) if run_extras_str else []
    if ignore_docker:
        if not pytest_target:
            print("Error: Failed to resolve a pytest target to run.")
            return 1
        return _run_pytest_host(pytest_target)

    dockerfiles = _find_dockerfiles(project_root)
    if len(dockerfiles) > 1:
        print("Error: Multiple Dockerfiles found. Only one Dockerfile is allowed for local-test.")
        for df in dockerfiles:
            print(f" - {df}")
        print("Hint: use --ignore-docker to bypass Docker.")
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
