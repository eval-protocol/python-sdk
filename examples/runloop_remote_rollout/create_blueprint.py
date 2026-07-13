from __future__ import annotations

import argparse
import tarfile
from datetime import timedelta
from pathlib import Path

from runloop_api_client import RunloopSDK


DEFAULT_DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir ".[runloop]"
"""

IGNORED_CONTEXT_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
}


def _ignore_build_context(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = set(Path(member.name).parts)
    if parts & IGNORED_CONTEXT_DIRS:
        return None
    return member


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Runloop blueprint for the remote rollout example.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the eval-protocol repository root.",
    )
    parser.add_argument(
        "--name",
        default="eval-protocol-runloop-remote-rollout",
        help="Runloop blueprint name.",
    )
    args = parser.parse_args()

    runloop = RunloopSDK()
    build_context = runloop.storage_object.upload_from_dir(
        args.repo_root,
        name=f"{args.name}.tar.gz",
        ttl=timedelta(hours=1),
        ignore=_ignore_build_context,
    )
    blueprint = runloop.blueprint.create(
        name=args.name,
        dockerfile=DEFAULT_DOCKERFILE,
        build_context={"type": "object", "object_id": build_context.id},
    )

    print(f"export RUNLOOP_BLUEPRINT_ID={blueprint.id}")


if __name__ == "__main__":
    main()
