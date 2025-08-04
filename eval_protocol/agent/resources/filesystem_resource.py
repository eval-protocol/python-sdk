"""
FileSystemResource: A ForkableResource for managing a directory structure as state.
"""

import os
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..resource_abc import ForkableResource


class FileSystemResource(ForkableResource):
    """
    A ForkableResource that manages a directory and its contents as its state.

    Allows for initializing a directory structure, forking it (deep copy),
    checkpointing (archiving to tar.gz), and restoring. File system operations
    can be performed via the step() method.

    Attributes:
        _config (Dict[str, Any]): Configuration for the resource.
        _managed_dir_path (Optional[Path]): Path to the root of the managed directory.
        _base_managed_dir_path (Optional[Path]): Path to the initially set up directory,
                                                 used as a template for forking.
        _temp_base_dir (Path): Base directory to store all managed directories and checkpoints.
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._managed_dir_path: Optional[Path] = None
        self._base_managed_dir_path: Optional[Path] = None  # Stores the path of the initial setup
        self._temp_base_dir = Path("./.rk_temp_fs").resolve()  # Resolve to absolute path
        self._temp_base_dir.mkdir(parents=True, exist_ok=True)

    def _get_new_managed_path(self, prefix: str = "fs_") -> Path:
        """Generates a new unique path within the temp base directory."""
        return self._temp_base_dir / f"{prefix}{uuid.uuid4().hex}"

    async def setup(self, config: Dict[str, Any]) -> None:
        """
        Initializes the managed directory structure.

        Args:
            config: Configuration dictionary. Expected keys:
                - 'base_dir_name' (Optional[str]): Name for the root managed directory.
                                                   Defaults to a UUID.
                - 'initial_files' (Optional[Dict[str, str]]):
                    A dictionary where keys are relative file paths within the
                    managed directory and values are their content.
                    Example: {"subdir/file.txt": "Hello", "root_file.py": "print('world')"}
        """
        self._config = config.copy()

        base_dir_name = self._config.get("base_dir_name", f"fs_base_{uuid.uuid4().hex}")
        self._base_managed_dir_path = self._temp_base_dir / base_dir_name
        self._managed_dir_path = self._base_managed_dir_path  # Initially, current is base

        if self._base_managed_dir_path is not None and self._base_managed_dir_path.exists():
            shutil.rmtree(self._base_managed_dir_path)  # Clean start
        if self._base_managed_dir_path is not None:
            self._base_managed_dir_path.mkdir(parents=True)

        initial_files = self._config.get("initial_files", {})
        for rel_path_str, content in initial_files.items():
            if self._base_managed_dir_path is not None:
                abs_path = self._base_managed_dir_path / Path(rel_path_str)
            else:
                raise ValueError("Base managed directory path is not set")
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)

    async def fork(self) -> "FileSystemResource":
        """
        Creates a new FileSystemResource instance with a deep copy of the
        current managed directory state.
        """
        if not self._managed_dir_path or not self._managed_dir_path.exists():
            raise RuntimeError("Cannot fork: managed directory does not exist or setup was not called.")

        forked_resource = FileSystemResource()
        forked_resource._config = self._config.copy()
        forked_resource._temp_base_dir = self._temp_base_dir

        # The new fork's "base" is the current state of this resource

        forked_dir_path = self._get_new_managed_path(prefix="fs_fork_")
        shutil.copytree(self._managed_dir_path, forked_dir_path)

        forked_resource._managed_dir_path = forked_dir_path
        # The concept of _base_managed_dir_path for a fork is tricky.
        # For now, a fork doesn't have its own "base template" in the same way the first instance does.
        # It's just a live copy. If it forks again, its current state is copied.
        forked_resource._base_managed_dir_path = None  # Or perhaps self._managed_dir_path?

        return forked_resource

    async def checkpoint(self) -> Dict[str, Any]:
        """
        Creates a tar.gz archive of the current managed directory and returns its path.
        """
        if not self._managed_dir_path or not self._managed_dir_path.exists():
            raise RuntimeError("Cannot checkpoint: managed directory does not exist.")

        checkpoint_filename = f"checkpoint_fs_{self._managed_dir_path.name}_{uuid.uuid4().hex}.tar.gz"
        checkpoint_path = self._temp_base_dir / checkpoint_filename

        with tarfile.open(checkpoint_path, "w:gz") as tar:
            # Add files relative to the managed_dir_path so they extract correctly
            tar.add(str(self._managed_dir_path), arcname=".")

        return {"type": "filesystem_tar_gz", "checkpoint_path": str(checkpoint_path)}

    async def restore(self, state_data: Dict[str, Any]) -> None:
        """
        Restores the managed directory state from a tar.gz archive.
        The current managed directory will be replaced.
        """
        archive_type = state_data.get("type")
        checkpoint_path_str = state_data.get("checkpoint_path")

        if archive_type != "filesystem_tar_gz" or not checkpoint_path_str:
            raise ValueError("Invalid state_data for FileSystemResource restore.")

        checkpoint_path = Path(checkpoint_path_str)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint archive not found: {checkpoint_path}")

        if not self._managed_dir_path:
            self._managed_dir_path = self._get_new_managed_path(prefix="fs_restored_")

        if self._managed_dir_path.exists():
            shutil.rmtree(self._managed_dir_path)  # Clean before restore
        self._managed_dir_path.mkdir(parents=True)

        with tarfile.open(checkpoint_path, "r:gz") as tar:
            tar.extractall(path=str(self._managed_dir_path))

        # The restored state becomes the new "base" for subsequent forks from this instance

    # _resolve_path is a synchronous helper, no need to make it async unless it performs async I/O
    def _resolve_path(self, rel_path: Union[str, Path]) -> Path:
        """Resolves a relative path against the managed directory and ensures it's within."""
        if not self._managed_dir_path:
            raise RuntimeError("Managed directory path not set.")

        abs_path = (self._managed_dir_path / rel_path).resolve()

        # Security check: ensure the path is within the managed directory
        if self._managed_dir_path.resolve() not in abs_path.parents and abs_path != self._managed_dir_path.resolve():
            raise ValueError(f"Path '{rel_path}' attempts to access outside the managed directory.")
        return abs_path

    async def step(self, action_name: str, action_params: Dict[str, Any]) -> Any:
        """
        Performs a file system operation within the managed directory.

        Supported actions:
        - 'create_file': Creates an empty file or overwrites an existing one.
            Params: {'path': str, 'content': Optional[str]}
        - 'read_file': Reads the content of a file.
            Params: {'path': str} -> Returns: str (content)
        - 'delete_file': Deletes a file.
            Params: {'path': str}
        - 'list_dir': Lists contents of a directory.
            Params: {'path': str (relative to managed_dir), 'recursive': Optional[bool]} -> Returns: List[str]
        - 'create_dir': Creates a directory.
            Params: {'path': str}
        - 'delete_dir': Deletes a directory recursively.
            Params: {'path': str}
        """
        path_str = action_params.get("path")
        if path_str is None and action_name not in []:  # Some actions might not need a path
            raise ValueError(f"Missing 'path' in action_params for '{action_name}'.")

        abs_path = self._resolve_path(path_str) if path_str else None

        if action_name == "create_file" or action_name == "write_file":
            if not abs_path:
                raise ValueError("Path is required for create/write_file")
            content = action_params.get("content", "")
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {
                "status": "success",
                "path": (
                    str(abs_path.relative_to(self._managed_dir_path))
                    if self._managed_dir_path is not None
                    else str(abs_path)
                ),
            }

        elif action_name == "read_file":
            if not abs_path:
                raise ValueError("Path is required for read_file")
            if not abs_path.is_file():
                raise FileNotFoundError(f"File not found: {path_str}")
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read()

        elif action_name == "delete_file":
            if not abs_path:
                raise ValueError("Path is required for delete_file")
            if not abs_path.is_file():
                raise FileNotFoundError(f"File not found: {path_str}")
            abs_path.unlink()
            return {
                "status": "success",
                "path": (
                    str(abs_path.relative_to(self._managed_dir_path))
                    if self._managed_dir_path is not None
                    else str(abs_path)
                ),
            }

        elif action_name == "list_dir":
            if not abs_path:
                raise ValueError("Path is required for list_dir")
            if not abs_path.is_dir():
                raise NotADirectoryError(f"Not a directory: {path_str}")

            recursive = action_params.get("recursive", False)
            items = []
            if recursive:
                for item in abs_path.rglob("*"):
                    items.append(
                        str(item.relative_to(self._managed_dir_path))
                        if self._managed_dir_path is not None
                        else str(item)
                    )
            else:
                for item in abs_path.iterdir():
                    items.append(
                        str(item.relative_to(self._managed_dir_path))
                        if self._managed_dir_path is not None
                        else str(item)
                    )
            return items

        elif action_name == "create_dir":
            if not abs_path:
                raise ValueError("Path is required for create_dir")
            abs_path.mkdir(parents=True, exist_ok=True)
            return {
                "status": "success",
                "path": (
                    str(abs_path.relative_to(self._managed_dir_path))
                    if self._managed_dir_path is not None
                    else str(abs_path)
                ),
            }

        elif action_name == "delete_dir":
            if not abs_path:
                raise ValueError("Path is required for delete_dir")
            if not abs_path.is_dir():
                raise NotADirectoryError(f"Not a directory: {path_str}")
            if abs_path == self._managed_dir_path:  # Safety: don't delete the root managed dir itself via step
                raise ValueError("Cannot delete the root managed directory via 'delete_dir' action.")
            shutil.rmtree(abs_path)
            return {
                "status": "success",
                "path": (
                    str(abs_path.relative_to(self._managed_dir_path))
                    if self._managed_dir_path is not None
                    else str(abs_path)
                ),
            }

        else:
            raise NotImplementedError(f"Action '{action_name}' not supported by FileSystemResource.")

    async def get_observation(self) -> Dict[str, Any]:
        """
        Returns the path to the managed directory.
        """
        return {
            "type": "filesystem",
            "managed_dir_path": (str(self._managed_dir_path) if self._managed_dir_path else None),
            "status": ("ready" if self._managed_dir_path and self._managed_dir_path.exists() else "uninitialized"),
        }

    async def get_tools_spec(self) -> List[Dict[str, Any]]:
        """
        Returns tool specifications for file system operations.
        """
        # This can be extensive. For now, a few examples.
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Creates or overwrites a file with the given content within the managed directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Reads the content of a file from the managed directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file.",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "Lists files and directories at the given relative path within the managed directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the directory.",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "List recursively (default: false).",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            # Add more tools: delete_file, create_dir, delete_dir etc.
        ]

    async def close(self) -> None:
        """
        Cleans up by deleting the managed directory and any checkpoints if they are temporary.
        For now, it only deletes the current _managed_dir_path.
        A more robust strategy for _temp_base_dir cleanup is needed for production.
        """
        if self._managed_dir_path and self._managed_dir_path.exists():
            try:
                shutil.rmtree(self._managed_dir_path)
            except OSError as e:
                print(f"Error deleting managed directory {self._managed_dir_path}: {e}")

        # self._base_managed_dir_path might also need cleanup if it's different and temporary.
        self._managed_dir_path = None
        self._base_managed_dir_path = None
        # Deleting _temp_base_dir itself could be too aggressive if it holds checkpoints
        # or other active resources.
