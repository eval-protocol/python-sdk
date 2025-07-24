"""
SQLResource: A ForkableResource for managing SQL database states, initially focusing on SQLite.
"""

import os
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..resource_abc import ForkableResource


class SQLResource(ForkableResource):
    """
    A ForkableResource for managing SQL database states, primarily SQLite.

    Manages a SQLite database file, allowing it to be initialized with a schema
    and seed data, forked (by copying the DB file), checkpointed (by copying),
    and restored.

    Attributes:
        _config (Dict[str, Any]): Configuration for the resource.
        _db_path (Optional[Path]): Path to the current SQLite database file.
        _base_db_path (Optional[Path]): Path to the initially set up database, used for forking.
        _temp_dir (Path): Directory to store database files.
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._db_path: Optional[Path] = None
        self._base_db_path: Optional[Path] = None
        # Consider making temp_dir configurable or using a more robust temp solution
        self._temp_dir = Path("./.rk_temp_dbs").resolve()  # Ensure absolute path
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    def _get_db_connection(self) -> sqlite3.Connection:
        if not self._db_path:
            raise ConnectionError("Database path not set. Call setup() or fork() first.")
        # Set timeout to prevent indefinite hangs
        return sqlite3.connect(str(self._db_path), timeout=10)

    async def setup(self, config: Dict[str, Any]) -> None:
        """
        Initializes the SQLite database.

        Args:
            config: Configuration dictionary. Expected keys:
                - 'db_type' (str): Must be 'sqlite'.
                - 'db_name' (Optional[str]): Name for the database file. Defaults to a UUID.
                - 'schema_file' (Optional[str]): Path to an SQL file to execute for schema setup.
                - 'seed_data_file' (Optional[str]): Path to an SQL file for initial data seeding.
                - 'schema_sql' (Optional[str]): SQL string for schema setup.
                - 'seed_sql' (Optional[str]): SQL string for initial data seeding.
        """
        self._config = config.copy()
        db_type = self._config.get("db_type", "sqlite")
        if db_type != "sqlite":
            raise ValueError("SQLResource currently only supports 'sqlite'.")

        db_name = self._config.get("db_name", f"db_{uuid.uuid4().hex}.sqlite")
        self._base_db_path = self._temp_dir / db_name
        self._db_path = self._base_db_path  # Initially, the current DB is the base DB

        # Ensure a fresh start if the base DB file already exists from a previous run
        if self._base_db_path is not None and self._base_db_path.exists():
            self._base_db_path.unlink()

        conn = self._get_db_connection()
        try:
            with conn:
                # Apply schema
                schema_file = self._config.get("schema_file")
                if schema_file and Path(schema_file).exists():
                    with open(schema_file, "r") as f:
                        conn.executescript(f.read())

                schema_sql = self._config.get("schema_sql")
                if schema_sql:
                    conn.executescript(schema_sql)

                # Apply seed data
                seed_data_file = self._config.get("seed_data_file")
                if seed_data_file and Path(seed_data_file).exists():
                    with open(seed_data_file, "r") as f:
                        conn.executescript(f.read())

                seed_sql = self._config.get("seed_sql")
                if seed_sql:
                    conn.executescript(seed_sql)
        finally:
            conn.close()

    async def fork(self) -> "SQLResource":
        """
        Creates a new SQLResource instance with a copy of the base database state.
        If called on an already forked resource, it forks from its current state.
        """
        if not self._db_path or not self._db_path.exists():
            raise RuntimeError("Cannot fork: original database does not exist or setup was not called.")

        forked_resource = SQLResource()
        forked_resource._config = self._config.copy()
        forked_resource._temp_dir = self._temp_dir  # Share the same temp dir base

        # The new fork's base is the current state of this resource
        forked_resource._base_db_path = self._db_path

        # Create a new unique DB file for this fork
        forked_db_name = f"fork_{uuid.uuid4().hex}.sqlite"
        forked_resource._db_path = self._temp_dir / forked_db_name

        shutil.copyfile(str(self._db_path), str(forked_resource._db_path))
        return forked_resource

    async def checkpoint(self) -> Dict[str, Any]:
        """
        Returns a serializable representation of the resource's current state.
        For SQLite, this involves copying the database file to a checkpoint location
        and returning the path.
        """
        if not self._db_path or not self._db_path.exists():
            raise RuntimeError("Cannot checkpoint: database does not exist.")

        checkpoint_name = f"checkpoint_{self._db_path.stem}_{uuid.uuid4().hex}.sqlite"
        checkpoint_path = self._temp_dir / checkpoint_name
        shutil.copyfile(str(self._db_path), str(checkpoint_path))
        return {"db_type": "sqlite", "checkpoint_path": str(checkpoint_path)}

    async def restore(self, state_data: Dict[str, Any]) -> None:
        """
        Restores the resource's state from a previously checkpointed state.
        For SQLite, this means copying the checkpointed DB file to become the current DB.
        """
        db_type = state_data.get("db_type")
        checkpoint_path_str = state_data.get("checkpoint_path")

        if db_type != "sqlite" or not checkpoint_path_str:
            raise ValueError("Invalid state_data for SQLite restore.")

        checkpoint_path = Path(checkpoint_path_str)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # If current db_path is not set (e.g. fresh resource), assign one
        if not self._db_path:
            self._db_path = self._temp_dir / f"restored_{uuid.uuid4().hex}.sqlite"

        shutil.copyfile(str(checkpoint_path), str(self._db_path))
        self._base_db_path = self._db_path  # The restored state becomes the new base for future forks

    async def step(self, action_name: str, action_params: Dict[str, Any]) -> Any:
        """
        Executes a SQL query on the database.

        Args:
            action_name: Should be 'execute_sql'.
            action_params: Dictionary containing:
                - 'query' (str): The SQL query to execute.
                - 'parameters' (Optional[Dict | List]): Parameters for the query.
                - 'fetch_mode' (Optional[str]): 'one', 'all', or 'val'. If None, no fetch.

        Returns:
            Query result based on fetch_mode, or rowcount for DML.
        """
        if action_name != "execute_sql":
            raise NotImplementedError(f"Action '{action_name}' not supported by SQLResource.")

        query = action_params.get("query")
        if not query:
            raise ValueError("Missing 'query' in action_params for 'execute_sql'.")

        params = action_params.get("parameters", [])
        fetch_mode = action_params.get("fetch_mode")  # 'one', 'all', 'val'

        conn = self._get_db_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

                if fetch_mode == "one":
                    columns = [desc[0] for desc in cursor.description]
                    row = cursor.fetchone()
                    return dict(zip(columns, row)) if row else None
                elif fetch_mode == "all":
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                elif fetch_mode == "val":
                    row = cursor.fetchone()
                    return row[0] if row else None
                else:  # DML or no fetch needed
                    return {"rowcount": cursor.rowcount}
        finally:
            conn.close()

    async def get_observation(self) -> Dict[str, Any]:
        """
        Returns the current observable state of the resource.
        For SQLResource, this could be the path to the DB or a status message.
        """
        return {
            "db_type": "sqlite",
            "db_path": str(self._db_path) if self._db_path else None,
            "status": ("ready" if self._db_path and self._db_path.exists() else "uninitialized"),
        }

    async def get_tools_spec(self) -> List[Dict[str, Any]]:
        """
        Returns tool specifications for interacting with the SQL database.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Executes a SQL query against the database. "
                    "Use 'fetch_mode' to control return value: "
                    "'one' for a single row, "
                    "'all' for all rows, "
                    "'val' for a single value from the first row. "
                    "If 'fetch_mode' is not provided, returns rowcount for DML statements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The SQL query to execute.",
                            },
                            "parameters": {
                                "type": "array",  # Or object for named parameters, sqlite3 supports both
                                "description": "Parameters for the SQL query (optional).",
                                "items": {"type": "any"},
                            },
                            "fetch_mode": {
                                "type": "string",
                                "enum": ["one", "all", "val"],
                                "description": "Specifies how to fetch results (optional).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def close(self) -> None:
        """
        Cleans up by deleting the created database file(s).
        More robust cleanup of the _temp_dir might be needed if it's shared or persistent.
        """
        if self._db_path and self._db_path.exists():
            try:
                self._db_path.unlink()
            except OSError as e:
                print(f"Error deleting database file {self._db_path}: {e}")

        # Potentially clean up base_db_path if it's different and also temporary
        # if self._base_db_path and self._base_db_path.exists() and self._base_db_path != self._db_path:
        #     try:
        #         self._base_db_path.unlink()
        #     except OSError:
        #         pass # ignore if it was already deleted or moved

        # For now, we don't delete the _temp_dir itself, as it might contain checkpoints
        # or other DBs from concurrent operations. A more sophisticated cleanup strategy
        # for _temp_dir might be needed for long-running processes.
        self._db_path = None
        self._base_db_path = None
