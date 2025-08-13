"""
Tests for V2 ForkableResource implementations.
"""

import asyncio  # For async tests
import copy
import pickle
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict  # Added Dict, Any
from unittest.mock import AsyncMock  # For mocking async methods if needed later

import pytest
import pytest_asyncio  # Import pytest_asyncio

from eval_protocol.agent.resources.docker_resource import (
    DOCKER_DAEMON_AVAILABLE,
    DOCKER_SDK_AVAILABLE,
    DockerResource,
)
from eval_protocol.agent.resources.filesystem_resource import FileSystemResource
from eval_protocol.agent.resources.python_state_resource import PythonStateResource
from eval_protocol.agent.resources.sql_resource import SQLResource


@pytest.mark.asyncio
class TestPythonStateResource:
    """Tests for the PythonStateResource."""

    async def test_setup_initial_state(self):
        config: Dict[str, Any] = {"initial_state": {"key1": "value1", "count": 10}}
        resource = PythonStateResource()
        await resource.setup(config)
        observation = await resource.get_observation()
        assert observation == {"key1": "value1", "count": 10}
        assert resource._state is not config["initial_state"]

    async def test_setup_empty_config(self):
        config: Dict[str, Any] = {}
        resource = PythonStateResource()
        await resource.setup(config)
        assert await resource.get_observation() == {}

    async def test_fork_state_copy(self):
        initial_state = {"data": [1, 2], "nested": {"a": "b"}}
        config = {"initial_state": initial_state}
        original_resource = PythonStateResource()
        await original_resource.setup(config)
        forked_resource = await original_resource.fork()
        assert await forked_resource.get_observation() == initial_state
        assert forked_resource._state is not original_resource._state
        assert forked_resource._state["data"] is not original_resource._state["data"]
        assert forked_resource._state["nested"] is not original_resource._state["nested"]

    async def test_fork_independent_state(self):
        config = {"initial_state": {"key": "original_value", "num": 1}}
        original_resource = PythonStateResource()
        await original_resource.setup(config)
        forked_resource = await original_resource.fork()
        await forked_resource.step("update_state", {"key": "forked_value"})
        await forked_resource.step("update_state", {"num": 2})
        assert (await original_resource.get_observation())["key"] == "original_value"
        assert (await original_resource.get_observation())["num"] == 1
        assert (await forked_resource.get_observation())["key"] == "forked_value"
        assert (await forked_resource.get_observation())["num"] == 2
        await original_resource.step("update_state", {"key": "new_original_value"})
        await original_resource.step("update_state", {"num": 0})
        assert (await forked_resource.get_observation())["key"] == "forked_value"
        assert (await original_resource.get_observation())["key"] == "new_original_value"

    async def test_checkpoint_and_restore(self):
        initial_config = {"initial_state": {"val": 42, "items": ["a", "b"]}}
        resource1 = PythonStateResource()
        await resource1.setup(initial_config)
        await resource1.step("update_state", {"val": 43, "new_item": "c"})
        checkpoint_data = await resource1.checkpoint()
        assert isinstance(checkpoint_data, bytes)
        resource2 = PythonStateResource()
        await resource2.setup({"initial_state": {"val": 0, "items": []}})
        await resource2.restore(checkpoint_data)
        expected_state = {"val": 43, "items": ["a", "b"], "new_item": "c"}
        assert await resource2.get_observation() == expected_state
        assert await resource1.get_observation() == expected_state
        with pytest.raises((pickle.UnpicklingError, TypeError, AttributeError, ValueError)):
            await resource2.restore(b"invalid pickle data")

    async def test_step_update_state(self):
        resource = PythonStateResource()
        await resource.setup({"initial_state": {"a": 1}})
        result = await resource.step("update_state", {"b": 2, "a": 3})
        expected_state = {"a": 3, "b": 2}
        assert await resource.get_observation() == expected_state
        assert result == expected_state

    async def test_step_get_value(self):
        resource = PythonStateResource()
        await resource.setup({"initial_state": {"name": "test", "value": 100}})
        assert await resource.step("get_value", {"key": "name"}) == "test"
        assert await resource.step("get_value", {"key": "value"}) == 100
        assert await resource.step("get_value", {"key": "nonexistent"}) is None
        with pytest.raises(ValueError, match="Missing 'key'"):
            await resource.step("get_value", {})

    async def test_step_unknown_action(self):
        resource = PythonStateResource()
        await resource.setup({})
        with pytest.raises(NotImplementedError):
            await resource.step("non_existent_action", {})

    async def test_get_observation_is_copy(self):
        resource = PythonStateResource()
        initial_state = {"list_key": [1, 2]}
        await resource.setup({"initial_state": initial_state})
        obs1 = await resource.get_observation()
        obs1["list_key"].append(3)
        assert (await resource.get_observation())["list_key"] == [1, 2]

    async def test_get_tools_spec(self):
        resource = PythonStateResource()
        specs = await resource.get_tools_spec()
        assert isinstance(specs, list)
        assert len(specs) == 2
        tool_names = [spec["function"]["name"] for spec in specs]
        assert "update_state" in tool_names
        assert "get_value" in tool_names

    async def test_close_clears_state(self):
        resource = PythonStateResource()
        await resource.setup({"initial_state": {"a": 1}})
        assert await resource.get_observation() == {"a": 1}
        await resource.close()
        assert await resource.get_observation() == {}
        assert resource._config == {}
        assert resource._state == {}


@pytest.mark.asyncio
class TestSQLResource:
    """Tests for the SQLResource."""

    async def test_setup_sqlite_with_schema_file_and_seed_file(self, tmp_path):
        schema_content = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        seed_content = "INSERT INTO users (name) VALUES ('Alice'); INSERT INTO users (name) VALUES ('Bob');"
        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(schema_content)
        seed_file = tmp_path / "seed.sql"
        seed_file.write_text(seed_content)
        config = {
            "db_type": "sqlite",
            "db_name": "test_setup.sqlite",
            "schema_file": str(schema_file),
            "seed_data_file": str(seed_file),
        }
        resource = SQLResource()
        try:
            await resource.setup(config)
            assert resource._db_path is not None and resource._db_path.exists()
            conn = sqlite3.connect(resource._db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            assert cursor.fetchone()[0] == 2
            conn.close()
        finally:
            await resource.close()

    async def test_fork_sqlite(self):
        config = {
            "db_type": "sqlite",
            "db_name": "original.sqlite",
            "schema_sql": "CREATE TABLE data (val TEXT);",
        }
        original_resource = SQLResource()
        try:
            await original_resource.setup(config)
            await original_resource.step("execute_sql", {"query": "INSERT INTO data (val) VALUES ('initial')"})
            forked_resource = await original_resource.fork()
            try:
                assert forked_resource._db_path is not None and forked_resource._db_path.exists()
                res_fork = await forked_resource.step(
                    "execute_sql",
                    {"query": "SELECT val FROM data", "fetch_mode": "one"},
                )
                assert res_fork["val"] == "initial"
                await forked_resource.step(
                    "execute_sql",
                    {"query": "UPDATE data SET val = 'forked_changed' WHERE val = 'initial'"},
                )
                res_orig = await original_resource.step(
                    "execute_sql",
                    {"query": "SELECT val FROM data", "fetch_mode": "one"},
                )
                assert res_orig["val"] == "initial"
            finally:
                await forked_resource.close()
        finally:
            await original_resource.close()

    async def test_checkpoint_and_restore_sqlite(self):
        config = {
            "db_type": "sqlite",
            "db_name": "chkpt_orig.sqlite",
            "schema_sql": "CREATE TABLE log (msg TEXT);",
        }
        res1 = SQLResource()
        try:
            await res1.setup(config)
            await res1.step("execute_sql", {"query": "INSERT INTO log (msg) VALUES ('message1')"})
            checkpoint_info = await res1.checkpoint()
            checkpoint_file = Path(checkpoint_info["checkpoint_path"])
            assert checkpoint_file.exists()
            res2 = SQLResource()
            try:
                await res2.setup({"db_type": "sqlite", "db_name": "chkpt_res2_init.sqlite"})
                await res2.restore(checkpoint_info)
                count_res2 = await res2.step(
                    "execute_sql",
                    {"query": "SELECT COUNT(*) FROM log", "fetch_mode": "val"},
                )
                assert count_res2 == 1
            finally:
                await res2.close()
            if checkpoint_file.exists():
                checkpoint_file.unlink()
        finally:
            await res1.close()

    async def test_close_sqlite(self):
        config = {"db_type": "sqlite", "db_name": "closetest.sqlite"}
        resource = SQLResource()
        await resource.setup(config)
        db_path = resource._db_path
        assert db_path is not None and db_path.exists()
        await resource.close()
        assert not db_path.exists()


@pytest.mark.asyncio
class TestFileSystemResource:
    """Tests for the FileSystemResource."""

    async def test_setup_with_initial_files(self):
        config = {
            "initial_files": {
                "file1.txt": "Hello World",
                "subdir/file2.py": "print('Python code')",
            }
        }
        resource = FileSystemResource()
        try:
            await resource.setup(config)
            assert resource._managed_dir_path is not None and resource._managed_dir_path.exists()
            # Add assertion for mypy
            assert resource._managed_dir_path is not None
            assert (resource._managed_dir_path / "file1.txt").read_text() == "Hello World"
            assert (resource._managed_dir_path / "subdir" / "file2.py").read_text() == "print('Python code')"
        finally:
            await resource.close()

    async def test_fork_filesystem(self):
        config = {"initial_files": {"original.txt": "original content"}}
        original_resource = FileSystemResource()
        try:
            await original_resource.setup(config)
            forked_resource = await original_resource.fork()
            try:
                assert forked_resource._managed_dir_path is not None and forked_resource._managed_dir_path.exists()
                # Add assertion for mypy
                assert forked_resource._managed_dir_path is not None
                assert (forked_resource._managed_dir_path / "original.txt").read_text() == "original content"
                (forked_resource._managed_dir_path / "forked_new.txt").write_text("forked specific")
                # Add assertion for mypy
                assert original_resource._managed_dir_path is not None
                assert not (original_resource._managed_dir_path / "forked_new.txt").exists()
            finally:
                await forked_resource.close()
        finally:
            await original_resource.close()

    async def test_checkpoint_and_restore_filesystem(self, tmp_path):
        config = {"initial_files": {"data.txt": "checkpoint me"}}
        res1 = FileSystemResource()
        try:
            await res1.setup(config)
            checkpoint_info = await res1.checkpoint()
            checkpoint_archive = Path(checkpoint_info["checkpoint_path"])
            assert checkpoint_archive.exists()
            res2 = FileSystemResource()
            try:
                await res2.setup({"base_dir_name": "fs_res2_init"})
                await res2.restore(checkpoint_info)
                assert res2._managed_dir_path is not None  # Ensure for mypy
                assert (res2._managed_dir_path / "data.txt").read_text() == "checkpoint me"
            finally:
                await res2.close()
            if checkpoint_archive.exists():
                checkpoint_archive.unlink()
        finally:
            await res1.close()

    async def test_close_filesystem(self):
        resource = FileSystemResource()
        await resource.setup({"initial_files": {"a.txt": "content"}})
        managed_path = resource._managed_dir_path
        assert managed_path is not None and managed_path.exists()
        await resource.close()
        assert not managed_path.exists()


pytestmark_docker = pytest.mark.skipif(
    not DOCKER_SDK_AVAILABLE or not DOCKER_DAEMON_AVAILABLE,
    reason="Docker SDK not installed or Docker daemon not running",
)


@pytestmark_docker
@pytest.mark.asyncio
class TestDockerResource:
    """Tests for the DockerResource."""

    TEST_IMAGE = "alpine:latest"

    @pytest_asyncio.fixture(scope="function")  # Changed to pytest_asyncio.fixture
    async def docker_resource(self):  # Made async fixture
        resource = DockerResource()
        try:
            client = resource._client
            await asyncio.to_thread(client.images.get, self.TEST_IMAGE)
        except Exception:
            try:
                print(f"\nPulling Docker test image: {self.TEST_IMAGE}...")
                await asyncio.to_thread(client.images.pull, self.TEST_IMAGE)
                print("Pull complete.")
            except Exception as e:
                pytest.skip(f"Failed to pull Docker image {self.TEST_IMAGE}: {e}")
        yield resource
        await resource.close()

    async def test_setup_docker_container(self, docker_resource: DockerResource):
        config = {
            "image_name": self.TEST_IMAGE,
            "docker_run_options": {"command": ["tail", "-f", "/dev/null"]},
        }
        await docker_resource.setup(config)
        assert docker_resource._container is not None
        # Docker SDK calls are blocking, wrap in to_thread for async tests
        container_status = await asyncio.to_thread(getattr, docker_resource._container, "status")
        assert container_status == "running"
        result = await docker_resource.step("exec_command", {"command": "echo hello"})
        assert result["exit_code"] == 0 and "hello" in result["output"]

    async def test_fork_docker_container(self, docker_resource: DockerResource):
        await docker_resource.setup(
            {
                "image_name": self.TEST_IMAGE,
                "docker_run_options": {"command": ["tail", "-f", "/dev/null"]},
            }
        )
        await docker_resource.step("exec_command", {"command": "touch /tmp/original_file.txt"})
        forked_resource = await docker_resource.fork()
        try:
            assert forked_resource._container is not None
            res_fork = await forked_resource.step("exec_command", {"command": "ls /tmp/original_file.txt"})
            assert res_fork["exit_code"] == 0
        finally:
            await forked_resource.close()

    async def test_checkpoint_and_restore_docker(self, docker_resource: DockerResource):
        config = {
            "image_name": self.TEST_IMAGE,
            "docker_run_options": {"command": ["tail", "-f", "/dev/null"]},
        }
        await docker_resource.setup(config)
        create_file_command = "sh -c \"echo 'initial_data' > /data.txt\""
        create_file_result = await docker_resource.step("exec_command", {"command": create_file_command})
        assert create_file_result["exit_code"] == 0, (
            f"Failed to create /data.txt with '{create_file_command}': {create_file_result['output']}"
        )

        # Optionally, verify file content immediately after creation in the source container
        verify_result = await docker_resource.step("exec_command", {"command": "cat /data.txt"})
        assert verify_result["exit_code"] == 0, f"Failed to cat /data.txt after creation: {verify_result['output']}"
        assert "initial_data" in verify_result["output"], (
            f"/data.txt content mismatch after creation: {verify_result['output']}"
        )

        checkpoint_info = await docker_resource.checkpoint()
        checkpoint_image_id = checkpoint_info["image_id"]
        restored_resource = DockerResource()
        try:
            await restored_resource.setup(config)
            await restored_resource.restore(checkpoint_info)
            result = await restored_resource.step("exec_command", {"command": "cat /data.txt"})
            assert "initial_data" in result["output"]
        finally:
            await restored_resource.close()
            # Note: The original image_name from config is self.TEST_IMAGE.
            # The committed image (checkpoint_image_id) will be different.
            # We should always try to clean up the checkpoint_image_id if it exists.
            if checkpoint_image_id:
                try:
                    # Check if it's the base image to avoid removing it if tests are re-run without pulling
                    base_image_obj = await asyncio.to_thread(docker_resource._client.images.get, self.TEST_IMAGE)
                    if base_image_obj.id != checkpoint_image_id:
                        await asyncio.to_thread(
                            docker_resource._client.images.remove,
                            image=checkpoint_image_id,
                            force=True,
                        )
                except Exception as e:  # Catch NotFound from get or APIError from remove
                    print(f"Warning: failed to cleanup checkpoint image {checkpoint_image_id}: {e}")

    async def test_close_docker_resource(self, docker_resource: DockerResource):
        await docker_resource.setup(
            {
                "image_name": self.TEST_IMAGE,
                "docker_run_options": {"command": ["tail", "-f", "/dev/null"]},
            }
        )
        assert docker_resource._container is not None  # Ensure container exists before accessing id
        container_id = docker_resource._container.id
        await docker_resource.close()
        assert docker_resource._container is None
        with pytest.raises(Exception):
            await asyncio.to_thread(docker_resource._client.containers.get, container_id)
