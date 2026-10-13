"""Exercise both E2B SDK APIs without provisioning a sandbox."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from eval_protocol.rewards import code_execution


@pytest.fixture(params=["modern", "legacy"])
def sandbox_api(request, monkeypatch):
    filesystem = Mock(spec=["make_dir", "write"])
    commands = Mock(spec=["run"])
    attributes = {"commands": commands}
    attributes["files" if request.param == "modern" else "_filesystem"] = filesystem
    sandbox = SimpleNamespace(**attributes)
    context = MagicMock()
    context.__enter__.return_value = sandbox
    constructor = Mock(return_value=context)
    factory = Mock(return_value=context)

    class Sandbox:
        def __new__(cls, **kwargs):
            return constructor(**kwargs)

    if request.param == "modern":
        Sandbox.create = factory

    monkeypatch.setattr(code_execution, "Sandbox", Sandbox, raising=False)
    monkeypatch.setattr(code_execution, "_HAS_E2B", True)
    monkeypatch.setattr(code_execution, "_E2B_SOURCE", "e2b")
    return SimpleNamespace(
        filesystem=filesystem,
        commands=commands,
        context=context,
        constructor=constructor,
        factory=factory,
        modern=request.param == "modern",
    )


@pytest.mark.parametrize(
    "language,code,path,command,output",
    [
        ("python", "print(5)", "/code/script.py", "python3 /code/script.py", "5"),
        ("javascript", "console.log(true)", "/code/script.js", "node /code/script.js", "true"),
    ],
)
def test_execute_with_sdk_factory(sandbox_api, language, code, path, command, output):
    def run(cmd, on_stdout, on_stderr, timeout):
        assert cmd == command
        assert timeout == 15
        on_stdout(SimpleNamespace(line=output))
        return SimpleNamespace(exit_code=0)

    sandbox_api.commands.run.side_effect = run
    result = code_execution.execute_code_with_e2b(code, language=language, timeout=15, api_key="test-key")

    assert result == {"success": True, "output": output, "error": None}
    sandbox_api.filesystem.write.assert_called_once_with(path, code)
    chosen = sandbox_api.factory if sandbox_api.modern else sandbox_api.constructor
    unused = sandbox_api.constructor if sandbox_api.modern else sandbox_api.factory
    chosen.assert_called_once_with(api_key="test-key")
    unused.assert_not_called()
    sandbox_api.context.__exit__.assert_called_once()


def test_write_failure_cleans_up_sandbox(sandbox_api):
    sandbox_api.filesystem.write.side_effect = RuntimeError("missing sandbox access token")
    result = code_execution.execute_code_with_e2b("print(5)", api_key="test-key")
    assert result["success"] is False
    assert "missing sandbox access token" in result["error"]
    sandbox_api.commands.run.assert_not_called()
    sandbox_api.context.__exit__.assert_called_once()


def test_command_failure_cleans_up_sandbox(sandbox_api):
    sandbox_api.commands.run.side_effect = RuntimeError("command timed out")
    result = code_execution.execute_code_with_e2b("print(5)", api_key="test-key")
    assert result["success"] is False
    assert "command timed out" in result["error"]
    sandbox_api.context.__exit__.assert_called_once()
