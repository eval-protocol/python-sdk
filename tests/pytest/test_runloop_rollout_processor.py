import asyncio
from types import SimpleNamespace

import pytest

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.runloop_rollout_processor import RunloopRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
import eval_protocol.pytest.runloop_rollout_processor as runloop_rollout_processor_module


class FakeRemoteRolloutProcessor:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.cleanup_called = False
        self.acleanup_called = False
        FakeRemoteRolloutProcessor.instances.append(self)

    def __call__(self, rows, config):
        self.calls.append((rows, config))

        async def _return_row(row):
            return row

        return [asyncio.create_task(_return_row(row)) for row in rows]

    async def acleanup(self):
        self.acleanup_called = True

    def cleanup(self):
        self.cleanup_called = True


class FakeCommandInterface:
    def __init__(self, state):
        self._state = state

    def exec_async(self, command):
        self._state.server_commands.append(command)
        return SimpleNamespace(execution_id="exec-1")


class FakeNetworkInterface:
    def __init__(self, state):
        self._state = state

    def create_tunnel(self, *, port):
        self._state.tunnel_ports.append(port)
        return SimpleNamespace(tunnel_key=self._state.tunnel_key)


class FakeDevbox:
    def __init__(self, state, devbox_id):
        self._state = state
        self.id = devbox_id
        self.cmd = FakeCommandInterface(state)
        self.net = FakeNetworkInterface(state)

    def await_running(self):
        self._state.await_running_calls += 1

    def get_tunnel_url(self, port):
        return f"https://{port}-{self._state.tunnel_key}.tunnel.runloop.ai"

    def shutdown(self):
        self._state.shutdown_calls.append(self.id)


class FakeDevboxOps:
    def __init__(self, state):
        self._state = state

    def create_from_blueprint_id(self, blueprint_id):
        self._state.created_blueprints.append(blueprint_id)
        return FakeDevbox(self._state, "devbox-created")

    def from_id(self, devbox_id):
        self._state.attached_devboxes.append(devbox_id)
        return FakeDevbox(self._state, devbox_id)


class FakeRunloopSDK:
    def __init__(self, state, bearer_token):
        self._state = state
        self._state.bearer_tokens.append(bearer_token)
        self.devbox = FakeDevboxOps(state)


@pytest.fixture
def fake_runloop(monkeypatch):
    state = SimpleNamespace(
        bearer_tokens=[],
        created_blueprints=[],
        attached_devboxes=[],
        tunnel_ports=[],
        server_commands=[],
        shutdown_calls=[],
        await_running_calls=0,
        tunnel_key="test-tunnel-key",
    )

    def _load_sdk():
        class BoundFakeRunloopSDK(FakeRunloopSDK):
            def __init__(self, bearer_token):
                super().__init__(state, bearer_token)

        return BoundFakeRunloopSDK

    FakeRemoteRolloutProcessor.instances.clear()
    monkeypatch.setattr(runloop_rollout_processor_module, "_load_runloop_sdk", _load_sdk)
    monkeypatch.setattr(runloop_rollout_processor_module, "RemoteRolloutProcessor", FakeRemoteRolloutProcessor)
    monkeypatch.setenv("RUNLOOP_API_KEY", "runloop-key")
    return state


def _config():
    return RolloutProcessorConfig(completion_params={}, mcp_config_path="", semaphore=asyncio.Semaphore(10))


def test_setup_creates_devbox_from_blueprint_and_starts_server(fake_runloop):
    processor = RunloopRolloutProcessor(
        blueprint_id="bp-123",
        server_command="python -m uvicorn server:app --host 0.0.0.0 --port 9000",
        port=9000,
        startup_timeout_seconds=0,
    )

    processor.setup()

    assert fake_runloop.bearer_tokens == ["runloop-key"]
    assert fake_runloop.created_blueprints == ["bp-123"]
    assert fake_runloop.attached_devboxes == []
    assert fake_runloop.await_running_calls == 1
    assert fake_runloop.tunnel_ports == [9000]
    assert fake_runloop.server_commands == ["python -m uvicorn server:app --host 0.0.0.0 --port 9000"]
    assert processor.devbox_id == "devbox-created"
    assert processor.remote_base_url == "https://9000-test-tunnel-key.tunnel.runloop.ai"


def test_setup_uses_existing_devbox_without_shutting_it_down(fake_runloop):
    processor = RunloopRolloutProcessor(
        devbox_id="devbox-existing",
        server_command="python server.py",
        startup_timeout_seconds=0,
    )

    processor.setup()
    processor.cleanup()

    assert fake_runloop.created_blueprints == []
    assert fake_runloop.attached_devboxes == ["devbox-existing"]
    assert fake_runloop.shutdown_calls == []


@pytest.mark.asyncio
async def test_delegates_rows_and_config_to_remote_rollout_processor(fake_runloop):
    processor = RunloopRolloutProcessor(
        blueprint_id="bp-123",
        server_command="python server.py",
        port=7000,
        model_base_url="https://example.test/tracing",
        poll_interval=2.5,
        timeout_seconds=300,
        include_payloads=True,
        startup_timeout_seconds=0,
    )
    processor.setup()

    rows = [EvaluationRow()]
    config = _config()
    tasks = processor(rows, config)
    results = await asyncio.gather(*tasks)

    remote = FakeRemoteRolloutProcessor.instances[0]
    assert results == rows
    assert remote.kwargs == {
        "remote_base_url": "https://7000-test-tunnel-key.tunnel.runloop.ai",
        "model_base_url": "https://example.test/tracing",
        "poll_interval": 2.5,
        "timeout_seconds": 300,
        "include_payloads": True,
    }
    assert remote.calls == [(rows, config)]


def test_setup_requires_runloop_api_key(monkeypatch):
    monkeypatch.delenv("RUNLOOP_API_KEY", raising=False)

    processor = RunloopRolloutProcessor(blueprint_id="bp-123", server_command="python server.py")

    with pytest.raises(ValueError, match="RUNLOOP_API_KEY"):
        processor.setup()


def test_setup_reports_missing_runloop_dependency(monkeypatch):
    monkeypatch.setenv("RUNLOOP_API_KEY", "runloop-key")

    def _raise_missing_dependency():
        raise ImportError(
            "RunloopRolloutProcessor requires the optional Runloop dependency. "
            "Install it with `pip install 'eval-protocol[runloop]'`."
        )

    monkeypatch.setattr(runloop_rollout_processor_module, "_load_runloop_sdk", _raise_missing_dependency)
    processor = RunloopRolloutProcessor(blueprint_id="bp-123", server_command="python server.py")

    with pytest.raises(ImportError, match="eval-protocol\\[runloop\\]"):
        processor.setup()


@pytest.mark.asyncio
async def test_async_cleanup_closes_remote_processor_and_owned_devbox(fake_runloop):
    processor = RunloopRolloutProcessor(
        blueprint_id="bp-123",
        server_command="python server.py",
        startup_timeout_seconds=0,
    )
    processor.setup()

    await processor.acleanup()
    await processor.acleanup()

    remote = FakeRemoteRolloutProcessor.instances[0]
    assert remote.acleanup_called is True
    assert fake_runloop.shutdown_calls == ["devbox-created"]
