"""Runloop-backed remote rollout processor."""

from __future__ import annotations

import asyncio
import os
import time
import urllib.error
import urllib.request
from typing import Any

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig


def _load_runloop_sdk() -> Any:
    try:
        from runloop_api_client import RunloopSDK
    except ImportError as exc:
        raise ImportError(
            "RunloopRolloutProcessor requires the optional Runloop dependency. "
            "Install it with `pip install 'eval-protocol[runloop]'`."
        ) from exc
    return RunloopSDK


class RunloopRolloutProcessor(RolloutProcessor):
    """Host a remote rollout server in a Runloop Devbox.

    This processor only orchestrates Runloop lifecycle. Row processing is delegated
    to :class:`RemoteRolloutProcessor`, so completion and trace collection continue
    to use Eval Protocol's existing remote rollout contract.
    """

    def __init__(
        self,
        *,
        blueprint_id: str | None = None,
        devbox_id: str | None = None,
        server_command: str,
        port: int = 8000,
        model_base_url: str = "https://tracing.fireworks.ai",
        poll_interval: float = 1.0,
        timeout_seconds: float = 120.0,
        startup_timeout_seconds: float = 60.0,
        include_payloads: bool = False,
        shutdown_on_cleanup: bool = True,
        runloop_api_key: str | None = None,
    ) -> None:
        if not blueprint_id and not devbox_id:
            raise ValueError("Either blueprint_id or devbox_id is required for RunloopRolloutProcessor")
        if not server_command:
            raise ValueError("server_command is required for RunloopRolloutProcessor")

        self._blueprint_id = blueprint_id
        self._devbox_id = devbox_id
        self._server_command = server_command
        self._port = port
        self._model_base_url = model_base_url
        self._poll_interval = poll_interval
        self._timeout_seconds = timeout_seconds
        self._startup_timeout_seconds = startup_timeout_seconds
        self._include_payloads = include_payloads
        self._shutdown_on_cleanup = shutdown_on_cleanup
        self._runloop_api_key = runloop_api_key

        self._client: Any | None = None
        self._devbox: Any | None = None
        self._server_execution: Any | None = None
        self._remote_base_url: str | None = None
        self._remote_processor: RemoteRolloutProcessor | None = None
        self._owns_devbox = False
        self._shutdown_complete = False

    @property
    def remote_base_url(self) -> str | None:
        """The derived public URL for the Runloop-hosted rollout server."""
        return self._remote_base_url

    @property
    def devbox_id(self) -> str | None:
        """The Devbox ID used by this processor once setup has completed."""
        if self._devbox is not None and hasattr(self._devbox, "id"):
            return self._devbox.id
        return self._devbox_id

    def setup(self) -> None:
        """Create or attach to a Devbox, expose the server port, and start the server."""
        if self._remote_processor is not None:
            return

        api_key = self._runloop_api_key or os.getenv("RUNLOOP_API_KEY")
        if not api_key:
            raise ValueError(
                "RUNLOOP_API_KEY is required for RunloopRolloutProcessor. "
                "Set the environment variable or pass runloop_api_key explicitly."
            )

        RunloopSDK = _load_runloop_sdk()
        client: Any = RunloopSDK(bearer_token=api_key)
        self._client = client

        try:
            if self._devbox_id:
                devbox = client.devbox.from_id(self._devbox_id)
                self._owns_devbox = False
            else:
                assert self._blueprint_id is not None
                devbox = client.devbox.create_from_blueprint_id(self._blueprint_id)
                self._owns_devbox = True

            self._devbox = devbox
            self._await_running()
            tunnel = self._create_tunnel()
            self._remote_base_url = self._derive_remote_base_url(tunnel)
            self._server_execution = devbox.cmd.exec_async(self._server_command)
            self._wait_for_server_startup()
            self._remote_processor = RemoteRolloutProcessor(
                remote_base_url=self._remote_base_url,
                model_base_url=self._model_base_url,
                poll_interval=self._poll_interval,
                timeout_seconds=self._timeout_seconds,
                include_payloads=self._include_payloads,
            )
        except Exception:
            self._cleanup_partial_setup()
            raise

    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        if self._remote_processor is None:
            self.setup()
        assert self._remote_processor is not None
        return self._remote_processor(rows, config)

    async def acleanup(self) -> None:
        """Async cleanup for the delegated processor and any owned Devbox."""
        if self._remote_processor is not None:
            await self._remote_processor.acleanup()
        if self._should_shutdown_devbox():
            await asyncio.to_thread(self._shutdown_devbox)

    def cleanup(self) -> None:
        """Best-effort synchronous cleanup."""
        if self._remote_processor is not None:
            self._remote_processor.cleanup()
        if self._should_shutdown_devbox():
            self._shutdown_devbox()

    def _await_running(self) -> None:
        await_running = getattr(self._devbox, "await_running", None)
        if await_running is None:
            return
        await_running()

    def _create_tunnel(self) -> Any:
        assert self._devbox is not None
        net = self._devbox.net
        create_tunnel = getattr(net, "create_tunnel", None)
        if create_tunnel is not None:
            return create_tunnel(port=self._port)

        enable_tunnel = getattr(net, "enable_tunnel", None)
        if enable_tunnel is None:
            raise RuntimeError("Runloop Devbox networking API does not expose create_tunnel or enable_tunnel")
        return enable_tunnel(auth_mode="open")

    def _derive_remote_base_url(self, tunnel: Any) -> str:
        get_tunnel_url = getattr(self._devbox, "get_tunnel_url", None)
        if get_tunnel_url is not None:
            url = get_tunnel_url(self._port)
            if url:
                return str(url).rstrip("/")

        for attr in ("url", "base_url", "public_url"):
            value = getattr(tunnel, attr, None)
            if value:
                return str(value).rstrip("/")

        tunnel_key = getattr(tunnel, "tunnel_key", None)
        if tunnel_key:
            return f"https://{self._port}-{tunnel_key}.tunnel.runloop.ai"

        raise RuntimeError("Could not determine Runloop tunnel URL for the rollout server")

    def _wait_for_server_startup(self) -> None:
        if self._startup_timeout_seconds <= 0:
            return
        assert self._remote_base_url is not None

        deadline = time.monotonic() + self._startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                request = urllib.request.Request(self._remote_base_url, method="GET")
                with urllib.request.urlopen(request, timeout=min(5.0, self._startup_timeout_seconds)) as response:
                    response.read(1)
                    return
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return
                last_error = exc
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            except Exception as exc:
                last_error = exc
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

        message = f"Runloop rollout server did not become reachable within {self._startup_timeout_seconds} seconds"
        if last_error is not None:
            message = f"{message}: {last_error}"
        raise TimeoutError(message)

    def _should_shutdown_devbox(self) -> bool:
        return (
            self._devbox is not None
            and self._owns_devbox
            and self._shutdown_on_cleanup
            and not self._shutdown_complete
        )

    def _shutdown_devbox(self) -> None:
        if self._devbox is None or self._shutdown_complete:
            return
        self._devbox.shutdown()
        self._shutdown_complete = True

    def _cleanup_partial_setup(self) -> None:
        if self._remote_processor is not None:
            self._remote_processor.cleanup()
            self._remote_processor = None
        if self._should_shutdown_devbox():
            self._shutdown_devbox()
        self._devbox = None
        self._server_execution = None
        self._remote_base_url = None
        self._owns_devbox = False
        self._shutdown_complete = False
