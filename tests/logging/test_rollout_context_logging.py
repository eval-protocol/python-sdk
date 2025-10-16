import asyncio
import logging
import sys
from typing import Any, Dict, List

import importlib.util
from pathlib import Path

import pytest


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to load module {module_name} from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


FireworksTracingHttpHandler = _load_module(
    "eval_protocol.log_utils.fireworks_tracing_http_handler",
    "eval_protocol/log_utils/fireworks_tracing_http_handler.py",
).FireworksTracingHttpHandler

_rollout_context_module = _load_module(
    "eval_protocol.log_utils.rollout_context", "eval_protocol/log_utils/rollout_context.py"
)
ContextRolloutIdFilter = _rollout_context_module.ContextRolloutIdFilter
rollout_logging_context = _rollout_context_module.rollout_logging_context


def _make_record(message: str = "msg") -> logging.LogRecord:
    return logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=0, msg=message, args=(), exc_info=None
    )


def test_context_filter_respects_explicit_rollout_id() -> None:
    record = _make_record()
    record.rollout_id = "explicit-rid"

    filt = ContextRolloutIdFilter()

    assert filt.filter(record)
    assert record.rollout_id == "explicit-rid"


def test_context_filter_respects_environment_rollout_id(monkeypatch) -> None:
    monkeypatch.setenv("EP_ROLLOUT_ID", "env-rid")
    record = _make_record()

    filt = ContextRolloutIdFilter()

    try:
        assert filt.filter(record)
        assert record.rollout_id == "env-rid"
    finally:
        monkeypatch.delenv("EP_ROLLOUT_ID", raising=False)


@pytest.mark.asyncio
async def test_context_filter_correlates_concurrent_logs() -> None:
    logger = logging.getLogger("ep.test.fireworks")
    logger.setLevel(logging.INFO)

    handler = FireworksTracingHttpHandler(gateway_base_url="http://localhost:1")
    captured: List[Dict[str, Any]] = []

    # Monkeypatch the requests call used by the handler
    def fake_post(url: str, json: Dict[str, Any], timeout: int) -> Any:  # type: ignore[override]
        captured.append(json)

        class _Resp:
            status_code = 200

        return _Resp()

    handler._session.post = fake_post  # type: ignore[attr-defined]
    handler.addFilter(ContextRolloutIdFilter())
    logger.addHandler(handler)

    try:

        async def _emit(rollout_id: str, message_prefix: str) -> None:
            async with rollout_logging_context(rollout_id, experiment_id="exp", run_id="run"):
                logger.info(f"{message_prefix}-1")
                await asyncio.sleep(0)
                logger.info(f"{message_prefix}-2")
                await asyncio.sleep(0)
                logger.info(f"{message_prefix}-3")

        await asyncio.gather(
            _emit("rid-A", "A"),
            _emit("rid-B", "B"),
        )

        # We expect 6 captured payloads
        assert len(captured) == 6

        # Ensure each payload includes the correct rollout tag and message
        tags_sets = [set(entry.get("tags", [])) for entry in captured]
        messages = [entry.get("message", "") for entry in captured]

        assert any("rollout_id:rid-A" in tags for tags in tags_sets)
        assert any("rollout_id:rid-B" in tags for tags in tags_sets)
        assert any(msg.startswith("A-") for msg in messages)
        assert any(msg.startswith("B-") for msg in messages)
    finally:
        logger.removeHandler(handler)


@pytest.mark.asyncio
async def test_context_filter_groupwise_rollout_ids_tagged() -> None:
    logger = logging.getLogger("ep.test.fireworks.group")
    logger.setLevel(logging.INFO)

    handler = FireworksTracingHttpHandler(gateway_base_url="http://localhost:1")
    captured: List[Dict[str, Any]] = []

    def fake_post(url: str, json: Dict[str, Any], timeout: int) -> Any:  # type: ignore[override]
        captured.append(json)

        class _Resp:
            status_code = 200

        return _Resp()

    handler._session.post = fake_post  # type: ignore[attr-defined]
    handler.addFilter(ContextRolloutIdFilter())
    logger.addHandler(handler)

    try:
        group_ids = ["rid-1", "rid-2", "rid-3"]
        async with rollout_logging_context(group_ids[0], experiment_id="exp2", run_id="run2", rollout_ids=group_ids):
            logger.info("group-message")

        assert len(captured) == 1
        tags = set(captured[0].get("tags", []))
        # All rollout_ids should be present as tags
        for rid in group_ids:
            assert f"rollout_id:{rid}" in tags
    finally:
        logger.removeHandler(handler)
