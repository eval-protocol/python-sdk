import json
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from eval_protocol.cli_commands import create_rft as cr


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_load_and_save_last_evaluator(tmp_path, monkeypatch):
    # Force HOME to temp so expanduser paths remain inside tmp
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()

    # Initially none
    assert cr._load_last_evaluator(str(project)) is None

    # Save and load
    cr._save_last_evaluator(str(project), "evaluator-abc")
    assert cr._load_last_evaluator(str(project)) == "evaluator-abc"


def test_auto_select_uses_last_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()

    # Write last pointer under project
    last_path = project / ".eval_protocol" / "last_evaluator.json"
    _write_json(str(last_path), {"evaluator_id": "chosen-id"})

    eid = cr._auto_select_evaluator_id(str(project))
    assert eid == "chosen-id"


def test_auto_select_single_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()

    # Single evaluator trace under project
    trace = project / ".eval_protocol" / "evaluators" / "only-one.json"
    _write_json(str(trace), {"dummy": True})

    eid = cr._auto_select_evaluator_id(str(project))
    assert eid == "only-one"


def test_auto_select_multiple_traces_non_interactive_most_recent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()

    # Two traces with different mtimes
    older = project / ".eval_protocol" / "evaluators" / "older.json"
    newer = project / ".eval_protocol" / "evaluators" / "newer.json"
    _write_json(str(older), {})
    _write_json(str(newer), {})
    # Set older then newer mtime
    t0 = time.time() - 100
    os.utime(str(older), (t0, t0))
    t1 = time.time()
    os.utime(str(newer), (t1, t1))

    eid = cr._auto_select_evaluator_id(str(project), non_interactive=True)
    assert eid == "newer"


def test_auto_select_multiple_traces_interactive_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()

    # Two traces with different mtimes to force ordering: newer first, older second
    older = project / ".eval_protocol" / "evaluators" / "older.json"
    newer = project / ".eval_protocol" / "evaluators" / "newer.json"
    _write_json(str(older), {})
    _write_json(str(newer), {})
    t0 = time.time() - 100
    os.utime(str(older), (t0, t0))
    t1 = time.time()
    os.utime(str(newer), (t1, t1))

    with patch("builtins.input", return_value="2"):
        eid = cr._auto_select_evaluator_id(str(project), non_interactive=False)
    # Choosing "2" should pick the second item by recency => "older"
    assert eid == "older"


def test_auto_select_falls_back_to_single_discovered_test(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()

    # No traces; provide exactly one discovered test
    test_file = project / "metric" / "test_calendar.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# dummy", encoding="utf-8")

    dummy = SimpleNamespace(qualname="calendar_agent.test_calendar_agent_evaluation", file_path=str(test_file))
    monkeypatch.setattr(cr, "_discover_tests", lambda cwd: [dummy])

    eid = cr._auto_select_evaluator_id(str(project))
    assert eid is not None
    # Should incorporate function name suffix
    assert "test_calendar_agent_evaluation".split("_")[-1] in eid or "test-calendar-agent-evaluation" in eid


def test_auto_select_returns_none_when_no_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()

    # No traces, no tests
    monkeypatch.setattr(cr, "_discover_tests", lambda cwd: [])
    eid = cr._auto_select_evaluator_id(str(project))
    assert eid is None
