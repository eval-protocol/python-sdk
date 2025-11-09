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
    test_file = project / "metric" / "test_dummy.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# dummy", encoding="utf-8")

    dummy = SimpleNamespace(qualname="dummy_module.test_dummy_evaluation", file_path=str(test_file))
    monkeypatch.setattr(cr, "_discover_tests", lambda cwd: [dummy])

    eid = cr._auto_select_evaluator_id(str(project))
    assert eid is not None
    # Should incorporate function name suffix
    assert "test_dummy_evaluation".split("_")[-1] in eid or "test-dummy-evaluation" in eid


def test_auto_select_returns_none_when_no_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()

    # No traces, no tests
    monkeypatch.setattr(cr, "_discover_tests", lambda cwd: [])
    eid = cr._auto_select_evaluator_id(str(project))
    assert eid is None


def test_create_rft_picks_most_recent_evaluator_and_dataset_id_follows(tmp_path, monkeypatch):
    # Isolate HOME so expanduser paths remain inside tmp
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    # Create a fake project and chdir into it (create_rft uses os.getcwd())
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    # Prepare two evaluator traces with different mtimes
    traces_dir = project / ".eval_protocol" / "evaluators"
    traces_dir.mkdir(parents=True, exist_ok=True)
    older = traces_dir / "example-eval-1.json"
    newer = traces_dir / "example-eval-2.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    t0 = time.time() - 200
    os.utime(str(older), (t0, t0))
    t1 = time.time()
    os.utime(str(newer), (t1, t1))

    # Create a dummy dataset jsonl file
    ds_path = project / "evaluator" / "dummy_dataset.jsonl"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    ds_path.write_text('{"input":"x"}\n', encoding="utf-8")

    # Env required by create_rft_command
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_dummy")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")

    # Stub out networked/subcommands used by create_rft
    # Patch upload command in its own module (create_rft imports it at call time)
    import eval_protocol.cli_commands.upload as upload_mod

    monkeypatch.setattr(upload_mod, "upload_command", lambda args: 0)
    monkeypatch.setattr(cr, "_poll_evaluator_status", lambda **kwargs: True)

    captured = {"dataset_id": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["dataset_id"] = dataset_id
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)
    monkeypatch.setattr(cr, "create_reinforcement_fine_tuning_job", lambda *a, **k: {"name": "jobs/123"})

    # Build args: non_interactive (yes=True), no explicit evaluator_id, valid warm_start_from
    args = type("Args", (), {})()
    setattr(args, "evaluator_id", None)
    setattr(args, "yes", True)
    setattr(args, "dry_run", False)
    setattr(args, "force", False)
    setattr(args, "env_file", None)
    setattr(args, "dataset_id", None)
    setattr(args, "dataset_jsonl", str(ds_path))
    setattr(args, "dataset_display_name", None)
    setattr(args, "dataset_builder", None)
    setattr(args, "base_model", None)
    setattr(args, "warm_start_from", "accounts/acct123/models/ft-abc123")
    setattr(args, "output_model", None)
    setattr(args, "n", None)
    setattr(args, "max_tokens", None)
    setattr(args, "learning_rate", None)
    setattr(args, "batch_size", None)
    setattr(args, "epochs", None)
    setattr(args, "lora_rank", None)
    setattr(args, "max_context_length", None)
    setattr(args, "chunk_size", None)
    setattr(args, "eval_auto_carveout", None)

    rc = cr.create_rft_command(args)
    assert rc == 0

    # Assert dataset id followed the most recent evaluator id ("example-eval-2")
    assert captured["dataset_id"] is not None
    assert captured["dataset_id"].startswith("example-eval-2-dataset-")


def test_create_rft_passes_matching_evaluator_id_and_entry_with_multiple_tests(tmp_path, monkeypatch):
    # Ensure expanduser paths stay under tmp
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    # Project structure and CWD
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    # Two evaluator traces: make the target evaluator the most recent
    traces_dir = project / ".eval_protocol" / "evaluators"
    traces_dir.mkdir(parents=True, exist_ok=True)
    svg_id = "example-svg-evaluation"
    # Use an evaluator id that matches normalization logic for mapping to foo_eval.py::test_bar_evaluation
    target_id = cr._normalize_evaluator_id("foo_eval-test_bar_evaluation")
    older = traces_dir / f"{svg_id}.json"
    newer = traces_dir / f"{target_id}.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    t0 = time.time() - 200
    os.utime(str(older), (t0, t0))
    t1 = time.time()
    os.utime(str(newer), (t1, t1))

    # Create dummy test files for discovery
    eval_dir = project / "evaluator"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cal_file = eval_dir / "foo_eval.py"
    svg_file = eval_dir / "bar_eval.py"
    cal_file.write_text("# foo", encoding="utf-8")
    svg_file.write_text("# bar", encoding="utf-8")

    # Fake discovered tests: foo and bar
    cal_disc = SimpleNamespace(qualname="foo_eval.test_bar_evaluation", file_path=str(cal_file))
    svg_disc = SimpleNamespace(qualname="bar_eval.test_baz_evaluation", file_path=str(svg_file))
    monkeypatch.setattr(cr, "_discover_tests", lambda cwd: [cal_disc, svg_disc])

    # Env for CLI
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_dummy")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")

    # Capture what upload receives (id and entry)
    captured = {"id": None, "entry": None, "dataset_id": None}

    # Monkeypatch the upload command from the upload module (the function imports it inside)
    import eval_protocol.cli_commands.upload as upload_mod

    def _fake_upload(ns):
        captured["id"] = getattr(ns, "id", None)
        captured["entry"] = getattr(ns, "entry", None)
        return 0

    monkeypatch.setattr(upload_mod, "upload_command", _fake_upload)

    # Avoid network and capture dataset id
    monkeypatch.setattr(cr, "_poll_evaluator_status", lambda **kwargs: True)

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["dataset_id"] = dataset_id
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)
    monkeypatch.setattr(cr, "create_reinforcement_fine_tuning_job", lambda *a, **k: {"name": "jobs/123"})

    # Provide a dataset jsonl so flow proceeds
    ds_path = eval_dir / "dummy_dataset.jsonl"
    ds_path.write_text('{"input":"x"}\n', encoding="utf-8")

    # Build args: non-interactive, no explicit evaluator id
    import argparse

    args = argparse.Namespace(
        evaluator_id=None,
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset_id=None,
        dataset_jsonl=str(ds_path),
        dataset_display_name=None,
        dataset_builder=None,
        base_model=None,
        warm_start_from="accounts/acct123/models/ft-abc123",
        output_model=None,
        n=None,
        max_tokens=None,
        learning_rate=None,
        batch_size=None,
        epochs=None,
        lora_rank=None,
        max_context_length=None,
        chunk_size=None,
        eval_auto_carveout=None,
    )

    rc = cr.create_rft_command(args)
    assert rc == 0

    # Assert evaluator_id passed to upload matches the most recent trace (target)
    assert captured["id"] == target_id
    # Assert entry points to the foo test (should map when id matches normalization)
    assert captured["entry"] is not None and captured["entry"].endswith("foo_eval.py::test_bar_evaluation")
    # Assert dataset id is derived from the same evaluator id (trimmed base + '-dataset-<timestamp>')
    assert captured["dataset_id"] is not None
    expected_prefix = cr._build_trimmed_dataset_id(target_id).split("-dataset-")[0] + "-dataset-"
    assert captured["dataset_id"].startswith(expected_prefix)
