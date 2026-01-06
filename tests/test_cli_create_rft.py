import json
import os
import argparse
import requests
from types import SimpleNamespace
from unittest.mock import patch
from typing import Any, cast
import pytest

from eval_protocol.cli_commands import create_rft as cr
from eval_protocol.cli_commands import upload as upload_mod
import eval_protocol.fireworks_rft as fr
from eval_protocol.cli import parse_args
import eval_protocol.cli_commands.utils as cli_utils


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.fixture
def stub_fireworks(monkeypatch) -> dict[str, Any]:
    """
    Stub Fireworks SDK so tests stay offline and so create_rft.py can inspect a stable
    create() signature (it uses inspect.signature(Fireworks().reinforcement_fine_tuning_jobs.create)).

    Returns:
        A dict containing the last captured create() kwargs under key "kwargs".
    """
    captured: dict[str, Any] = {"kwargs": None}

    class _FakeJobs:
        # Mirror the SDK method signature for inspect.signature(...)
        def create(
            self,
            *,
            account_id=None,
            dataset=None,
            evaluator=None,
            reinforcement_fine_tuning_job_id=None,
            chunk_size=None,
            display_name=None,
            eval_auto_carveout=None,
            evaluation_dataset=None,
            inference_parameters=None,
            loss_config=None,
            mcp_server=None,
            node_count=None,
            training_config=None,
            wandb_config=None,
            **kwargs,
        ):
            captured["kwargs"] = {
                "account_id": account_id,
                "dataset": dataset,
                "evaluator": evaluator,
                "reinforcement_fine_tuning_job_id": reinforcement_fine_tuning_job_id,
                "chunk_size": chunk_size,
                "display_name": display_name,
                "eval_auto_carveout": eval_auto_carveout,
                "evaluation_dataset": evaluation_dataset,
                "inference_parameters": inference_parameters,
                "loss_config": loss_config,
                "mcp_server": mcp_server,
                "node_count": node_count,
                "training_config": training_config,
                "wandb_config": wandb_config,
                **kwargs,
            }
            return SimpleNamespace(name=f"accounts/{account_id}/reinforcementFineTuningJobs/xyz")

    class _FakeFW:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url
            self.reinforcement_fine_tuning_jobs = _FakeJobs()

    monkeypatch.setattr(cr, "Fireworks", _FakeFW)
    return captured


@pytest.fixture
def rft_test_harness(tmp_path, monkeypatch, stub_fireworks):
    """
    Common setup for create_rft_command tests:
    - Creates a temp project and chdirs into it
    - Sets FIREWORKS_* env vars
    - Stubs out upload / polling / evaluator activation to avoid real network calls
    """
    # Isolate HOME and CWD
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    # Environment required by command
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_dummy")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    # Account id is derived from API key; mock the verify call to keep tests offline.
    monkeypatch.setattr(cli_utils, "verify_api_key_and_get_account_id", lambda *a, **k: "acct123")

    monkeypatch.setattr(upload_mod, "_prompt_select", lambda tests, non_interactive=False: tests[:1])
    monkeypatch.setattr(upload_mod, "upload_command", lambda args: 0)
    monkeypatch.setattr(cr, "_poll_evaluator_status", lambda **kwargs: True)
    monkeypatch.setattr(cr, "_upload_and_ensure_evaluator", lambda *a, **k: True)

    return project


def test_create_rft_passes_all_flags_into_request_body(rft_test_harness, stub_fireworks):
    _ = rft_test_harness
    captured = stub_fireworks

    args = argparse.Namespace(
        # Required top-level SDK fields
        dataset="accounts/acct123/datasets/my-ds",
        evaluator="accounts/acct123/evaluators/my-evaluator",
        # Model selection (exactly one) - prefixed dests from signature introspection
        training_config_base_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        training_config_warm_start_from=None,
        training_config_output_model="my-output-model",
        # Training config - prefixed
        training_config_epochs=3,
        training_config_batch_size=65536,
        training_config_learning_rate=5e-5,
        training_config_lora_rank=32,
        training_config_max_context_length=131072,
        training_config_region="us-east4",
        # Inference params - prefixed
        inference_parameters_temperature=0.9,
        inference_parameters_top_p=0.95,
        inference_parameters_top_k=50,
        inference_parameters_max_output_tokens=4096,
        inference_parameters_response_candidates_count=6,
        inference_parameters_extra_body='{"foo":"bar"}',
        # Rollout chunking and eval carveout
        chunk_size=250,
        eval_auto_carveout=False,  # explicitly disabled via --no-eval-auto-carveout
        evaluation_dataset="accounts/acct123/datasets/eval-ds",
        # Loss config - prefixed
        loss_config_method="grpo",
        loss_config_kl_beta=0.1,
        # W&B - prefixed
        wandb_config_enabled=True,
        wandb_config_project="proj",
        wandb_config_entity="ent",
        wandb_config_run_id="run123",
        wandb_config_api_key="key123",
        reinforcement_fine_tuning_job_id="my-job-id",
        display_name="My Job",
    )

    rc = cr._create_rft_job(
        account_id="acct123",
        api_key="fw_dummy",
        api_base="https://api.fireworks.ai",
        evaluator_id="my-evaluator",
        evaluator_resource_name="accounts/acct123/evaluators/my-evaluator",
        dataset_id="my-ds",
        dataset_resource="accounts/acct123/datasets/my-ds",
        args=args,
        dry_run=False,
    )
    assert rc == 0
    assert captured["kwargs"] is not None
    kw = cast(dict[str, Any], captured["kwargs"])

    # Top-level kwargs
    assert kw["account_id"] == "acct123"
    assert kw["dataset"] == "accounts/acct123/datasets/my-ds"
    assert kw["evaluator"] == "accounts/acct123/evaluators/my-evaluator"
    assert kw["chunk_size"] == 250
    assert kw["eval_auto_carveout"] is False
    assert kw["evaluation_dataset"] == "accounts/acct123/datasets/eval-ds"
    assert kw["reinforcement_fine_tuning_job_id"] == "my-job-id"
    assert kw["display_name"] == "My Job"

    # Training config mapping (snake_case)
    tc = kw["training_config"]
    assert tc["base_model"] == "accounts/fireworks/models/llama-v3p1-8b-instruct"
    assert tc["output_model"] == "my-output-model"
    assert tc["epochs"] == 3
    assert tc["batch_size"] == 65536
    assert abs(tc["learning_rate"] - 5e-5) < 1e-12
    assert tc["lora_rank"] == 32
    assert tc["max_context_length"] == 131072
    assert tc["region"] == "us-east4"

    # Inference params mapping (snake_case)
    ip = kw["inference_parameters"]
    assert abs(ip["temperature"] - 0.9) < 1e-12
    assert abs(ip["top_p"] - 0.95) < 1e-12
    assert ip["top_k"] == 50
    assert ip["max_output_tokens"] == 4096
    assert ip["response_candidates_count"] == 6
    assert ip["extra_body"] == '{"foo":"bar"}'

    # Loss config mapping (snake_case)
    lc = kw["loss_config"]
    assert lc["method"] == "grpo"
    assert abs(lc["kl_beta"] - 0.1) < 1e-12

    # W&B mapping (snake_case)
    wb = kw["wandb_config"]
    assert wb["enabled"] is True
    assert wb["project"] == "proj"
    assert wb["entity"] == "ent"
    assert wb["run_id"] == "run123"
    assert wb["api_key"] == "key123"


def test_create_rft_evaluator_validation_fails(rft_test_harness, monkeypatch):
    project = rft_test_harness

    # Valid dataset JSONL so dataset validation passes; focus on evaluator validation
    ds_path = project / "dataset_valid.jsonl"
    ds_path.write_text('{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")

    # Single discovered test for evaluator resolution
    test_file = project / "metric" / "test_eval_validation.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# dummy eval test", encoding="utf-8")
    single_disc = SimpleNamespace(qualname="metric.test_eval_validation", file_path=str(test_file))
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])

    # Force local evaluator validation to fail
    calls = {"count": 0, "pytest_target": None}

    def _fake_run_evaluator_test(project_root, pytest_target, ignore_docker, docker_build_extra, docker_run_extra):
        calls["count"] += 1
        calls["pytest_target"] = pytest_target
        return 1  # non-zero exit code => validation failure

    monkeypatch.setattr(cr, "run_evaluator_test", _fake_run_evaluator_test)

    args = argparse.Namespace(
        evaluator=None,
        yes=True,
        dry_run=True,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=str(ds_path),
        dataset_display_name=None,
        dataset_builder=None,
        base_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        warm_start_from=None,
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
        skip_validation=False,
        ignore_docker=True,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 1
    # Evaluator validation should have been invoked once and failed
    assert calls["count"] == 1
    assert isinstance(calls["pytest_target"], str)
    assert "test_eval_validation.py::test_eval_validation" in calls["pytest_target"]


def test_create_rft_evaluator_validation_passes(rft_test_harness, monkeypatch):
    project = rft_test_harness

    # Valid dataset JSONL so dataset validation passes; focus on evaluator validation
    ds_path = project / "dataset_valid.jsonl"
    ds_path.write_text('{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")

    # Single discovered test for evaluator resolution
    test_file = project / "metric" / "test_eval_ok.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# dummy ok eval test", encoding="utf-8")
    single_disc = SimpleNamespace(qualname="metric.test_eval_ok", file_path=str(test_file))
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])

    # Force local evaluator validation to succeed
    calls = {"count": 0, "pytest_target": None}

    def _fake_run_evaluator_test(project_root, pytest_target, ignore_docker, docker_build_extra, docker_run_extra):
        calls["count"] += 1
        calls["pytest_target"] = pytest_target
        return 0  # success

    monkeypatch.setattr(cr, "run_evaluator_test", _fake_run_evaluator_test)

    args = argparse.Namespace(
        evaluator=None,
        yes=True,
        dry_run=True,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=str(ds_path),
        dataset_display_name=None,
        dataset_builder=None,
        base_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        warm_start_from=None,
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
        skip_validation=False,
        ignore_docker=True,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 0
    # Evaluator validation should have been invoked once and passed
    assert calls["count"] == 1
    assert isinstance(calls["pytest_target"], str)
    assert "test_eval_ok.py::test_eval_ok" in calls["pytest_target"]


def test_create_rft_dataset_validation_fails(rft_test_harness, monkeypatch):
    project = rft_test_harness

    # Invalid dataset JSONL (schema mismatch for EvaluationRow)
    ds_path = project / "dataset_invalid.jsonl"
    ds_path.write_text('{"messages": "not-a-list"}\n', encoding="utf-8")

    # Ensure evaluator validation would pass if reached (so failure is from dataset)
    calls = {"evaluator_validation_calls": 0}

    def _fake_run_evaluator_test(project_root, pytest_target, ignore_docker, docker_build_extra, docker_run_extra):
        calls["evaluator_validation_calls"] += 1
        return 0

    monkeypatch.setattr(cr, "run_evaluator_test", _fake_run_evaluator_test)

    args = argparse.Namespace(
        evaluator="my-evaluator",
        yes=True,
        dry_run=True,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=str(ds_path),
        dataset_display_name=None,
        dataset_builder=None,
        base_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        warm_start_from=None,
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
        skip_validation=False,
        ignore_docker=True,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 1
    # Dataset validation should fail before evaluator validation is invoked
    assert calls["evaluator_validation_calls"] == 0


def test_create_rft_dataset_validation_passes(rft_test_harness, monkeypatch):
    project = rft_test_harness

    # Valid dataset JSONL compatible with EvaluationRow
    ds_path = project / "dataset_valid_evalrow.jsonl"
    ds_path.write_text('{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")

    # Evaluator validation should run and succeed
    calls = {"evaluator_validation_calls": 0}

    def _fake_run_evaluator_test(project_root, pytest_target, ignore_docker, docker_build_extra, docker_run_extra):
        calls["evaluator_validation_calls"] += 1
        return 0

    monkeypatch.setattr(cr, "run_evaluator_test", _fake_run_evaluator_test)

    args = argparse.Namespace(
        evaluator="my-evaluator",
        yes=True,
        dry_run=True,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=str(ds_path),
        dataset_display_name=None,
        dataset_builder=None,
        base_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        warm_start_from=None,
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
        skip_validation=False,
        ignore_docker=True,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 0
    # Dataset validation should pass; evaluator validation may be skipped when no local test is associated


def test_create_rft_picks_most_recent_evaluator_and_dataset_id_follows(rft_test_harness, monkeypatch):
    project = rft_test_harness

    # Create a dummy dataset jsonl file
    ds_path = project / "evaluator" / "dummy_dataset.jsonl"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    ds_path.write_text('{"input":"x"}\n', encoding="utf-8")

    # Simulate exactly one discovered test and selector returning it
    one_file = project / "metric" / "test_single.py"
    one_file.parent.mkdir(parents=True, exist_ok=True)
    one_file.write_text("# single", encoding="utf-8")
    single_disc = SimpleNamespace(qualname="metric.test_single", file_path=str(one_file))
    # New flow uses _discover_and_select_tests; patch it to return our single test.
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])
    monkeypatch.setattr(upload_mod, "_prompt_select", lambda tests, non_interactive=False: tests[:1])
    monkeypatch.setattr(upload_mod, "upload_command", lambda args: 0)
    monkeypatch.setattr(cr, "_poll_evaluator_status", lambda **kwargs: True)

    captured = {"dataset_id": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["dataset_id"] = dataset_id
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)

    # Build args: non_interactive (yes=True), no explicit evaluator_id, valid warm_start_from
    args = type("Args", (), {})()
    setattr(args, "evaluator", None)
    setattr(args, "yes", True)
    setattr(args, "dry_run", False)
    setattr(args, "force", False)
    setattr(args, "env_file", None)
    setattr(args, "dataset", None)
    setattr(args, "dataset_jsonl", str(ds_path))
    setattr(args, "dataset_display_name", None)
    setattr(args, "dataset_builder", None)
    setattr(args, "training_config_base_model", None)
    setattr(args, "training_config_warm_start_from", "accounts/acct123/models/ft-abc123")
    setattr(args, "training_config_output_model", None)
    setattr(args, "n", None)
    setattr(args, "max_tokens", None)
    setattr(args, "training_config_learning_rate", None)
    setattr(args, "training_config_batch_size", None)
    setattr(args, "training_config_epochs", None)
    setattr(args, "training_config_lora_rank", None)
    setattr(args, "training_config_max_context_length", None)
    setattr(args, "chunk_size", None)
    setattr(args, "eval_auto_carveout", None)
    setattr(args, "skip_validation", True)
    setattr(args, "ignore_docker", False)
    setattr(args, "docker_build_extra", "")
    setattr(args, "docker_run_extra", "")

    rc = cr.create_rft_command(args)
    assert rc == 0

    # Assert dataset id derived from selected test: metric-test_single
    assert captured["dataset_id"] is not None
    assert captured["dataset_id"].startswith("test-single-test-single-dataset-")


def test_create_rft_passes_matching_evaluator_id_and_entry_with_multiple_tests(rft_test_harness, monkeypatch):
    # Project structure and CWD from shared harness
    project = rft_test_harness

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

    # Capture dataset id used during dataset creation
    captured = {"dataset_id": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["dataset_id"] = dataset_id
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)
    # Job creation is handled via the (stubbed) Fireworks SDK client in the fixture.

    # Provide a dataset jsonl so flow proceeds
    ds_path = eval_dir / "dummy_dataset.jsonl"
    ds_path.write_text('{"input":"x"}\n', encoding="utf-8")

    # Build args: no explicit evaluator id, selector will not be used here; mapping by id

    args = argparse.Namespace(
        evaluator=cr._normalize_evaluator_id("foo_eval-test_bar_evaluation"),
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
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
        skip_validation=True,
        ignore_docker=False,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 0

    # Assert dataset id is derived from the evaluator id (trimmed base + '-dataset-<timestamp>')
    assert captured["dataset_id"] is not None
    expected_prefix = (
        cr._build_trimmed_dataset_id(cr._normalize_evaluator_id("foo_eval-test_bar_evaluation")).split("-dataset-")[0]
        + "-dataset-"
    )
    assert captured["dataset_id"].startswith(expected_prefix)


def test_create_rft_interactive_selector_single_test(rft_test_harness, monkeypatch):
    # Setup project using shared harness
    project = rft_test_harness

    # Single discovered test
    test_file = project / "metric" / "test_one.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# one", encoding="utf-8")
    single_disc = SimpleNamespace(qualname="metric.test_one", file_path=str(test_file))
    # New flow uses _discover_and_select_tests; patch it to return our single test.
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])

    # Capture dataset id used during dataset creation
    captured = {"dataset_id": None}

    # Provide dataset jsonl
    ds_path = project / "metric" / "dataset.jsonl"
    ds_path.write_text('{"input":"x"}\n', encoding="utf-8")
    monkeypatch.setattr(
        cr,
        "create_dataset_from_jsonl",
        lambda account_id, api_key, api_base, dataset_id, display_name, jsonl_path: (
            captured.__setitem__("dataset_id", dataset_id) or dataset_id,
            {"name": f"accounts/{account_id}/datasets/{dataset_id}"},
        ),
    )
    # Job creation is handled via the (stubbed) Fireworks SDK client in the fixture.

    # Run without evaluator_id; use --yes so selector returns tests directly (no UI)

    args = argparse.Namespace(
        evaluator=None,
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
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
        skip_validation=True,
        ignore_docker=False,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 0
    # Assert dataset id is derived from the selected test's evaluator id
    assert captured["dataset_id"] is not None
    expected_prefix = (
        cr._build_trimmed_dataset_id(cr._normalize_evaluator_id("test_one-test_one")).split("-dataset-")[0]
        + "-dataset-"
    )
    assert captured["dataset_id"].startswith(expected_prefix)


def test_create_rft_quiet_existing_evaluator_skips_upload(tmp_path, monkeypatch, stub_fireworks):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    # Env
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_dummy")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    monkeypatch.setattr(cli_utils, "verify_api_key_and_get_account_id", lambda *a, **k: "acct123")

    # Mock evaluator exists and is ACTIVE
    class _Resp:
        ok = True

        def json(self):
            return {"state": "ACTIVE"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(cr.requests, "get", lambda *a, **k: _Resp())

    # Provide dataset via --dataset-jsonl so no test discovery needed
    ds_path = project / "dataset.jsonl"
    ds_path.write_text('{"input":"x"}\n', encoding="utf-8")
    monkeypatch.setattr(
        cr,
        "create_dataset_from_jsonl",
        lambda account_id, api_key, api_base, dataset_id, display_name, jsonl_path: (
            dataset_id,
            {"name": f"accounts/{account_id}/datasets/{dataset_id}"},
        ),
    )
    _ = stub_fireworks

    args = argparse.Namespace(
        evaluator="some-eval",
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
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


def test_create_rft_quiet_new_evaluator_ambiguous_without_entry_errors(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    # Env
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_dummy")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    monkeypatch.setattr(cli_utils, "verify_api_key_and_get_account_id", lambda *a, **k: "acct123")

    # Evaluator does not exist (force path into upload section)
    def _raise(*a, **k):
        raise requests.exceptions.RequestException("nope")

    monkeypatch.setattr(cr.requests, "get", _raise)

    # Two discovered tests (ambiguous)
    f1 = project / "a.py"
    f2 = project / "b.py"
    f1.write_text("# a", encoding="utf-8")
    f2.write_text("# b", encoding="utf-8")
    d1 = SimpleNamespace(qualname="a.test_one", file_path=str(f1))
    d2 = SimpleNamespace(qualname="b.test_two", file_path=str(f2))
    monkeypatch.setattr(cr, "_discover_tests", lambda cwd: [d1, d2])

    args = argparse.Namespace(
        evaluator="some-eval",
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=str(project / "dataset.jsonl"),
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
    # create the dataset file so we don't fail earlier
    (project / "dataset.jsonl").write_text('{"input":"x"}\n', encoding="utf-8")

    rc = cr.create_rft_command(args)
    assert rc == 1


def test_create_rft_fallback_to_dataset_builder(rft_test_harness, monkeypatch):
    project = rft_test_harness
    # Single discovered test without data_loaders or input_dataset
    test_file = project / "metric" / "test_builder.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# builder case", encoding="utf-8")
    single_disc = SimpleNamespace(qualname="metric.test_builder", file_path=str(test_file))
    # New flow uses _discover_and_select_tests for evaluator resolution; patch it to return our single test.
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])
    # Also patch _discover_tests for any direct calls during dataset inference.
    monkeypatch.setattr(cr, "_discover_tests", lambda cwd: [single_disc])

    # Dataset builder fallback
    out_jsonl = project / "metric" / "builder_out.jsonl"
    out_jsonl.write_text('{"row":1}\n{"row":2}\n', encoding="utf-8")

    monkeypatch.setattr(cr, "detect_dataset_builder", lambda metric_dir: "builder.py::build_training_dataset")
    monkeypatch.setattr(cr, "materialize_dataset_via_builder", lambda spec: (str(out_jsonl), 2))

    # Capture dataset creation args
    captured = {"dataset_id": None, "jsonl_path": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["dataset_id"] = dataset_id
        captured["jsonl_path"] = jsonl_path
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)
    # Job creation is handled via the (stubbed) Fireworks SDK client in the fixture.

    # Run without dataset inputs so builder path is used

    args = argparse.Namespace(
        evaluator=None,
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=None,
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
        skip_validation=True,
    )

    rc = cr.create_rft_command(args)
    assert rc == 0
    # Evaluator id derived from test_builder -> "test-builder-test-builder"
    assert captured["dataset_id"] is not None
    assert captured["dataset_id"].startswith("test-builder-test-builder-dataset-")
    # Ensure we used the materialized JSONL
    assert captured["jsonl_path"] == str(out_jsonl)


def test_create_rft_rejects_dataloader_jsonl(rft_test_harness, monkeypatch):
    project = rft_test_harness
    # Single discovered test
    test_file = project / "metric" / "test_loader.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# loader case", encoding="utf-8")
    single_disc = SimpleNamespace(qualname="metric.test_loader", file_path=str(test_file))
    # New flow uses _discover_and_select_tests; patch it to return our single test.
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])

    # Provide JSONL via dataloader extractor
    dl_jsonl = project / "metric" / "loader_out.jsonl"
    dl_jsonl.write_text('{"a":1}\n', encoding="utf-8")
    monkeypatch.setattr(cr, "_extract_jsonl_from_dataloader", lambda f, fn: str(dl_jsonl))
    monkeypatch.setattr(cr, "_extract_jsonl_from_input_dataset", lambda f, fn: None)
    monkeypatch.setattr(cr, "detect_dataset_builder", lambda metric_dir: None)

    captured = {"dataset_id": None, "jsonl_path": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["dataset_id"] = dataset_id
        captured["jsonl_path"] = jsonl_path
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)
    # Job creation is handled via the (stubbed) Fireworks SDK client in the fixture.

    args = argparse.Namespace(
        evaluator=None,
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=None,
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
        skip_validation=True,
        ignore_docker=False,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    # Dataloader-provided JSONL is now rejected for create rft
    assert rc == 1
    assert captured["dataset_id"] is None
    assert captured["jsonl_path"] is None


def test_create_rft_uses_input_dataset_jsonl_when_available(rft_test_harness, monkeypatch):
    project = rft_test_harness
    # Single discovered test
    test_file = project / "metric" / "test_input_ds.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# input_dataset case", encoding="utf-8")
    single_disc = SimpleNamespace(qualname="metric.test_input_ds", file_path=str(test_file))
    # New flow uses _discover_and_select_tests; patch it to return our single test.
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])

    # Provide JSONL via input_dataset extractor
    id_jsonl = project / "metric" / "input_ds_out.jsonl"
    id_jsonl.write_text('{"b":2}\n', encoding="utf-8")
    monkeypatch.setattr(cr, "_extract_jsonl_from_dataloader", lambda f, fn: None)
    monkeypatch.setattr(cr, "_extract_jsonl_from_input_dataset", lambda f, fn: str(id_jsonl))
    monkeypatch.setattr(cr, "detect_dataset_builder", lambda metric_dir: None)

    captured = {"dataset_id": None, "jsonl_path": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["dataset_id"] = dataset_id
        captured["jsonl_path"] = jsonl_path
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)
    # Job creation is handled via the (stubbed) Fireworks SDK client in the fixture.

    args = argparse.Namespace(
        evaluator=None,
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=None,
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
        skip_validation=True,
        ignore_docker=False,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 0
    assert captured["dataset_id"] is not None
    assert captured["dataset_id"].startswith("test-input-ds-test-input-ds-dataset-")
    assert captured["jsonl_path"] == str(id_jsonl)


def test_create_rft_quiet_existing_evaluator_infers_dataset_from_matching_test(rft_test_harness, monkeypatch):
    # Setup project with multiple tests; evaluator exists (skip upload)
    project = rft_test_harness

    # Two tests discovered
    f1 = project / "evals" / "alpha.py"
    f2 = project / "evals" / "beta.py"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_text("# alpha", encoding="utf-8")
    f2.write_text("# beta", encoding="utf-8")
    d1 = SimpleNamespace(qualname="alpha.test_one", file_path=str(f1))
    d2 = SimpleNamespace(qualname="beta.test_two", file_path=str(f2))
    monkeypatch.setattr(cr, "_discover_tests", lambda cwd: [d1, d2])

    # Evaluator exists and is ACTIVE (skip upload)
    class _Resp:
        ok = True

        def json(self):
            return {"state": "ACTIVE"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(cr.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(cr, "_poll_evaluator_status", lambda **kwargs: True)

    # We will provide JSONL via input_dataset extractor for matching test (beta.test_two)
    jsonl_path = project / "data.jsonl"
    jsonl_path.write_text('{"c":3}\n', encoding="utf-8")

    # Stub extractors: only the matching test name should matter; our implementation calls extractor with file+func
    def _extract_input_jsonl(file_path, func_name):
        # Simulate returning JSONL regardless; dataset inference uses the selected test determined by evaluator_id
        return str(jsonl_path)

    monkeypatch.setattr(cr, "_extract_jsonl_from_dataloader", lambda f, fn: None)
    monkeypatch.setattr(cr, "_extract_jsonl_from_input_dataset", _extract_input_jsonl)
    monkeypatch.setattr(cr, "detect_dataset_builder", lambda metric_dir: None)

    captured = {"dataset_id": None, "jsonl_path": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["dataset_id"] = dataset_id
        captured["jsonl_path"] = jsonl_path
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)
    # Job creation is handled via the (stubbed) Fireworks SDK client in the fixture.

    # Provide evaluator_id that matches beta.test_two
    eval_id = cr._normalize_evaluator_id("beta-test_two")
    # Ensure evaluator_id maps back to the beta test for dataset inference
    monkeypatch.setattr(
        cr,
        "_resolve_selected_test",
        lambda project_root, evaluator_id, selected_tests=None: (str(f2), "test_two")
        if evaluator_id == eval_id
        else (None, None),
    )
    args = argparse.Namespace(
        evaluator=eval_id,
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=None,
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
        skip_validation=True,
        ignore_docker=False,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 0
    assert captured["dataset_id"] is not None
    # Ensure the dataset id is based on evaluator_id
    assert captured["dataset_id"].startswith(f"{eval_id}-dataset-")
    assert captured["jsonl_path"] == str(jsonl_path)


def test_cli_full_command_style_evaluator_and_dataset_flags(tmp_path, monkeypatch, stub_fireworks):
    # Isolate CWD so _discover_tests doesn't run pytest in the real project
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    # Env
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_dummy")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    monkeypatch.setattr(cli_utils, "verify_api_key_and_get_account_id", lambda *a, **k: "pyroworks-dev")

    # Mock evaluator exists and ACTIVE
    class _Resp:
        ok = True

        def json(self):
            return {"state": "ACTIVE"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(cr.requests, "get", lambda *a, **k: _Resp())

    captured = stub_fireworks

    argv = [
        "create",
        "rft",
        "--base-model",
        "accounts/fireworks/models/qwen3-0p6b",
        "--dataset",
        "svgbench-small",
        "--output-model",
        "svgbench-agent-small-bchen-2",
        "--evaluator",
        "accounts/pyroworks-dev/evaluators/test-livesvgbench-test-svg-combined-evaluation1",
        "--max-context-length",
        "65536",
        "--response-candidates-count",
        "4",
        "--batch-size",
        "128000",
        "--chunk-size",
        "50",
        "--epochs",
        "4",
        "--max-output-tokens",
        "32768",
        "--learning-rate",
        "0.00003",
        "--lora-rank",
        "16",
        "--job-id",
        "custom-job-123",
        "--yes",
    ]
    args, _ = parse_args(argv)

    # Execute command
    rc = cr.create_rft_command(args)
    assert rc == 0
    assert captured["kwargs"] is not None
    kw = cast(dict[str, Any], captured["kwargs"])

    # Evaluator and dataset resources (from CLI args)
    assert kw["evaluator"] == "accounts/pyroworks-dev/evaluators/test-livesvgbench-test-svg-combined-evaluation1"
    # NOTE: current create_rft.py seeds dataset_resource but then may be overridden by args.dataset;
    # this assertion reflects the parsed CLI value.
    assert kw["dataset"] in ("svgbench-small", "accounts/pyroworks-dev/datasets/svgbench-small")

    # Training config mapping (snake_case; values come from prefixed args)
    tc = kw["training_config"]
    assert tc["base_model"] == "accounts/fireworks/models/qwen3-0p6b"
    assert tc["output_model"] == "svgbench-agent-small-bchen-2"
    assert tc["epochs"] == 4
    assert tc["batch_size"] == 128000
    assert abs(tc["learning_rate"] - 0.00003) < 1e-12
    assert tc["lora_rank"] == 16
    assert tc["max_context_length"] == 65536

    # Inference params mapping
    ip = kw["inference_parameters"]
    assert ip["response_candidates_count"] == 4
    assert ip["max_output_tokens"] == 32768

    # Other top-level
    assert kw["chunk_size"] == 50
    assert kw["reinforcement_fine_tuning_job_id"] == "custom-job-123"


def test_create_rft_prefers_explicit_dataset_jsonl_over_input_dataset(rft_test_harness, monkeypatch):
    # Setup project
    project = rft_test_harness

    # Single discovered test
    test_file = project / "metric" / "test_pref.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# prefer explicit dataset_jsonl", encoding="utf-8")
    single_disc = SimpleNamespace(qualname="metric.test_pref", file_path=str(test_file))
    # New flow uses _discover_and_select_tests; patch it to return our single test.
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])

    monkeypatch.setattr(upload_mod, "_prompt_select", lambda tests, non_interactive=False: tests[:1])
    monkeypatch.setattr(upload_mod, "upload_command", lambda args: 0)
    monkeypatch.setattr(cr, "_poll_evaluator_status", lambda **kwargs: True)

    # Prepare two JSONL paths: one explicit via --dataset-jsonl and one inferable via input_dataset
    explicit_jsonl = project / "metric" / "explicit.jsonl"
    explicit_jsonl.write_text('{"row":"explicit"}\n', encoding="utf-8")
    inferred_jsonl = project / "metric" / "inferred.jsonl"
    inferred_jsonl.write_text('{"row":"inferred"}\n', encoding="utf-8")

    # If inference were to happen, return inferred path — but explicit should win
    monkeypatch.setattr(cr, "_extract_jsonl_from_dataloader", lambda f, fn: None)
    calls = {"input_dataset": 0}

    def _extract_input_dataset(file_path, func_name):
        calls["input_dataset"] += 1
        return str(inferred_jsonl)

    monkeypatch.setattr(cr, "_extract_jsonl_from_input_dataset", _extract_input_dataset)
    monkeypatch.setattr(cr, "detect_dataset_builder", lambda metric_dir: None)

    captured = {"jsonl_path": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["jsonl_path"] = jsonl_path
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)
    # Job creation is handled via the (stubbed) Fireworks SDK client in the fixture.

    args = argparse.Namespace(
        evaluator=None,
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=str(explicit_jsonl),
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
        skip_validation=True,
        ignore_docker=False,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 0
    # Ensure the explicitly provided JSONL file is used, not the inferred one
    assert captured["jsonl_path"] == str(explicit_jsonl)
    assert captured["jsonl_path"] != str(inferred_jsonl)
    # And because --dataset-jsonl was provided, we should never call the input_dataset extractor
    assert calls["input_dataset"] == 0


def test_create_rft_transforms_raw_input_dataset_via_dataset_adapter_before_upload(rft_test_harness, monkeypatch):
    project = rft_test_harness

    # Create a real @evaluation_test-decorated module so create_rft can extract __ep_params__.dataset_adapter
    metric_dir = project / "metric"
    metric_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl = metric_dir / "raw.jsonl"
    raw_jsonl.write_text('{"q":"hi","a":"ok"}\n{"q":"yo","a":"ok2"}\n', encoding="utf-8")

    test_file = metric_dir / "test_adapt.py"
    test_file.write_text(
        """
from typing import Any
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test

def my_adapter(rows: list[dict[str, Any]]) -> list[EvaluationRow]:
    return [
        EvaluationRow(messages=[Message(role="user", content=r["q"])], ground_truth=r.get("a"))
        for r in rows
    ]

@evaluation_test(
    input_dataset=["raw.jsonl"],
    dataset_adapter=my_adapter,
    num_runs=1,
    max_dataset_rows=2,
    mode="pointwise",
)
def test_adapt(row: EvaluationRow) -> EvaluationRow:
    return row
""".lstrip(),
        encoding="utf-8",
    )

    # Discovery: exactly one test, and resolve_selected_test points to our module/function
    single_disc = SimpleNamespace(qualname="metric.test_adapt.test_adapt", file_path=str(test_file))
    monkeypatch.setattr(cr, "_discover_and_select_tests", lambda cwd, non_interactive=False: [single_disc])
    monkeypatch.setattr(
        cr,
        "_resolve_selected_test",
        lambda project_root, evaluator_id, selected_tests=None: (str(test_file), "test_adapt"),
    )

    captured = {"jsonl_path": None}

    def _fake_create_dataset_from_jsonl(account_id, api_key, api_base, dataset_id, display_name, jsonl_path):
        captured["jsonl_path"] = jsonl_path
        return dataset_id, {"name": f"accounts/{account_id}/datasets/{dataset_id}", "state": "UPLOADING"}

    monkeypatch.setattr(cr, "create_dataset_from_jsonl", _fake_create_dataset_from_jsonl)

    # Ensure upload path doesn't touch the network; job creation via stub_fireworks fixture
    args = argparse.Namespace(
        evaluator=None,
        yes=True,
        dry_run=False,
        force=False,
        env_file=None,
        dataset=None,
        dataset_jsonl=None,
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
        skip_validation=True,
        ignore_docker=False,
        docker_build_extra="",
        docker_run_extra="",
    )

    rc = cr.create_rft_command(args)
    assert rc == 0
    assert captured["jsonl_path"] is not None
    # Raw JSONL should NOT be uploaded; transformed EvaluationRow JSONL should be.
    assert os.path.abspath(captured["jsonl_path"]) != os.path.abspath(str(raw_jsonl))
    assert os.path.basename(captured["jsonl_path"]).endswith(".jsonl")
    # The transformed file should validate as EvaluationRow JSONL
    assert cr._validate_dataset_jsonl(captured["jsonl_path"])
