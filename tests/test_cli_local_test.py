import os
from types import SimpleNamespace

import pytest


def test_local_test_runs_host_pytest_with_entry(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    # Create a dummy test file
    test_file = project / "metric" / "test_one.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    # Import module under test
    from eval_protocol.cli_commands import local_test as lt

    # Avoid Docker path
    monkeypatch.setattr(lt, "_find_dockerfiles", lambda root: [])

    captured = {"target": ""}

    def _fake_host(target: str) -> int:
        captured["target"] = target
        return 0

    monkeypatch.setattr(lt, "_run_pytest_host", _fake_host)

    args = SimpleNamespace(entry=str(test_file), ignore_docker=False, yes=True)
    rc = lt.local_test_command(args)  # pyright: ignore[reportArgumentType]
    assert rc == 0
    # Expect relative path target
    assert captured["target"] == os.path.relpath(str(test_file), str(project))


def test_local_test_ignores_docker_when_flag_set(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    test_file = project / "metric" / "test_two.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    from eval_protocol.cli_commands import local_test as lt

    # Pretend we have Dockerfile(s), but ignore_docker=True should skip
    monkeypatch.setattr(lt, "_find_dockerfiles", lambda root: [str(project / "Dockerfile")])

    called = {"host": False}

    def _fake_host(target: str) -> int:
        called["host"] = True
        return 0

    monkeypatch.setattr(lt, "_run_pytest_host", _fake_host)

    args = SimpleNamespace(entry=str(test_file), ignore_docker=True, yes=True)
    rc = lt.local_test_command(args)  # pyright: ignore[reportArgumentType]
    assert rc == 0
    assert called["host"] is True


def test_local_test_errors_on_multiple_dockerfiles(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    test_file = project / "metric" / "test_three.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    from eval_protocol.cli_commands import local_test as lt

    monkeypatch.setattr(
        lt, "_find_dockerfiles", lambda root: [str(project / "Dockerfile"), str(project / "another" / "Dockerfile")]
    )

    args = SimpleNamespace(entry=str(test_file), ignore_docker=False, yes=True)
    rc = lt.local_test_command(args)  # pyright: ignore[reportArgumentType]
    assert rc == 1


def test_local_test_builds_and_runs_in_docker(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    test_file = project / "metric" / "test_four.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    from eval_protocol.cli_commands import local_test as lt

    monkeypatch.setattr(lt, "_find_dockerfiles", lambda root: [str(project / "Dockerfile")])
    monkeypatch.setattr(lt, "_build_docker_image", lambda dockerfile, tag, build_extras=None: True)

    captured = {"target": "", "image": ""}

    def _fake_run_docker(root: str, image_tag: str, pytest_target: str, run_extras=None) -> int:
        captured["target"] = pytest_target
        captured["image"] = image_tag
        return 0

    monkeypatch.setattr(lt, "_run_pytest_in_docker", _fake_run_docker)

    args = SimpleNamespace(entry=str(test_file), ignore_docker=False, yes=True)
    rc = lt.local_test_command(args)  # pyright: ignore[reportArgumentType]
    assert rc == 0
    assert captured["image"] == "ep-evaluator:local"
    assert captured["target"] == os.path.relpath(str(test_file), str(project))


def test_local_test_selector_single_test(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    test_file = project / "metric" / "test_sel.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    from eval_protocol.cli_commands import local_test as lt

    # No entry; force discover + selector
    disc = SimpleNamespace(qualname="metric.test_sel", file_path=str(test_file))
    monkeypatch.setattr(lt, "_discover_tests", lambda root: [disc])
    monkeypatch.setattr(lt, "_prompt_select", lambda tests, non_interactive=False: tests[:1])
    monkeypatch.setattr(lt, "_find_dockerfiles", lambda root: [])

    called = {"host": False}

    def _fake_host(target: str) -> int:
        called["host"] = True
        return 0

    monkeypatch.setattr(lt, "_run_pytest_host", _fake_host)

    args = SimpleNamespace(entry=None, ignore_docker=False, yes=True)
    rc = lt.local_test_command(args)  # pyright: ignore[reportArgumentType]
    assert rc == 0
    assert called["host"] is True


def test_local_test_passes_docker_build_extra(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    test_file = project / "metric" / "test_build_extra.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    from eval_protocol.cli_commands import local_test as lt

    monkeypatch.setattr(lt, "_find_dockerfiles", lambda root: [str(project / "Dockerfile")])

    captured = {"extras": None}

    def _fake_build(dockerfile, tag, build_extras=None):
        captured["extras"] = build_extras
        return True

    def _fake_run_docker(root: str, image_tag: str, pytest_target: str, run_extras=None) -> int:
        return 0

    monkeypatch.setattr(lt, "_build_docker_image", _fake_build)
    monkeypatch.setattr(lt, "_run_pytest_in_docker", _fake_run_docker)

    # Extras string with multiple flags and equals-arg
    args = SimpleNamespace(
        entry=str(test_file),
        ignore_docker=False,
        yes=True,
        docker_build_extra="--no-cache --pull --progress=plain --build-arg KEY=VAL",
        docker_run_extra="",
    )
    rc = lt.local_test_command(args)  # pyright: ignore[reportArgumentType]
    assert rc == 0
    # Expect split list preserving tokens order
    assert captured["extras"] == ["--no-cache", "--pull", "--progress=plain", "--build-arg", "KEY=VAL"]


def test_local_test_passes_docker_run_extra(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    test_file = project / "metric" / "test_run_extra.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    from eval_protocol.cli_commands import local_test as lt

    monkeypatch.setattr(lt, "_find_dockerfiles", lambda root: [str(project / "Dockerfile")])
    monkeypatch.setattr(lt, "_build_docker_image", lambda dockerfile, tag, build_extras=None: True)

    captured = {"extras": None}

    def _fake_run_docker(root: str, image_tag: str, pytest_target: str, run_extras=None) -> int:
        captured["extras"] = run_extras
        return 0

    monkeypatch.setattr(lt, "_run_pytest_in_docker", _fake_run_docker)

    args = SimpleNamespace(
        entry=str(test_file),
        ignore_docker=False,
        yes=True,
        docker_build_extra="",
        docker_run_extra="--env-file .env --memory=8g --cpus=2 --add-host=host.docker.internal:host-gateway",
    )
    rc = lt.local_test_command(args)  # pyright: ignore[reportArgumentType]
    assert rc == 0
    assert captured["extras"] == [
        "--env-file",
        ".env",
        "--memory=8g",
        "--cpus=2",
        "--add-host=host.docker.internal:host-gateway",
    ]


def test_local_test_normalizes_entry_with_selector(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    # Create a dummy test file
    test_file = project / "metric" / "test_sel_abs.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    abs_entry = f"{str(test_file)}::test_dummy"

    from eval_protocol.cli_commands import local_test as lt

    # Avoid Docker path
    monkeypatch.setattr(lt, "_find_dockerfiles", lambda root: [])

    captured = {"target": ""}

    def _fake_host(target: str) -> int:
        captured["target"] = target
        return 0

    monkeypatch.setattr(lt, "_run_pytest_host", _fake_host)

    args = SimpleNamespace(entry=abs_entry, ignore_docker=False, yes=True)
    rc = lt.local_test_command(args)  # pyright: ignore[reportArgumentType]
    assert rc == 0
    # Expect project-relative path plus selector
    rel = os.path.relpath(str(test_file), str(project))
    assert captured["target"] == f"{rel}::test_dummy"
