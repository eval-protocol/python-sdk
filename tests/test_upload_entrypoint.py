import os
import sys
from types import SimpleNamespace


def _write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_entrypoint_from_path_is_path_based_nodeid(tmp_path, monkeypatch):
    # Arrange: create a simple module file in a temp project root
    project_root = tmp_path
    quickstart_path = project_root / "quickstart.py"
    _write_file(
        str(quickstart_path),
        """
def test_llm_judge(row=None):
    return 1
""".lstrip(),
    )

    # Capture arguments passed to create_evaluation
    captured = {}

    from eval_protocol.cli_commands import upload as upload_mod

    def fake_create_evaluation(**kwargs):
        captured.update(kwargs)
        # Simulate API response
        return {"name": kwargs.get("evaluator_id", "eval")}

    monkeypatch.setattr(upload_mod, "create_evaluation", fake_create_evaluation)

    # Act: call upload_command with --entry as path::function
    args = SimpleNamespace(
        path=str(project_root),
        entry=f"{quickstart_path.name}::test_llm_judge",
        id=None,
        display_name=None,
        description=None,
        force=False,
        yes=True,
    )

    exit_code = upload_mod.upload_command(args)

    # Assert
    assert exit_code == 0
    assert captured.get("entry_point") == f"{quickstart_path.name}::test_llm_judge"


def test_entrypoint_from_module_is_path_based_nodeid(tmp_path, monkeypatch):
    # Arrange: create a package with a module
    project_root = tmp_path
    pkg_dir = project_root / "mypkg"
    _write_file(str(pkg_dir / "__init__.py"), "")
    _write_file(
        str(pkg_dir / "quickstart.py"),
        """
def test_llm_judge(row=None):
    return 1
""".lstrip(),
    )

    # Ensure importable
    sys.path.insert(0, str(project_root))

    captured = {}
    from eval_protocol.cli_commands import upload as upload_mod

    def fake_create_evaluation(**kwargs):
        captured.update(kwargs)
        return {"name": kwargs.get("evaluator_id", "eval")}

    monkeypatch.setattr(upload_mod, "create_evaluation", fake_create_evaluation)

    # Act: use module::function
    args = SimpleNamespace(
        path=str(project_root),
        entry="mypkg.quickstart::test_llm_judge",
        id=None,
        display_name=None,
        description=None,
        force=False,
        yes=True,
    )

    try:
        exit_code = upload_mod.upload_command(args)
    finally:
        # Cleanup path
        if str(project_root) in sys.path:
            sys.path.remove(str(project_root))

    # Assert: path-based nodeid relative to project root
    assert exit_code == 0
    assert captured.get("entry_point") == "mypkg/quickstart.py::test_llm_judge"


def test_entrypoint_from_legacy_module_colon_is_path_based_nodeid(tmp_path, monkeypatch):
    # Arrange: create a package and module
    project_root = tmp_path
    pkg_dir = project_root / "pkg"
    _write_file(str(pkg_dir / "__init__.py"), "")
    _write_file(
        str(pkg_dir / "quickstart.py"),
        """
def test_llm_judge(row=None):
    return 1
""".lstrip(),
    )

    sys.path.insert(0, str(project_root))

    captured = {}
    from eval_protocol.cli_commands import upload as upload_mod

    def fake_create_evaluation(**kwargs):
        captured.update(kwargs)
        return {"name": kwargs.get("evaluator_id", "eval")}

    monkeypatch.setattr(upload_mod, "create_evaluation", fake_create_evaluation)

    # Act: use legacy module:function
    args = SimpleNamespace(
        path=str(project_root),
        entry="pkg.quickstart:test_llm_judge",
        id=None,
        display_name=None,
        description=None,
        force=False,
        yes=True,
    )

    try:
        exit_code = upload_mod.upload_command(args)
    finally:
        if str(project_root) in sys.path:
            sys.path.remove(str(project_root))

    # Assert
    assert exit_code == 0
    assert captured.get("entry_point") == "pkg/quickstart.py::test_llm_judge"


def test_dashboard_url_mapping_dev_host(tmp_path, monkeypatch, capsys):
    # Arrange: minimal file and capture printed URL
    project_root = tmp_path
    quickstart_path = project_root / "quickstart.py"
    _write_file(
        str(quickstart_path),
        """
def test_llm_judge(row=None):
    return 1
""".lstrip(),
    )

    from eval_protocol.cli_commands import upload as upload_mod

    # Force API base to dev.api and account to fireworks→pyroworks-dev mapping path
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://dev.api.fireworks.ai")

    def fake_create_evaluation(**kwargs):
        # Simulate creation result with evaluator name
        return {"name": kwargs.get("evaluator_id", "eval")}

    monkeypatch.setattr(upload_mod, "create_evaluation", fake_create_evaluation)

    args = SimpleNamespace(
        path=str(project_root),
        entry=f"{quickstart_path.name}::test_llm_judge",
        id="quickstart-test-llm-judge",
        display_name=None,
        description=None,
        force=True,
        yes=True,
    )

    # Act
    exit_code = upload_mod.upload_command(args)
    out = capsys.readouterr().out

    # Assert
    assert exit_code == 0
    assert "https://dev.fireworks.ai/dashboard/evaluators/quickstart-test-llm-judge" in out


def test_dashboard_url_mapping_prod_host(tmp_path, monkeypatch, capsys):
    # Arrange: minimal file
    project_root = tmp_path
    quickstart_path = project_root / "quickstart.py"
    _write_file(
        str(quickstart_path),
        """
def test_llm_judge(row=None):
    return 1
""".lstrip(),
    )

    from eval_protocol.cli_commands import upload as upload_mod

    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")

    def fake_create_evaluation(**kwargs):
        return {"name": kwargs.get("evaluator_id", "eval")}

    monkeypatch.setattr(upload_mod, "create_evaluation", fake_create_evaluation)

    args = SimpleNamespace(
        path=str(project_root),
        entry=f"{quickstart_path.name}::test_llm_judge",
        id="quickstart-test-llm-judge",
        display_name=None,
        description=None,
        force=False,
        yes=True,
    )

    # Act
    exit_code = upload_mod.upload_command(args)
    out = capsys.readouterr().out

    # Assert
    assert exit_code == 0
    assert "https://app.fireworks.ai/dashboard/evaluators/quickstart-test-llm-judge" in out
