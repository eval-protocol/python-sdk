"""
Test to ensure load_dotenv() is never called without an explicit path.

When load_dotenv() is called without a dotenv_path argument, it uses find_dotenv()
which searches up the directory tree for a .env file. This can cause unexpected
behavior when running the CLI from a subdirectory, as it may find a .env file
in a parent directory (e.g., the python-sdk repo's .env) instead of the intended
project's .env file.

This test scans all Python files in the SDK to ensure that every call to
load_dotenv() includes an explicit dotenv_path argument.
"""

import ast
import os
from pathlib import Path
from typing import List, Set, Tuple

# Directories to scan for implicit load_dotenv calls
SCAN_DIRECTORIES = [
    "eval_protocol",
]

# Directories to exclude from scanning (relative to repo root)
EXCLUDE_DIRECTORIES: Set[str] = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "build",
    "dist",
    ".eggs",
    "*.egg-info",
}


def find_implicit_load_dotenv_calls(file_path: Path) -> List[Tuple[int, str]]:
    """
    Parse a Python file and find any load_dotenv() calls without explicit dotenv_path.

    Returns a list of (line_number, code_snippet) tuples for violations.
    """
    violations = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except (IOError, UnicodeDecodeError):
        return violations

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if this is a call to load_dotenv
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "load_dotenv":
                # Check if dotenv_path is provided as a positional or keyword argument
                has_explicit_path = False

                # Check positional arguments (dotenv_path is the first positional arg)
                if node.args:
                    has_explicit_path = True

                # Check keyword arguments
                for keyword in node.keywords:
                    if keyword.arg == "dotenv_path":
                        has_explicit_path = True
                        break

                if not has_explicit_path:
                    # Get the source line for context
                    try:
                        lines = source.splitlines()
                        line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "<unknown>"
                    except (IndexError, AttributeError):
                        line = "<unknown>"

                    violations.append((node.lineno, line))

    return violations


def _should_exclude_dir(dir_name: str) -> bool:
    """Check if a directory should be excluded from scanning."""
    return dir_name in EXCLUDE_DIRECTORIES or dir_name.startswith(".")


def _scan_directory(directory: Path, repo_root: Path) -> List[Tuple[Path, int, str]]:
    """Scan a directory for implicit load_dotenv calls."""
    all_violations: List[Tuple[Path, int, str]] = []

    for root, dirs, files in os.walk(directory):
        # Filter out excluded directories in-place to prevent os.walk from descending into them
        dirs[:] = [d for d in dirs if not _should_exclude_dir(d)]

        for filename in files:
            if not filename.endswith(".py"):
                continue

            file_path = Path(root) / filename
            violations = find_implicit_load_dotenv_calls(file_path)

            for line_no, code in violations:
                all_violations.append((file_path, line_no, code))

    return all_violations


def test_no_implicit_load_dotenv_calls():
    """
    Ensure no load_dotenv() calls exist without an explicit dotenv_path argument.

    This prevents the CLI from accidentally loading .env files from parent directories
    when running from a subdirectory.
    """
    repo_root = Path(__file__).parent.parent

    all_violations: List[Tuple[Path, int, str]] = []

    for scan_dir in SCAN_DIRECTORIES:
        directory = repo_root / scan_dir
        if directory.exists():
            violations = _scan_directory(directory, repo_root)
            all_violations.extend(violations)

    if all_violations:
        error_msg = [
            "Found load_dotenv() calls without explicit dotenv_path argument.",
            "This can cause the CLI to load .env files from parent directories unexpectedly.",
            "",
            "Violations:",
        ]
        for file_path, line_no, code in all_violations:
            try:
                rel_path = file_path.relative_to(repo_root)
            except ValueError:
                rel_path = file_path
            error_msg.append(f"  {rel_path}:{line_no}: {code}")

        error_msg.extend(
            [
                "",
                "Fix by providing an explicit path:",
                "  load_dotenv(dotenv_path=Path('.') / '.env', override=True)",
                "",
            ]
        )

        assert False, "\n".join(error_msg)


def test_load_dotenv_ast_detection():
    """Test that our AST detection correctly identifies implicit vs explicit calls."""
    import tempfile

    # Test case: implicit call (should be detected)
    implicit_code = """
from dotenv import load_dotenv
load_dotenv()
load_dotenv(override=True)
load_dotenv(verbose=True, override=True)
"""

    # Test case: explicit call (should NOT be detected)
    explicit_code = """
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
load_dotenv('.env')
load_dotenv(Path('.') / '.env')
load_dotenv(dotenv_path=Path('.') / '.env', override=True)
load_dotenv(env_file_path)  # positional arg counts as explicit
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(implicit_code)
        implicit_file = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(explicit_code)
        explicit_file = Path(f.name)

    try:
        implicit_violations = find_implicit_load_dotenv_calls(implicit_file)
        explicit_violations = find_implicit_load_dotenv_calls(explicit_file)

        # Should find 3 violations in implicit code
        assert len(implicit_violations) == 3, (
            f"Expected 3 implicit violations, got {len(implicit_violations)}: {implicit_violations}"
        )

        # Should find 0 violations in explicit code
        assert len(explicit_violations) == 0, (
            f"Expected 0 explicit violations, got {len(explicit_violations)}: {explicit_violations}"
        )

    finally:
        implicit_file.unlink()
        explicit_file.unlink()
