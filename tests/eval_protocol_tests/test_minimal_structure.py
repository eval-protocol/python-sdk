#!/usr/bin/env python3
"""
Minimal test to verify the eval_protocol package structure is correct.
This test focuses on structure verification without importing heavy dependencies.
"""

import ast
import importlib.util
import os
import sys


def test_package_directories():
    """Test that the required directories exist."""
    print("Testing package directories...")

    required_dirs = [
        "eval_protocol",
        "eval_protocol",
    ]

    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            missing_dirs.append(dir_name)
        else:
            print(f"  ✓ {dir_name}/ exists")

    if missing_dirs:
        print(f"  ✗ Missing directories: {missing_dirs}")
        return False

    return True


def test_init_files():
    """Test that __init__.py files exist."""
    print("\nTesting __init__.py files...")

    init_files = [
        "eval_protocol/__init__.py",
        "eval_protocol/__init__.py",
    ]

    missing_files = []
    for file_path in init_files:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path} exists")

    if missing_files:
        print(f"  ✗ Missing files: {missing_files}")
        return False

    return True


def test_eval_protocol_init():
    """Test that eval_protocol/__init__.py has the correct structure."""
    print("\nTesting eval_protocol/__init__.py structure...")

    try:
        with open("eval_protocol/__init__.py", "r") as f:
            content = f.read()

        # Check for key imports
        expected_patterns = [
            "from eval_protocol import *",
            "from eval_protocol import __version__",
            "from eval_protocol import __all__",
        ]

        for pattern in expected_patterns:
            if pattern in content:
                print(f"  ✓ Found: {pattern}")
            else:
                print(f"  ✗ Missing: {pattern}")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Error reading eval_protocol/__init__.py: {e}")
        return False


def test_setup_py_structure():
    """Test that setup.py includes both packages."""
    print("\nTesting setup.py structure...")

    try:
        with open("setup.py", "r") as f:
            content = f.read()

        # Check for package inclusion
        if "eval_protocol*" in content and "eval_protocol*" in content:
            print("  ✓ Both packages included in setup.py")
        else:
            print("  ✗ Package inclusion not found in setup.py")
            return False

        # Check for console scripts
        console_scripts = [
            "fireworks-reward=eval_protocol.cli:main",
            "eval-protocol=eval_protocol.cli:main",
        ]

        for script in console_scripts:
            if script in content:
                print(f"  ✓ Found console script: {script}")
            else:
                print(f"  ✗ Missing console script: {script}")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Error reading setup.py: {e}")
        return False


def test_find_packages():
    """Test that find_packages discovers both packages."""
    print("\nTesting find_packages...")

    try:
        from setuptools import find_packages

        packages = find_packages(include=["eval_protocol*", "eval_protocol*"])

        print(f"  Found {len(packages)} packages total")

        if "eval_protocol" in packages:
            print("  ✓ eval_protocol package found")
        else:
            print("  ✗ eval_protocol package not found")
            return False

        if "eval_protocol" in packages:
            print("  ✓ eval_protocol package found")
        else:
            print("  ✗ eval_protocol package not found")
            return False

        # Count subpackages
        rk_sub = [p for p in packages if p.startswith("eval_protocol.")]
        print(f"  Found {len(rk_sub)} eval_protocol subpackages")

        return True
    except Exception as e:
        print(f"  ✗ find_packages test failed: {e}")
        return False


def test_syntax_check():
    """Test that the Python files have valid syntax."""
    print("\nTesting syntax validity...")

    files_to_check = [
        "eval_protocol/__init__.py",
        "setup.py",
    ]

    for file_path in files_to_check:
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Parse the content to check syntax
            ast.parse(content)
            print(f"  ✓ {file_path} has valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {file_path} has syntax error: {e}")
            return False
        except Exception as e:
            print(f"  ✗ Error checking {file_path}: {e}")
            return False

    return True


def test_build_artifacts():
    """Test that build creates the expected artifacts."""
    print("\nTesting build artifacts...")

    dist_dir = "dist"
    if not os.path.isdir(dist_dir):
        print(f"  ✗ {dist_dir} directory not found")
        return False

    files = os.listdir(dist_dir)

    # Look for wheel and tar.gz files
    wheel_files = [f for f in files if f.endswith(".whl")]
    tar_files = [f for f in files if f.endswith(".tar.gz")]

    if wheel_files:
        print(f"  ✓ Found wheel file: {wheel_files[0]}")
    else:
        print("  ✗ No wheel file found")
        return False

    if tar_files:
        print(f"  ✓ Found source distribution: {tar_files[0]}")
    else:
        print("  ✗ No source distribution found")
        return False

    return True


def test_import_without_dependencies():
    """Test importing just the structure without dependencies."""
    print("\nTesting import structure (no dependencies)...")

    try:
        # Test that the module can be found
        spec = importlib.util.find_spec("eval_protocol")
        if spec is None:
            print("  ✗ eval_protocol module not found")
            return False

        print("  ✓ eval_protocol module is discoverable")

        # Test that the file exists and can be read
        if spec.origin and os.path.exists(spec.origin):
            print(f"  ✓ eval_protocol module file exists: {spec.origin}")
        else:
            print("  ✗ eval_protocol module file not found")
            return False

        return True
    except Exception as e:
        print(f"  ✗ Import structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Testing eval_protocol Structure ===")

    # Add project root to sys.path so we can find the packages
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    sys.path.insert(0, project_root)
    os.chdir(project_root)  # Change to project root for relative paths

    tests = [
        test_package_directories,
        test_init_files,
        test_eval_protocol_init,
        test_setup_py_structure,
        test_find_packages,
        test_syntax_check,
        test_build_artifacts,
        test_import_without_dependencies,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n=== Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("\n🎉 All structure tests passed! eval_protocol is properly configured.")
        return 0
    else:
        print(f"\n❌ {failed} tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
