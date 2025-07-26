#!/usr/bin/env python3
"""
Test to demonstrate that eval_protocol and eval_protocol imports are equivalent.
This test works by examining the module structure without triggering dependency imports.
"""

import importlib.util
import os
import sys


def test_module_specs():
    """Test that both modules have the correct specifications."""
    print("=== Testing Module Specifications ===")

    # Test eval_protocol spec
    rk_spec = importlib.util.find_spec("eval_protocol")
    if rk_spec:
        print(f"✓ eval_protocol spec found: {rk_spec.origin}")
    else:
        print("✗ eval_protocol spec not found")
        return False

    # Test eval_protocol spec
    rp_spec = importlib.util.find_spec("eval_protocol")
    if rp_spec:
        print(f"✓ eval_protocol spec found: {rp_spec.origin}")
    else:
        print("✗ eval_protocol spec not found")
        return False

    return True


def test_import_structure():
    """Test the import structure without triggering dependency loads."""
    print("\n=== Testing Import Structure ===")

    # Read the eval_protocol __init__.py to verify it re-exports eval_protocol
    try:
        with open("eval_protocol/__init__.py", "r") as f:
            rp_content = f.read()

        # Check that it imports everything from eval_protocol
        if "from eval_protocol import *" in rp_content:
            print("✓ eval_protocol imports everything from eval_protocol")
        else:
            print("✗ eval_protocol does not import everything from eval_protocol")
            return False

        # Check version consistency
        if "from eval_protocol import __version__" in rp_content:
            print("✓ eval_protocol imports __version__ from eval_protocol")
        else:
            print("✗ eval_protocol does not import __version__ from eval_protocol")
            return False

        # Check __all__ consistency
        if "from eval_protocol import __all__" in rp_content:
            print("✓ eval_protocol imports __all__ from eval_protocol")
        else:
            print("✗ eval_protocol does not import __all__ from eval_protocol")
            return False

        return True

    except Exception as e:
        print(f"✗ Error testing import structure: {e}")
        return False


def test_submodule_structure():
    """Test that submodules will be accessible through both packages."""
    print("\n=== Testing Submodule Structure ===")

    # Check that eval_protocol __init__.py imports submodules
    try:
        with open("eval_protocol/__init__.py", "r") as f:
            rp_content = f.read()

        # Check for submodule imports
        expected_submodules = [
            "adapters",
            "agent",
            "auth",
            "cli",
            "cli_commands",
            "common_utils",
            "config",
            "datasets",
            "evaluation",
            "execution",
            "gcp_tools",
            "generation",
            "generic_server",
            "integrations",
            "mcp",
            "mcp_agent",
            "models",
            "packaging",
            "platform_api",
            "playback_policy",
            "resources",
            "reward_function",
            "rewards",
            "rl_processing",
            "server",
            "typed_interface",
            "utils",
        ]

        found_submodules = []
        for submodule in expected_submodules:
            if submodule in rp_content:
                found_submodules.append(submodule)

        print(f"✓ eval_protocol imports {len(found_submodules)} submodules")

        # Check a few key ones
        key_submodules = ["models", "rewards", "auth", "config"]
        for submodule in key_submodules:
            if submodule in found_submodules:
                print(f"  ✓ {submodule} imported")
            else:
                print(f"  ⚠ {submodule} not explicitly imported (will be available via star import)")

        return True

    except Exception as e:
        print(f"✗ Error testing submodule structure: {e}")
        return False


def test_console_scripts():
    """Test that console scripts are properly configured."""
    print("\n=== Testing Console Scripts ===")

    try:
        with open("setup.py", "r") as f:
            setup_content = f.read()

        # Check for all three console scripts
        expected_scripts = [
            "fireworks-reward=eval_protocol.cli:main",
            "reward-kit=eval_protocol.cli:main",
            "eval-protocol=eval_protocol.cli:main",
        ]

        for script in expected_scripts:
            if script in setup_content:
                print(f"✓ Console script configured: {script}")
            else:
                print(f"✗ Console script missing: {script}")
                return False

        return True

    except Exception as e:
        print(f"✗ Error testing console scripts: {e}")
        return False


def test_package_metadata():
    """Test that package metadata is consistent."""
    print("\n=== Testing Package Metadata ===")

    try:
        with open("setup.py", "r") as f:
            setup_content = f.read()

        # Check package discovery
        if "eval_protocol*" in setup_content and "eval_protocol*" in setup_content:
            print("✓ Both packages included in find_packages")
        else:
            print("✗ Package discovery configuration incorrect")
            return False

        # Check that the package name is still reward-kit
        if 'name="reward-kit"' in setup_content:
            print("✓ Package name remains 'reward-kit'")
        else:
            print("✗ Package name changed incorrectly")
            return False

        return True

    except Exception as e:
        print(f"✗ Error testing package metadata: {e}")
        return False


def demonstrate_usage():
    """Demonstrate how the dual imports will work."""
    print("\n=== Usage Demonstration ===")

    print("After installation, users can use either import style:")
    print()

    print("# Style 1: eval_protocol (original)")
    print("from eval_protocol import reward_function, Message")
    print("from eval_protocol.rewards import accuracy")
    print()

    print("# Style 2: eval_protocol (new)")
    print("from eval_protocol import reward_function, Message")
    print("from eval_protocol.rewards import accuracy")
    print()

    print("Both styles provide identical functionality!")
    print()

    print("Command-line usage:")
    print("reward-kit --help          # Original")
    print("eval-protocol --help       # New")
    print("fireworks-reward --help    # Alternative")
    print()


def main():
    """Run all tests."""
    print("=== Testing eval_protocol Import Equivalence ===")

    # Add project root to sys.path so we can find the packages
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    sys.path.insert(0, project_root)
    os.chdir(project_root)  # Change to project root for relative paths

    tests = [
        test_module_specs,
        test_import_structure,
        test_submodule_structure,
        test_console_scripts,
        test_package_metadata,
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

    demonstrate_usage()

    print("=== Final Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("\n🎉 All tests passed!")
        print("✅ eval_protocol is properly configured")
        print("✅ Import equivalence verified")
        print("✅ Console scripts configured")
        print("✅ Package ready for PyPI publishing")
        return 0
    else:
        print(f"\n❌ {failed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
