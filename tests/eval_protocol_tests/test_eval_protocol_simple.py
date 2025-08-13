#!/usr/bin/env python3
"""
Simple test to verify that eval_protocol imports work correctly.
This can be run independently without pytest or complex dependencies.
"""

import os
import sys
import traceback


def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")
    try:
        # Add project root to path so we can import without installing
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        sys.path.insert(0, project_root)

        # Test importing both packages
        print("  Importing eval_protocol...")
        import eval_protocol

        print("  ✓ eval_protocol imported successfully")

        print("  Importing eval_protocol...")

        print("  ✓ eval_protocol imported successfully")

        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_version_consistency():
    """Test that versions are consistent."""
    print("\nTesting version consistency...")
    try:
        import eval_protocol

        rk_version = getattr(eval_protocol, "__version__", "unknown")
        rp_version = getattr(eval_protocol, "__version__", "unknown")

        print(f"  eval_protocol version: {rk_version}")
        print(f"  eval_protocol version: {rp_version}")

        if rk_version == rp_version:
            print("  ✓ Versions match")
            return True
        else:
            print("  ✗ Versions don't match")
            return False
    except Exception as e:
        print(f"  ✗ Version check failed: {e}")
        return False


def test_all_exports():
    """Test that __all__ exports are consistent."""
    print("\nTesting __all__ exports...")
    try:
        import eval_protocol

        rk_all = getattr(eval_protocol, "__all__", [])
        rp_all = getattr(eval_protocol, "__all__", [])

        print(f"  eval_protocol.__all__ has {len(rk_all)} items")
        print(f"  eval_protocol.__all__ has {len(rp_all)} items")

        if rk_all == rp_all:
            print("  ✓ __all__ exports match")
            print(f"  Common exports: {rk_all[:5]}..." if len(rk_all) > 5 else f"  All exports: {rk_all}")
            return True
        else:
            print("  ✗ __all__ exports don't match")
            print(f"  Difference: {set(rk_all) - set(rp_all)} vs {set(rp_all) - set(rk_all)}")
            return False
    except Exception as e:
        print(f"  ✗ __all__ export check failed: {e}")
        return False


def test_specific_imports():
    """Test specific imports that should work."""
    print("\nTesting specific imports...")
    success = True

    # Test core classes/functions that should always be available
    test_items = [
        "Message",
        "MetricResult",
        "EvaluateResult",
        "RewardFunction",
        "reward_function",
        "load_jsonl",
    ]

    for item in test_items:
        try:
            print(f"  Testing {item}...")

            # Try importing from both packages
            exec(f"from eval_protocol import {item}")
            exec(f"from eval_protocol import {item}")

            # Check if they're the same object
            rk_obj = eval(f"__import__('eval_protocol', fromlist=['{item}']).{item}")
            rp_obj = eval(f"__import__('eval_protocol', fromlist=['{item}']).{item}")

            if rk_obj is rp_obj:
                print(f"    ✓ {item} is consistent between packages")
            else:
                print(f"    ✗ {item} differs between packages")
                success = False

        except ImportError as e:
            print(f"    ⚠ {item} not available (expected for some items): {e}")
        except Exception as e:
            print(f"    ✗ {item} test failed: {e}")
            success = False

    return success


def test_submodule_access():
    """Test that submodules can be accessed through both packages."""
    print("\nTesting submodule access...")
    success = True

    submodules = ["models", "auth", "config", "rewards"]

    for submodule in submodules:
        try:
            print(f"  Testing {submodule}...")

            # Try accessing submodules
            rk_sub = getattr(__import__("eval_protocol"), submodule, None)
            rp_sub = getattr(__import__("eval_protocol"), submodule, None)

            if rk_sub is not None and rp_sub is not None:
                if rk_sub is rp_sub:
                    print(f"    ✓ {submodule} is consistent between packages")
                else:
                    print(f"    ✗ {submodule} differs between packages")
                    success = False
            else:
                print(f"    ⚠ {submodule} not available in one or both packages")

        except Exception as e:
            print(f"    ✗ {submodule} test failed: {e}")
            success = False

    return success


def test_package_structure():
    """Test that both packages are discoverable."""
    print("\nTesting package structure...")
    try:
        from setuptools import find_packages

        packages = find_packages(include=["eval_protocol*", "eval_protocol*"])

        rk_packages = [p for p in packages if p.startswith("eval_protocol")]
        rp_packages = [p for p in packages if p == "eval_protocol"]

        print(f"  Found {len(rk_packages)} eval_protocol packages")
        print(f"  Found {len(rp_packages)} eval_protocol packages")

        if "eval_protocol" in packages and "eval_protocol" in packages:
            print("  ✓ Both main packages are discoverable")
            return True
        else:
            print("  ✗ Missing main packages")
            return False

    except Exception as e:
        print(f"  ✗ Package structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Testing eval_protocol Integration ===")

    tests = [
        test_basic_imports,
        test_version_consistency,
        test_all_exports,
        test_specific_imports,
        test_submodule_access,
        test_package_structure,
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
        print("\n🎉 All tests passed! eval_protocol is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
