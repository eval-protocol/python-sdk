#!/usr/bin/env python3
"""Test script to verify that models.py can be imported without Pydantic errors."""

try:
    from eval_protocol.models import ErrorInfo, Status

    print("✅ Successfully imported ErrorInfo and Status from models.py")

    # Test creating instances
    error_info = ErrorInfo.termination_reason("test_reason")
    print(f"✅ Successfully created ErrorInfo: {error_info}")

    status = Status.rollout_running()
    print(f"✅ Successfully created Status: {status}")

    print("\n🎉 All tests passed! The Pydantic error has been resolved.")

except Exception as e:
    print(f"❌ Error importing models: {e}")
    import traceback

    traceback.print_exc()
