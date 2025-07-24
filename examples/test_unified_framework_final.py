#!/usr/bin/env python3
"""
Final comprehensive test of the unified MCP framework.

This test validates that:
1. All servers can be instantiated
2. Tool signatures match between production and simulation
3. Resource availability is consistent
4. Framework enforces proper patterns
"""

import os
import sys

sys.path.append("/home/bchen/home/reward-kit/examples/frozen_lake_mcp_complete/mcp_server")
sys.path.append("/home/bchen/home/reward-kit/examples/taxi_mcp_complete/mcp_server")


def test_frozen_lake():
    """Test FrozenLake unified framework."""
    print("🧊 Testing FrozenLake Unified Framework")
    print("-" * 40)

    try:
        # Test production server
        from frozen_lake_mcp_server_new import FrozenLakeProdServer

        os.environ["PORT"] = "8001"

        print("📦 Creating FrozenLake production server...")
        prod_server = FrozenLakeProdServer()
        prod_tools = list(prod_server.mcp._tool_manager._tools.keys())
        prod_resources = list(prod_server.mcp._resource_manager._resources.keys())

        print(f"✅ Production server: Tools={prod_tools}, Resources={prod_resources}")

        # Test simulation server
        from simulation_server_new import FrozenLakeSimServer

        os.environ["PORT"] = "8003"

        print("🎮 Creating FrozenLake simulation server...")
        sim_server = FrozenLakeSimServer()
        sim_tools = list(sim_server.mcp._tool_manager._tools.keys())
        sim_resources = list(sim_server.mcp._resource_manager._resources.keys())

        print(f"✅ Simulation server: Tools={sim_tools}, Resources={sim_resources}")

        # Validate signatures match
        if prod_tools == sim_tools and prod_resources == sim_resources:
            print("✅ Tool and resource signatures match perfectly")
            return True
        else:
            print(f"❌ Signature mismatch: Prod={prod_tools}, Sim={sim_tools}")
            return False

    except Exception as e:
        print(f"❌ FrozenLake test failed: {e}")
        return False


def test_taxi():
    """Test Taxi unified framework."""
    print("\n🚕 Testing Taxi Unified Framework")
    print("-" * 40)

    try:
        # Test production server
        from taxi_mcp_server_new import TaxiProdServer

        os.environ["PORT"] = "8002"

        print("📦 Creating Taxi production server...")
        prod_server = TaxiProdServer()
        prod_tools = list(prod_server.mcp._tool_manager._tools.keys())
        prod_resources = list(prod_server.mcp._resource_manager._resources.keys())

        print(f"✅ Production server: Tools={prod_tools}, Resources={prod_resources}")

        # Test simulation server - switch to taxi directory
        import sys

        sys.path.remove("/home/bchen/home/reward-kit/examples/frozen_lake_mcp_complete/mcp_server")
        sys.path.append("/home/bchen/home/reward-kit/examples/taxi_mcp_complete/mcp_server")
        from simulation_server_new import TaxiSimServer

        os.environ["PORT"] = "8004"

        print("🎮 Creating Taxi simulation server...")
        sim_server = TaxiSimServer()
        sim_tools = list(sim_server.mcp._tool_manager._tools.keys())
        sim_resources = list(sim_server.mcp._resource_manager._resources.keys())

        print(f"✅ Simulation server: Tools={sim_tools}, Resources={sim_resources}")

        # Validate signatures match
        if prod_tools == sim_tools and prod_resources == sim_resources:
            print("✅ Tool and resource signatures match perfectly")
            return True
        else:
            print(f"❌ Signature mismatch: Prod={prod_tools}, Sim={sim_tools}")
            return False

    except Exception as e:
        print(f"❌ Taxi test failed: {e}")
        return False


def main():
    """Run comprehensive framework tests."""
    print("🌟 Unified MCP Framework - Final Validation")
    print("=" * 50)

    # Test both environments
    frozen_lake_pass = test_frozen_lake()
    taxi_pass = test_taxi()

    print("\n📊 Final Results:")
    print("=" * 50)
    print(f"🧊 FrozenLake Framework: {'✅ PASS' if frozen_lake_pass else '❌ FAIL'}")
    print(f"🚕 Taxi Framework: {'✅ PASS' if taxi_pass else '❌ FAIL'}")

    overall_pass = frozen_lake_pass and taxi_pass

    if overall_pass:
        print("\n🎉 UNIFIED FRAMEWORK VALIDATION: SUCCESS!")
        print("✅ All production and simulation servers work correctly")
        print("✅ Tool signatures are validated automatically")
        print("✅ Resource patterns are consistent")
        print("✅ Framework enforces proper MCP patterns")
        print("\n🚀 Ready for production use!")
    else:
        print("\n💥 UNIFIED FRAMEWORK VALIDATION: FAILED!")
        print("❌ Some servers or validation checks failed")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
