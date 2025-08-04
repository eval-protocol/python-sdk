#!/usr/bin/env python3
"""
Quick test to verify both production and simulation servers are working
"""

import asyncio
import json

import httpx


async def test_server(url: str, server_name: str) -> bool:
    """Test if an MCP server is responding correctly."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Test initialize
            response = await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                    },
                },
            )

            if response.status_code != 200:
                print(f"❌ {server_name}: Initialize failed ({response.status_code})")
                return False

            # Test tools list
            response = await client.post(
                url,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )

            if response.status_code != 200:
                print(f"❌ {server_name}: Tools list failed ({response.status_code})")
                return False

            data = response.json()
            tools = data.get("result", {}).get("tools", [])
            tool_names = [tool["name"] for tool in tools]

            print(f"✅ {server_name}: {tool_names}")
            return True

        except Exception as e:
            print(f"❌ {server_name}: Connection failed - {e}")
            return False


async def main():
    """Test all servers."""
    print("🧪 Testing Unified MCP Servers")
    print("=" * 40)

    servers = [
        ("http://localhost:8001/mcp/", "FrozenLake Production"),
        ("http://localhost:8002/mcp/", "Taxi Production"),
        ("http://localhost:8003/mcp/", "FrozenLake Simulation"),
    ]

    results = []
    for url, name in servers:
        result = await test_server(url, name)
        results.append((name, result))

    print("\n📊 Results:")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All servers are working correctly!")
    else:
        print("\n⚠️  Some servers need attention")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
