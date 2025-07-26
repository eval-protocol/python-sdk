#!/usr/bin/env python3
"""
Simple test script to trigger seed logging in the MCP server.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import eval_protocol as ep


async def test_seed_logging():
    """Test seed logging by calling the MCP server."""
    print("🧪 Testing seed logging...")

    # Create a simple dataset with explicit seed
    dataset = [
        {
            "id": "test_001",
            "system_prompt": "You are playing FrozenLake. Use lake_move tool.",
            "user_prompt_template": "Current state: {observation}. Make a move.",
            "environment_context": {"seed": 42},
        }
    ]

    try:
        # Create environment pointing to our server
        print("🔌 Connecting to server...")
        envs = ep.make("http://localhost:9600/mcp/", dataset=dataset, model_id="test")
        print(f"✅ Created envs: {len(envs.sessions)} sessions")

        # Reset environments to trigger session creation
        print("🔄 Resetting environments...")
        observations, tool_schemas, system_prompts = await envs.reset()
        print(f"✅ Reset complete")
        print(f"📊 Observations: {observations}")
        print(f"🛠️  Tool schemas: {len(tool_schemas[0])} tools available")

        # Make a tool call to trigger more logging
        print("🎮 Making a test move...")
        from eval_protocol.mcp.types import MCPToolCall

        # Create a simple static action
        tool_call = MCPToolCall(tool_name="lake_move", arguments={"action": "RIGHT"})

        # This would normally be done by the policy, but let's trigger it manually
        # by calling the tool directly through the session
        session = envs.sessions[0]
        print(f"📞 Calling tool on session: {session.session_id}")

        # Call the tool through the MCP session
        result = await envs.session_manager.connection_manager.call_tool(
            session, tool_call.tool_name, tool_call.arguments
        )
        print(f"🎯 Tool call result: {result}")

        await envs.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_seed_logging())
