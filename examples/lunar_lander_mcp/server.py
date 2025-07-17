#!/usr/bin/env python3
"""
LunarLander MCP-Gym Server

This script launches the LunarLander MCP-Gym server using the proper MCP-Gym framework.
Compatible with CondaServerProcessManager for isolated execution.

Usage:
    python server.py --port 9004 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

from lunar_lander_mcp import LunarLanderMcp


def main():
    """Run the LunarLander MCP server."""
    parser = argparse.ArgumentParser(description="LunarLander MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport protocol to use",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transport"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for the environment"
    )

    args = parser.parse_args()

    # Set environment variable for HTTP port (required by FastMCP)
    if args.transport == "streamable-http":
        # TODO: Benny to fix this later
        os.environ["PORT"] = str(args.port)

    # Create and run server
    server = LunarLanderMcp(seed=args.seed)

    print(f"🚀 Starting LunarLander MCP server on port {args.port}")
    print(f"🌱 Seed: {args.seed}")
    print(f"📡 Transport: {args.transport}")

    server.run(transport=args.transport)


if __name__ == "__main__":
    main() 