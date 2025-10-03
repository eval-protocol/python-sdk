#!/usr/bin/env python3
"""
FrozenLake MCP-Gym Server

This script launches the FrozenLake MCP-Gym server using the proper MCP-Gym framework.
Compatible with CondaServerProcessManager for isolated execution.

Usage:
    python server.py --port 9004 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory first for local imports (frozen_lake_mcp)
sys.path.insert(0, str(Path(__file__).parent))

# Add eval_protocol parent to path, but use append to avoid priority conflicts
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from frozen_lake_mcp import FrozenLakeMcp


def main():
    """Run the FrozenLake MCP server."""
    parser = argparse.ArgumentParser(description="FrozenLake MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport protocol to use",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the environment")

    args = parser.parse_args()

    # Set environment variable for HTTP port (required by FastMCP)
    if args.transport == "streamable-http":
        os.environ["PORT"] = str(args.port)

    # Create and run server
    server = FrozenLakeMcp(seed=args.seed)

    print(f"🚀 Starting FrozenLake MCP server on port {args.port}")
    print(f"🌱 Seed: {args.seed}")
    print(f"📡 Transport: {args.transport}")

    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
