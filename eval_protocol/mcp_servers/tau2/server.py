#!/usr/bin/env python3
"""
General MCP-Gym Server (τ²-Bench domains)

This script launches MCP-Gym servers for different τ²-Bench domains.
It can serve airline, mock, or retail domains based on the --domain argument.
Compatible with CondaServerProcessManager for isolated execution.

Usage:
    python server.py --domain airline --port 9100 --seed 42
    python server.py --domain mock --port 9101 --seed 42
    python server.py --domain retail --port 9102 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory first for local imports (tau2_mcp)
sys.path.insert(0, str(Path(__file__).parent))

# Add eval_protocol parent to path, but use append to avoid priority conflicts
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tau2_mcp import AirlineDomainMcp, MockDomainMcp, RetailDomainMcp


def main():
    """Run the specified domain MCP server."""
    parser = argparse.ArgumentParser(description="General τ²-Bench MCP Server")
    parser.add_argument(
        "--domain",
        choices=["airline", "mock", "retail"],
        default="airline",
        help="Domain to serve (airline, mock, or retail)",
    )
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport protocol to use",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the environment")
    parser.add_argument(
        "--max-workers", type=int, default=None, help="Maximum number of workers for the ThreadPoolExecutor"
    )

    args = parser.parse_args()

    # Set environment variable for HTTP port (required by FastMCP)
    if args.transport == "streamable-http":
        # TODO: Benny to fix this later
        os.environ["PORT"] = str(args.port)

    # Create server based on domain
    domain_servers = {
        "airline": AirlineDomainMcp,
        "mock": MockDomainMcp,
        "retail": RetailDomainMcp,
    }

    domain_icons = {
        "airline": "✈️",
        "mock": "🧪",
        "retail": "🛒",
    }

    server_class = domain_servers[args.domain]
    server = server_class(seed=args.seed, max_workers=args.max_workers)

    print(f"{domain_icons[args.domain]} Starting {args.domain.title()} MCP server on port {args.port}")
    print(f"🌱 Seed: {args.seed}")
    print(f"📡 Transport: {args.transport}")

    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
