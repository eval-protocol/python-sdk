"""
CLI command for serving logs with file watching and real-time updates.
"""

import sys
from pathlib import Path

from ..utils.logs_server import serve_logs


def logs_command(args):
    """Serve logs with file watching and real-time updates"""

    # Parse watch paths
    watch_paths = None
    if args.watch_paths:
        watch_paths = args.watch_paths.split(",")
        watch_paths = [path.strip() for path in watch_paths if path.strip()]

    print(f"🚀 Starting Eval Protocol Logs Server")
    print(f"🌐 URL: http://{args.host}:{args.port}")
    print(f"🔌 WebSocket: ws://{args.host}:{args.port}/ws")
    print(f"👀 Watching paths: {watch_paths or ['current directory']}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        serve_logs()
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
