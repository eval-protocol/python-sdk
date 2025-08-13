"""
CLI command for serving logs with file watching and real-time updates.
"""

import sys
from pathlib import Path

from ..utils.logs_server import serve_logs


def logs_command(args):
    """Serve logs with file watching and real-time updates"""

    port = args.port
    print("🚀 Starting Eval Protocol Logs Server")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"🔌 WebSocket: ws://localhost:{port}/ws")
    print(f"👀 Watching paths: {['current directory']}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        serve_logs(port=args.port)
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
