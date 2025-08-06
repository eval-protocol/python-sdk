"""
CLI command for serving logs with file watching and real-time updates.
"""

import sys
from pathlib import Path

from ..utils.logs_server import serve_logs


def logs_command(args):
    """Serve logs with file watching and real-time updates"""

    print(f"🚀 Starting Eval Protocol Logs Server")
    print(f"🌐 URL: http://localhost:8000")
    print(f"🔌 WebSocket: ws://localhost:8000/ws")
    print(f"👀 Watching paths: {['current directory']}")
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
