"""
CLI command for serving logs with file watching and real-time updates.
"""

import sys
from pathlib import Path

import os
from ..utils.logs_server import serve_logs


def logs_command(args):
    """Serve logs with file watching and real-time updates"""

    port = args.port
    print("🚀 Starting Eval Protocol Logs Server")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"🔌 WebSocket: ws://localhost:{port}/ws")
    print(f"👀 Watching paths: {['current directory']}")
    print(f"🔍 Debug mode: {args.debug}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    # Backend selection: Fireworks first when API key present, unless overridden
    use_fireworks = False
    if getattr(args, "use_fireworks", False):
        use_fireworks = True
    elif getattr(args, "use_elasticsearch", False):
        use_fireworks = False
    else:
        use_fireworks = bool(os.environ.get("FIREWORKS_API_KEY"))

    # Setup backend configs
    elasticsearch_config = None
    # Prefer explicit FW_TRACING_GATEWAY_BASE_URL, then GATEWAY_URL from env (remote validation),
    # finally default to public tracing.fireworks.ai
    fireworks_base_url = (
        os.environ.get("FW_TRACING_GATEWAY_BASE_URL")
        or os.environ.get("GATEWAY_URL")
        or "https://tracing.fireworks.ai"
    )

    try:
        serve_logs(
            port=args.port,
            elasticsearch_config=elasticsearch_config,
            debug=args.debug,
            backend="fireworks" if use_fireworks else "elasticsearch",
            fireworks_base_url=fireworks_base_url if use_fireworks else None,
        )
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
