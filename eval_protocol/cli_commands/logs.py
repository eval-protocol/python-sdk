"""
CLI command for serving logs with file watching and real-time updates.
"""

import sys
from pathlib import Path

import os
from ..utils.logs_server import serve_logs
from ..event_bus.sqlite_event_bus_database import DatabaseCorruptedError, _backup_and_remove_database


def _handle_database_corruption(db_path: str) -> bool:
    """
    Handle database corruption by prompting user to fix it.

    Args:
        db_path: Path to the corrupted database

    Returns:
        True if user chose to fix and database was reset, False otherwise
    """
    print("\n" + "=" * 60)
    print("⚠️  DATABASE CORRUPTION DETECTED")
    print("=" * 60)
    print(f"\nThe database file at:\n  {db_path}\n")
    print("appears to be corrupted or is not a valid SQLite database.")
    print("\nThis can happen due to:")
    print("  • Incomplete writes during a crash")
    print("  • Concurrent access issues")
    print("  • File system errors")
    print("\n" + "-" * 60)
    print("Would you like to automatically fix this?")
    print("  • The corrupted file will be backed up")
    print("  • A fresh database will be created")
    print("  • You will lose existing log data, but can continue using the tool")
    print("-" * 60)

    try:
        response = input("\nFix database automatically? [Y/n]: ").strip().lower()
        if response in ("", "y", "yes"):
            _backup_and_remove_database(db_path)
            print("\n✅ Database has been reset. Restarting server...")
            return True
        else:
            print("\n❌ Database repair cancelled.")
            print(f"   You can manually delete the corrupted file: {db_path}")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Database repair cancelled.")
        return False


def _is_database_corruption_error(error: Exception) -> tuple[bool, str]:
    """
    Check if an exception is related to database corruption.

    Returns:
        Tuple of (is_corruption_error, db_path)
    """
    error_str = str(error).lower()
    corruption_indicators = [
        "file is not a database",
        "database disk image is malformed",
        "unable to open database file",
    ]

    for indicator in corruption_indicators:
        if indicator in error_str:
            # Try to find the database path
            from ..directory_utils import find_eval_protocol_dir

            try:
                eval_protocol_dir = find_eval_protocol_dir()
                db_path = os.path.join(eval_protocol_dir, "logs.db")
                return True, db_path
            except Exception:
                return True, ""

    # Check if it's a DatabaseCorruptedError
    if isinstance(error, DatabaseCorruptedError):
        return True, error.db_path

    return False, ""


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

    max_retries = 2
    for attempt in range(max_retries):
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
            is_corruption, db_path = _is_database_corruption_error(e)

            if is_corruption and db_path and attempt < max_retries - 1:
                if _handle_database_corruption(db_path):
                    # User chose to fix, retry
                    continue
                else:
                    # User declined fix
                    return 1

            print(f"❌ Error starting server: {e}")
            return 1

    return 1
