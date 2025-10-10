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
    print(f"🔍 Debug mode: {args.debug}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    # Setup Elasticsearch based on flags
    elasticsearch_config = None
    try:
        if getattr(args, "use_env_elasticsearch_config", False):
            # Use environment variables for configuration
            print("⚙️ Using environment variables for Elasticsearch config")
            from eval_protocol.pytest.remote_rollout_processor import (
                create_elasticsearch_config_from_env,
            )

            elasticsearch_config = create_elasticsearch_config_from_env()
            # Ensure index exists with correct mapping, mirroring Docker setup path
            try:
                from eval_protocol.log_utils.elasticsearch_index_manager import (
                    ElasticsearchIndexManager,
                )

                index_manager = ElasticsearchIndexManager(
                    elasticsearch_config.url,
                    elasticsearch_config.index_name,
                    elasticsearch_config.api_key,
                )
                created = index_manager.create_logging_index_mapping()
                if created:
                    print(
                        f"🧭 Verified Elasticsearch index '{elasticsearch_config.index_name}' mapping (created or already correct)"
                    )
                else:
                    print(
                        f"⚠️ Could not verify/create mapping for index '{elasticsearch_config.index_name}'. Searches may behave unexpectedly."
                    )
            except Exception as e:
                print(f"⚠️ Failed to ensure index mapping via IndexManager: {e}")
        elif not getattr(args, "disable_elasticsearch_setup", False):
            # Default behavior: start or connect to local Elasticsearch via Docker helper
            from eval_protocol.pytest.elasticsearch_setup import ElasticsearchSetup

            print("🧰 Auto-configuring local Elasticsearch (Docker)")
            elasticsearch_config = ElasticsearchSetup().setup_elasticsearch()
        else:
            print("🚫 Elasticsearch setup disabled; running without Elasticsearch integration")
    except Exception as e:
        print(f"❌ Failed to configure Elasticsearch: {e}")
        return 1

    try:
        serve_logs(port=args.port, elasticsearch_config=elasticsearch_config, debug=args.debug)
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
