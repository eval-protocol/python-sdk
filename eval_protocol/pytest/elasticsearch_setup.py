import os
import subprocess
import tempfile
import logging
from typing import Optional

from dotenv import load_dotenv
from eval_protocol.directory_utils import find_eval_protocol_dir
from eval_protocol.types.remote_rollout_processor import ElasticSearchConfig

logger = logging.getLogger(__name__)


class ElasticsearchSetupError(Exception):
    """Exception raised when Elasticsearch setup fails."""

    pass


class ElasticsearchSetup:
    """Handles Elasticsearch setup with retry logic for existing containers."""

    def __init__(self):
        self.eval_protocol_dir = find_eval_protocol_dir()

    def setup_elasticsearch(self) -> ElasticSearchConfig:
        """
        Set up Elasticsearch, handling both local and remote scenarios.
        Returns the ElasticSearchConfig for the running instance.
        """
        elastic_start_local_dir = os.path.join(self.eval_protocol_dir, "elastic-start-local")
        env_file_path = os.path.join(elastic_start_local_dir, ".env")

        # If elastic-start-local directory exists, use local script
        if os.path.exists(elastic_start_local_dir):
            return self._setup_local_elasticsearch(elastic_start_local_dir, env_file_path)

        # Otherwise, use remote curl command with retry logic
        return self._setup_remote_elasticsearch(env_file_path)

    def _setup_local_elasticsearch(self, elastic_start_local_dir: str, env_file_path: str) -> ElasticSearchConfig:
        """Set up Elasticsearch using local start.sh script."""
        from eval_protocol.utils.subprocess_utils import run_script_and_wait

        run_script_and_wait(
            script_name="start.sh",
            working_directory=elastic_start_local_dir,
            inherit_stdout=True,
        )
        return self._parse_elastic_env_file(env_file_path)

    def _setup_remote_elasticsearch(self, env_file_path: str) -> ElasticSearchConfig:
        """Set up Elasticsearch using remote curl command with retry logic."""
        max_retries = 2
        for attempt in range(max_retries):
            # Use a temporary file to capture output while also showing it in parent stdout
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
                temp_file_path = temp_file.name

            try:
                # Run the command and tee output to both stdout and temp file
                # Use set -o pipefail to ensure we get the return code of the first failing command
                process = subprocess.Popen(
                    [
                        "sh",
                        "-c",
                        f"set -o pipefail; curl -fsSL https://elastic.co/start-local | sh -s -- --esonly | tee {temp_file_path}",
                    ],
                    cwd=self.eval_protocol_dir,
                )
                returncode = process.wait()

                # Read the captured output
                with open(temp_file_path, "r") as f:
                    stdout = f.read()

                if returncode == 0:
                    return self._parse_elastic_env_file(env_file_path)

                # Check if container is already running and handle it
                if self._handle_existing_elasticsearch_container(stdout):
                    logger.info(f"Retrying Elasticsearch setup (attempt {attempt + 1}/{max_retries})")
                    continue

                # If we get here, it's a different error
                raise ElasticsearchSetupError(
                    f"Failed to start Elasticsearch (attempt {attempt + 1}/{max_retries}): {stdout}"
                )

            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

        raise ElasticsearchSetupError(f"Failed to start Elasticsearch after {max_retries} attempts")

    def _handle_existing_elasticsearch_container(self, output: str) -> bool:
        """
        Check if the curl command output indicates that the Elasticsearch container is already running.
        If so, stop the existing container and return True to indicate a retry is needed.
        """
        if "docker stop es-local-dev" in output:
            logger.info("Elasticsearch container 'es-local-dev' is already running. Stopping it...")
            try:
                subprocess.run(["docker", "stop", "es-local-dev"], check=True, capture_output=True, text=True)
                logger.info("Successfully stopped existing Elasticsearch container")
                return True  # Indicate retry is needed
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to stop existing container: {e}")
                return False
        return False

    def _parse_elastic_env_file(self, env_file_path: str) -> ElasticSearchConfig:
        """Parse ES_LOCAL_API_KEY and ES_LOCAL_URL from .env file."""
        loaded = load_dotenv(env_file_path)
        if not loaded:
            raise ElasticsearchSetupError("Failed to load .env file")

        api_key = os.getenv("ES_LOCAL_API_KEY")
        url = os.getenv("ES_LOCAL_URL")

        if not url or not api_key:
            raise ElasticsearchSetupError("Failed to parse ES_LOCAL_API_KEY and ES_LOCAL_URL from .env file")

        return ElasticSearchConfig(url=url, api_key=api_key)
