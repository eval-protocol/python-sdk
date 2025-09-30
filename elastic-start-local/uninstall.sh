#!/bin/sh
# Uninstall script for start-local
# More information: https://github.com/elastic/start-local
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ask_confirmation() {
    echo "Do you confirm? (yes/no)"
    read -r answer
    case "$answer" in
        yes|y|Y|Yes|YES)
            return 0  # true
            ;;
        no|n|N|No|NO)
            return 1  # false
            ;;
        *)
            echo "Please answer yes or no."
            ask_confirmation  # Ask again if the input is invalid
            ;;
    esac
}

cd "${SCRIPT_DIR}"
if [ ! -e "docker-compose.yml" ]; then
  echo "Error: I cannot find the docker-compose.yml file"
  echo "I cannot uninstall start-local."
fi
if [ ! -e ".env" ]; then
  echo "Error: I cannot find the .env file"
  echo "I cannot uninstall start-local."
fi
echo "This script will uninstall start-local."
echo "All data will be deleted and cannot be recovered."
if ask_confirmation; then
  docker compose rm -fsv
  docker compose down -v
  rm docker-compose.yml .env uninstall.sh start.sh stop.sh config/telemetry.yml
  if [ -z "$(ls -A config)" ]; then
    rm -d config
  fi
  echo
  echo "Do you want to remove the following Docker images?"
  echo "- docker.elastic.co/elasticsearch/elasticsearch:9.1.4-arm64"
  if ask_confirmation; then
    if docker rmi "docker.elastic.co/elasticsearch/elasticsearch:9.1.4-arm64" >/dev/null 2>&1; then
      echo "Image docker.elastic.co/elasticsearch/elasticsearch:9.1.4-arm64 removed successfully"
    else
      echo "Failed to remove image docker.elastic.co/elasticsearch/elasticsearch:9.1.4-arm64. It might be in use."
    fi
  fi
  echo "Start-local successfully removed"
fi
