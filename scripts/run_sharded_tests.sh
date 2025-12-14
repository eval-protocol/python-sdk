#!/usr/bin/env bash
# Script to run a shard of tests for parallel CI execution
# Usage: ./scripts/run_sharded_tests.sh <shard> <total_shards> [--dry-run]
# Example: ./scripts/run_sharded_tests.sh 1 4
# Example: ./scripts/run_sharded_tests.sh 1 4 --dry-run

set -e

SHARD=${1:-1}
TOTAL_SHARDS=${2:-4}
DRY_RUN=${3:-""}

if [ "$SHARD" -lt 1 ] || [ "$SHARD" -gt "$TOTAL_SHARDS" ]; then
	echo "Error: Shard must be between 1 and $TOTAL_SHARDS"
	exit 1
fi

# Collect all test files, excluding ignored ones
TEST_FILES=$(find tests -name "test_*.py" \
	! -path "tests/test_batch_evaluation.py" \
	! -path "tests/pytest/test_frozen_lake.py" \
	! -path "tests/pytest/test_lunar_lander.py" \
	! -path "tests/pytest/test_tau_bench_airline.py" \
	! -path "tests/pytest/test_apps_coding.py" \
	! -path "tests/test_tau_bench_airline_smoke.py" \
	! -path "tests/pytest/test_svgbench.py" \
	! -path "tests/pytest/test_livesvgbench.py" \
	! -path "tests/remote_server/test_remote_fireworks.py" \
	! -path "tests/remote_server/test_remote_fireworks_propagate_status.py" \
	! -path "tests/logging/test_elasticsearch_direct_http_handler.py" \
	| sort)

# Count total files
TOTAL_FILES=$(echo "$TEST_FILES" | wc -l | tr -d ' ')

# Calculate start and end line numbers for this shard (1-indexed for sed)
FILES_PER_SHARD=$(( (TOTAL_FILES + TOTAL_SHARDS - 1) / TOTAL_SHARDS ))
START_LINE=$(( (SHARD - 1) * FILES_PER_SHARD + 1 ))
END_LINE=$(( START_LINE + FILES_PER_SHARD - 1 ))
if [ $END_LINE -gt $TOTAL_FILES ]; then
	END_LINE=$TOTAL_FILES
fi

# Get files for this shard using sed
SHARD_FILES=$(echo "$TEST_FILES" | sed -n "${START_LINE},${END_LINE}p")
SHARD_COUNT=$(echo "$SHARD_FILES" | grep -c . || echo 0)

echo "========================================"
echo "Running shard $SHARD of $TOTAL_SHARDS"
echo "========================================"
echo "Total test files: $TOTAL_FILES"
echo "Files per shard: ~$FILES_PER_SHARD"
echo "Files in this shard: $SHARD_COUNT"
echo "Line range: $START_LINE to $END_LINE"
echo "----------------------------------------"
echo "Files:"
echo "$SHARD_FILES" | while read -r f; do
	echo "  $f"
done
echo "----------------------------------------"

if [ "$SHARD_COUNT" -eq 0 ] || [ -z "$SHARD_FILES" ]; then
	echo "No files in this shard, skipping tests"
	exit 0
fi

# Check if --dry-run flag is passed
if [ "$DRY_RUN" = "--dry-run" ]; then
	echo "Dry run mode - not executing tests"
	exit 0
fi

# Run tests for this shard
# shellcheck disable=SC2086
exec pytest \
	-n auto \
	--ignore=eval_protocol/benchmarks/ \
	--ignore=eval_protocol/quickstart/ \
	-v --durations=10 \
	$SHARD_FILES
