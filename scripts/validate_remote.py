import os
import sys
import time
import requests


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        print(f"Missing required env var: {var_name}", file=sys.stderr)
        sys.exit(1)
    return value


def require_logs_endpoints(base_url: str) -> None:
    try:
        r = requests.get(f"{base_url}/openapi.json", timeout=30)
        if not r.ok:
            print("OpenAPI schema unavailable", file=sys.stderr)
            sys.exit(1)
        paths = r.json().get("paths", {})
        ok = any(p.startswith("/logs") or p.startswith("/v1/logs") for p in paths.keys())
        if not ok:
            print("/logs endpoints not present on deployment", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Failed to check OpenAPI: {e}", file=sys.stderr)
        sys.exit(1)


def post_chat_completion(base_url: str, api_key: str, rollout_id: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    now = int(time.time())
    url = (
        f"{base_url}/rollout_id/{rollout_id}/"
        f"invocation_id/inv{now}/"
        f"experiment_id/remote-validate/"
        f"run_id/run-1/"
        f"row_id/row-1/"
        f"chat/completions"
    )
    body = {
        "model": "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct",
        "messages": [{"role": "user", "content": "Say 'ok' if you can read this."}],
        "temperature": 0.1,
    }
    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code != 200:
        print(f"Chat completion failed: {r.status_code} {r.text[:500]}", file=sys.stderr)
        sys.exit(1)
    print("chat: ok")


def wait_for_traces(base_url: str, api_key: str, rollout_id: str, max_attempts: int = 8) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "tags": [f"rollout_id:{rollout_id}"],
        "limit": 10,
        "hours_back": 6,
    }
    url = f"{base_url}/traces"
    for attempt in range(1, max_attempts + 1):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            total = int(data.get("total_traces") or 0)
            print(f"traces: ok total_traces={total}")
            if total > 0:
                return
        elif r.status_code != 404 and r.status_code != 401:
            print(f"Traces fetch failed: {r.status_code} {r.text[:500]}", file=sys.stderr)
            sys.exit(1)
        sleep_s = min(2 ** (attempt - 1), 10)
        time.sleep(sleep_s)
    print("Traces not available after retries (indexing delay?)", file=sys.stderr)
    sys.exit(1)


def validate_logs_endpoints(base_url: str, rollout_id: str) -> None:
    require_logs_endpoints(base_url)

    # Ingest a structured log
    payload = {
        "program": "eval_protocol",
        "status": "completed",
        "message": "Remote validation run finished",
        "tags": [f"rollout_id:{rollout_id}", "experiment_id:remote", "run_id:test"],
        "metadata": {"dataset": "AIME"},
        "extras": {"num_examples": 3},
    }
    r = requests.post(f"{base_url}/logs", json=payload, timeout=30)
    if r.status_code != 200:
        print(f"logs ingest failed: {r.status_code} {r.text[:500]}", file=sys.stderr)
        sys.exit(1)
    print("logs ingest: ok")

    # Retrieve logs (retry for indexing)
    params = {
        "tags": [f"rollout_id:{rollout_id}"],
        "program": "eval_protocol",
        "hours_back": 1,
        "limit": 10,
    }
    total = 0
    for attempt in range(1, 12):
        rr = requests.get(f"{base_url}/logs", params=params, timeout=30)
        if rr.status_code == 200:
            data = rr.json()
            total = int(data.get("total_entries") or 0)
            if total > 0:
                print(f"logs fetch: ok total_entries={total}")
                break
        sleep_s = min(2 ** (attempt - 1), 10)
        time.sleep(sleep_s)
    if total == 0:
        print("logs fetch: no entries found within retry window", file=sys.stderr)
        sys.exit(1)


def main():
    base_url = require_env("GATEWAY_URL")
    api_key = require_env("FIREWORKS_API_KEY")
    rollout_id = f"r{int(time.time())}"

    print(f"Gateway: {base_url}")
    print(f"Rollout: rollout_id:{rollout_id}")

    post_chat_completion(base_url, api_key, rollout_id)
    wait_for_traces(base_url, api_key, rollout_id)
    validate_logs_endpoints(base_url, rollout_id)

    print("remote validation: SUCCESS")


if __name__ == "__main__":
    main()
