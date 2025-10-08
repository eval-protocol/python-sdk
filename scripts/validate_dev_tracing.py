import os
import time
import logging
import requests

from eval_protocol import FireworksTracingHttpHandler


def main():
    gateway = os.getenv("FW_TRACING_GATEWAY_BASE_URL")
    if not gateway:
        # default to deployed dev gateway
        gateway = "https://metadata-gateway-dev-644257448872.us-central1.run.app"
    rollout_id = os.getenv("EP_ROLLOUT_ID", f"sdk-dev-{int(time.time())}")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(FireworksTracingHttpHandler(gateway_base_url=gateway))

    logger = logging.getLogger("eval_protocol.sdk.validate")

    logger.info(
        "SDK sending structured log to dev gateway",
        extra={
            "rollout_id": rollout_id,
            "program": "eval_protocol",
            "status": "completed",
            "experiment_id": "dev-exp",
            "run_id": "dev-run",
            "metadata": {"source": "sdk-validate"},
        },
    )

    # Poll fetch with retries for indexing
    params = {
        "tags": [f"rollout_id:{rollout_id}"],
        "program": "eval_protocol",
        "limit": 10,
        "hours_back": 1,
    }
    total = 0
    for _ in range(20):
        r = requests.get(f"{gateway}/logs", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        total = int(data.get("total_entries") or 0)
        if total > 0:
            print("Fetched entries:", total)
            for e in data.get("entries", []):
                print({k: e.get(k) for k in ["timestamp", "severity", "program", "status", "message", "tags"]})
            break
        time.sleep(3)
    if total == 0:
        print("Fetched entries: 0 (after retries)")


if __name__ == "__main__":
    main()
