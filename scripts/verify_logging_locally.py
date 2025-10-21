import os
import sys
import time
import json
import logging
from typing import Any, Dict, List

import requests

from eval_protocol.log_utils.init import init_external_logging_from_env
from eval_protocol.log_utils.rollout_context import rollout_logging_context


def _now_rollout_id() -> str:
    return f"verify-{int(time.time())}"


def _detect_gateway_base_url() -> str:
    # Prefer explicit FW_TRACING_GATEWAY_BASE_URL, else GATEWAY_URL, else public default
    return os.getenv("FW_TRACING_GATEWAY_BASE_URL") or os.getenv("GATEWAY_URL") or "https://tracing.fireworks.ai"


def _detect_logs_endpoint(base_url: str) -> str:
    # Inspect OpenAPI and choose the correct logs endpoint
    try:
        import requests

        r = requests.get(f"{base_url.rstrip('/')}/openapi.json", timeout=5)
        if r.ok:
            paths = (r.json() or {}).get("paths", {})
            if any(p.startswith("/v1/logs") for p in paths.keys()):
                return "/v1/logs"
            if any(p.startswith("/logs") for p in paths.keys()):
                return "/logs"
    except Exception:
        pass
    return "/logs"


def verify_fireworks(rollout_id: str) -> int:
    base_url = _detect_gateway_base_url()
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set; cannot verify Fireworks")
        return 2

    # Emit two logs under rollout context
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    init_external_logging_from_env()
    # Detect and use the correct logs endpoint
    logs_ep = _detect_logs_endpoint(base_url)
    # Print handler info for diagnostics
    handlers = [type(h).__name__ for h in root.handlers]
    print(
        json.dumps(
            {
                "gateway_url": base_url,
                "logs_endpoint": logs_ep,
                "root_handlers": handlers,
            }
        )
    )

    logger = logging.getLogger("ep.verify.fireworks")
    for i in range(2):
        logger.info(
            f"verify fireworks message {i}",
            extra={
                "rollout_id": rollout_id,
                "experiment_id": "verify-exp",
                "run_id": "verify-run",
                "status_code": 101 if i == 0 else 200,
                "status_message": "RUNNING" if i == 0 else "COMPLETED",
            },
        )

    # Poll /logs for the rollout tag
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "tags": [f"rollout_id:{rollout_id}"],
        "program": "eval_protocol",
        "limit": 50,
        "hours_back": 6,
    }
    candidate_eps = [logs_ep, "/v1/logs" if logs_ep != "/v1/logs" else "/logs"]
    for _ in range(20):
        try:
            data: Dict[str, Any] = {}
            last_err: str | None = None
            for ep in candidate_eps:
                url = f"{base_url.rstrip('/')}{ep}"
                r = requests.get(url, headers=headers, params=params, timeout=15)
                if r.status_code == 404:
                    last_err = f"404 for {ep}"
                    continue
                r.raise_for_status()
                data = r.json() or {}
                break
            else:
                raise Exception(last_err or "all endpoints failed")
            entries: List[Dict[str, Any]] = data.get("entries", []) or []
            matched = [e for e in entries if any(t == f"rollout_id:{rollout_id}" for t in e.get("tags", []))]
            if matched:
                print(json.dumps({"total": len(matched), "sample": matched[:3]}, indent=2))
                return 0
        except Exception as e:
            print(f"Fireworks fetch error: {e}")
        time.sleep(2)

    print("No Fireworks entries found for rollout_id after retries")
    return 1


def verify_elasticsearch(rollout_id: str) -> int:
    es_url = os.getenv("EP_ELASTICSEARCH_URL") or os.getenv("ELASTICSEARCH_URL")
    es_api_key = os.getenv("EP_ELASTICSEARCH_API_KEY") or os.getenv("ELASTICSEARCH_API_KEY")
    es_index = os.getenv("EP_ELASTICSEARCH_INDEX") or os.getenv("ELASTICSEARCH_INDEX_NAME") or "default-logs"
    if not (es_url and es_api_key):
        print("Elasticsearch env not set; set EP_ELASTICSEARCH_URL and EP_ELASTICSEARCH_API_KEY")
        return 2

    # Emit two logs under rollout context
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    init_external_logging_from_env()
    logger = logging.getLogger("ep.verify.elasticsearch")
    for i in range(2):
        logger.info(
            f"verify elasticsearch message {i}",
            extra={
                "rollout_id": rollout_id,
                "experiment_id": "verify-exp",
                "run_id": "verify-run",
                "status_code": 101 if i == 0 else 200,
                "status_message": "RUNNING" if i == 0 else "COMPLETED",
            },
        )

    # Poll ES index by rollout_id
    headers = {"Authorization": f"ApiKey {es_api_key}", "Content-Type": "application/json"}
    search_body = {
        "query": {"term": {"rollout_id": rollout_id}},
        "size": 50,
        "sort": [{"@timestamp": {"order": "desc"}}],
    }
    url = f"{es_url.rstrip('/')}/{es_index}/_search"
    for _ in range(20):
        try:
            r = requests.post(url, headers=headers, json=search_body, timeout=15)
            r.raise_for_status()
            data: Dict[str, Any] = r.json() or {}
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                docs = [h.get("_source", {}) for h in hits]
                print(json.dumps({"total": len(docs), "sample": docs[:3]}, indent=2))
                return 0
        except Exception as e:
            print(f"Elasticsearch fetch error: {e}")
        time.sleep(2)

    print("No Elasticsearch entries found for rollout_id after retries")
    return 1


def main() -> int:
    mode = os.getenv("MODE")
    if mode is None:
        # Default: Fireworks if key present, else Elasticsearch
        mode = "fireworks" if os.getenv("FIREWORKS_API_KEY") else "elasticsearch"
    rollout_id = os.getenv("EP_ROLLOUT_ID") or _now_rollout_id()

    if mode == "fireworks":
        return verify_fireworks(rollout_id)
    elif mode == "elasticsearch":
        return verify_elasticsearch(rollout_id)
    else:
        print(f"Unknown MODE: {mode} (expected 'fireworks' or 'elasticsearch')")
        return 2


if __name__ == "__main__":
    sys.exit(main())
