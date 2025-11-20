from typing import Any, Dict, List
import os
import re

import pytest
from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor

# Skip these integration-heavy tests on CI runners by default
pytestmark = pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip OpenEnv integration tests on CI")


def openenv_dataset_to_rows(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Adapter: convert simple {"id": "...", "prompt": "..."} rows into EvaluationRows.
    """
    rows: List[EvaluationRow] = []
    for row in data:
        prompt = str(row.get("prompt", "start"))
        rows.append(EvaluationRow(messages=[Message(role="user", content=prompt)]))
    return rows


# ---- prompt_builder and action_parser modeled after browsergym_grpo_evalp.py ----

ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)


def _as_scalar(x: Any) -> Any:
    try:
        return x.item()
    except Exception:
        return x


def _extract_goal_url_title(observation: Any) -> tuple[str, str, str]:
    goal = getattr(observation, "goal", "") or ""
    url = getattr(observation, "url", "") or ""
    title = ""
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    if not goal:
        goal = obs_dict.get("goal") or ""
    if not url:
        url = obs_dict.get("url") or ""
    titles = obs_dict.get("open_pages_titles") or ()
    active_idx = _as_scalar(obs_dict.get("active_page_index"))
    try:
        active_idx = int(active_idx)
    except Exception:
        active_idx = 0
    if isinstance(titles, (list, tuple)) and 0 <= active_idx < len(titles):
        title = titles[active_idx] or ""
    return goal, url, title


def _extract_clickable_elements_lines(observation: Any) -> List[str]:
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    extra_props = obs_dict.get("extra_element_properties", {}) or {}
    axtree_object = obs_dict.get("axtree_object") or {}
    focused_bid = obs_dict.get("focused_element_bid")
    bid_to_desc: Dict[str, tuple[str, str]] = {}
    try:
        nodes = axtree_object.get("nodes") or []
        for node in nodes:
            bid = node.get("browsergym_id")
            if bid is None:
                continue
            role = ""
            name = ""
            rf = node.get("role") or {}
            if isinstance(rf, dict):
                role = str(rf.get("value", "")).strip()
            nf = node.get("name") or {}
            if isinstance(nf, dict):
                name = str(nf.get("value", "")).strip()
            bid_to_desc[str(bid)] = (role, name)
    except Exception:
        pass
    lines: List[str] = []
    for bid in sorted(extra_props.keys(), key=lambda x: str(x)):
        props = extra_props[bid] or {}
        if not props.get("clickable"):
            continue
        bbox = props.get("bbox") or []
        bbox_str = ", ".join(str(v) for v in bbox) if bbox else "?"
        role, name = bid_to_desc.get(str(bid), ("", ""))
        focus_tag = " [FOCUSED]" if (str(bid) == str(focused_bid)) else ""
        rn = role or "-"
        if name:
            rn = f"{rn} | {name}"
        vis = props.get("visibility")
        vis_str = f"{vis:.2f}" if isinstance(vis, (int, float)) else str(vis) if vis is not None else "?"
        lines.append(f"- BID {bid}{focus_tag}: {rn} | bbox({bbox_str}) | visibility={vis_str}")
    return lines


def _rank_clickables_lines(observation: Any, goal: str, top_n: int = 8) -> tuple[List[str], str | None]:
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    goal_lc = (goal or "").lower().strip()
    extra_props = obs_dict.get("extra_element_properties", {}) or {}
    axtree_object = obs_dict.get("axtree_object") or {}
    focused_bid = str(obs_dict.get("focused_element_bid") or "")
    bid_to_desc: Dict[str, tuple[str, str]] = {}
    try:
        nodes = axtree_object.get("nodes") or []
        for node in nodes:
            bid = node.get("browsergym_id")
            if bid is None:
                continue
            role = ""
            name = ""
            rf = node.get("role") or {}
            if isinstance(rf, dict):
                role = str(rf.get("value", "")).strip()
            nf = node.get("name") or {}
            if isinstance(nf, dict):
                name = str(nf.get("value", "")).strip()
            bid_to_desc[str(bid)] = (role, name)
    except Exception:
        pass
    scored: List[tuple[float, str, str, str, str]] = []
    for bid_key in sorted(extra_props.keys(), key=lambda x: str(x)):
        props = extra_props[bid_key] or {}
        if not props.get("clickable"):
            continue
        role, name = bid_to_desc.get(str(bid_key), ("", ""))
        name_lc = (name or "").lower()
        score = 0.0
        if goal_lc and name_lc and (goal_lc in name_lc or name_lc in goal_lc):
            score += 2.0
        if (role or "").lower() == "button":
            score += 1.0
        if str(bid_key) == focused_bid:
            score += 0.5
        vis = props.get("visibility")
        try:
            vis_f = float(vis)
            score += max(0.0, min(1.0, vis_f))
        except Exception:
            pass
        bbox = props.get("bbox") or []
        bbox_str = ", ".join(str(v) for v in bbox) if bbox else "?"
        rn = role or "-"
        if name:
            rn = f"{rn} | {name}"
        vis_str = f"{vis:.2f}" if isinstance(vis, (int, float)) else str(vis) if vis is not None else "?"
        scored.append((score, str(bid_key), rn, bbox_str, vis_str))
    scored.sort(key=lambda t: t[0], reverse=True)
    lines: List[str] = []
    recommended = scored[0][1] if scored else None
    for idx, (score, bid, rn, bbox_str, vis_str) in enumerate(scored[:top_n], start=1):
        lines.append(f"{idx}. BID {bid}: score={score:.2f} | {rn} | bbox({bbox_str}) | visibility={vis_str}")
    return lines, recommended


def prompt_builder(observation: Any, step: int, history: List[str]) -> str:
    goal, url, title = _extract_goal_url_title(observation)
    url = url or "(unknown)"
    error_note = "Yes" if getattr(observation, "last_action_error", False) else "No"
    clickables_block = "\n".join(_extract_clickable_elements_lines(observation)) or "(none detected)"
    ranked_lines, rec = _rank_clickables_lines(observation, goal, top_n=10)
    ranked_block = "\n".join(ranked_lines) or "(none)"
    text = getattr(observation, "text", "") or ""
    text = text[:2048]
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    focused_bid = obs_dict.get("focused_element_bid") or ""
    last_action = obs_dict.get("last_action") or ""
    return (
        f"Step: {step}\n"
        f"Goal: {goal}\n"
        f"Current URL: {url}\n"
        f"Title: {title}\n"
        f"Previous steps:\n" + ("\n".join(history[-4:]) if history else "None") + "\n"
        f"Last action: {last_action}\n"
        f"Last action error: {error_note}\n"
        f"Focused BID: {focused_bid}\n\n"
        f"Clickable elements (BID: role | name | bbox | visibility):\n{clickables_block}\n\n"
        f"Ranked clickable candidates (best first):\n{ranked_block}\n"
        f"Recommended BID: {rec or '(none)'}\n\n"
        "Instructions:\n"
        "- Choose the most relevant clickable BID to achieve the goal.\n"
        "- Prefer role=button or elements whose name matches the goal.\n"
        "- Reply with a single action, e.g., click('13') or noop().\n\n"
        f"Page excerpt:\n{text}\n\n"
        "Reply with exactly one BrowserGym action string."
    ).strip()


def action_parser(response_text: str):
    try:
        from envs.browsergym_env import BrowserGymAction  # type: ignore
    except Exception:
        pytest.skip("OpenEnv (envs.browsergym_env) is not installed; skipping BrowserGym test.")
        raise
    if not response_text:
        return BrowserGymAction(action_str="noop()")
    for raw in response_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = ACTION_PATTERN.search(line)
        if m:
            parsed = re.sub(r"\s+", " ", m.group(0))
            return BrowserGymAction(action_str=parsed)
    m = ACTION_PATTERN.search(response_text)
    if m:
        parsed = re.sub(r"\s+", " ", m.group(0))
        return BrowserGymAction(action_str=parsed)
    return BrowserGymAction(action_str="noop()")


try:
    from envs.browsergym_env import BrowserGymEnv  # type: ignore

    _HAS_BG = True
except Exception:
    _HAS_BG = False


OPENENV_BROWSERGYM_INLINE_DATA: List[Dict[str, Any]] = [
    {"id": "click-test", "prompt": "start"},
]


@evaluation_test(  # type: ignore[misc]
    input_rows=[openenv_dataset_to_rows(OPENENV_BROWSERGYM_INLINE_DATA)],
    completion_params=[
        {
            "temperature": 0.0,
            "max_tokens": 512,
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct",
        }
    ],
    # Keep concurrency and steps low for a quick health-check
    num_runs=1,
    max_concurrent_rollouts=1,
    mode="pointwise",
    rollout_processor=(
        OpenEnvRolloutProcessor(
            env_client_cls=BrowserGymEnv if _HAS_BG else None,
            prompt_builder=prompt_builder,
            action_parser=action_parser,
            tasks=["click-test", "click-button"],
            task_var="BROWSERGYM_TASK_NAME",
            miniwob_url=os.getenv("MINIWOB_URL", "http://host.docker.internal:8888/miniwob/"),
            docker_image="browsergym-env:latest",
            benchmark="miniwob",
            timeout_ms=10000,
            num_generations=1,
            env_vars={
                "BROWSERGYM_BENCHMARK": "miniwob",
                "BROWSERGYM_HEADLESS": "true",
                "BROWSERGYM_VIEWPORT_WIDTH": "1280",
                "BROWSERGYM_VIEWPORT_HEIGHT": "720",
                "BROWSERGYM_TIMEOUT": "10000",
                "BROWSERGYM_OBS_AXTREE": "1",
                "BROWSERGYM_OBS_PRUNED_HTML": "1",
                "BROWSERGYM_RETURN_INFO": "1",
                "MINIWOB_URL": os.getenv("MINIWOB_URL", "http://host.docker.internal:8888/miniwob/"),
            },
        )
        if _HAS_BG
        else None
    ),
)
def test_openenv_browsergym_eval(row: EvaluationRow) -> EvaluationRow:
    """
    Smoke test to ensure OpenEnv + BrowserGym MiniWoB runs and returns a row.
    The evaluation harness will assert basic invariants (no exceptions, etc.).
    """
    if not _HAS_BG:
        pytest.skip("OpenEnv (envs.browsergym_env) is not installed; skipping BrowserGym test.")
    # Extract step rewards from execution metadata (set by OpenEnvRolloutProcessor)
    step_rewards: List[float] = []
    try:
        extra = getattr(row.execution_metadata, "extra", None)
        if isinstance(extra, dict):
            raw = extra.get("step_rewards") or []
            step_rewards = [float(r) for r in raw]
    except Exception:
        step_rewards = []

    total = float(sum(step_rewards)) if step_rewards else 0.0
    # Map total reward to a score in [0,1]; MiniWoB rewards are typically 0/1 or -1/1
    score = max(0.0, min(1.0, total))
    reason = f"Total reward={total:.2f} across {len(step_rewards)} steps"
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
