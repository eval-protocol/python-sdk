#!/usr/bin/env python3
"""
Simple vLLM + OpenEnv Training Script

Tiny GRPO example that wires together:
- TRL's vLLM server (``trl vllm-serve``) for inference
- OpenEnv's BrowserGym client for MiniWoB tasks
- A custom OpenEnv + vLLM rollout function.

Prerequisites (example):
1. Start TRL's vLLM server with an INSTRUCT model on a separate GPU:
   CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-7B-Instruct --port 8000

2. Serve MiniWoB HTML:
   cd ~/miniwob-plusplus/miniwob/html && python -m http.server 8888

3. Run this script on a different GPU from vLLM:
   CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python simple_vllm_train.py
"""

import sys
import os

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add paths
sys.path.insert(0, os.path.expanduser("~/python-sdk"))
sys.path.insert(0, os.path.expanduser("~/OpenEnv/src"))

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import re
from typing import Any, List, Tuple, Optional

from eval_protocol.pytest.integrations.openenv_trl_vllm import create_openenv_vllm_rollout_func
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Optional: LoRA configuration
USE_LORA = True  # Set to False to train full model

if USE_LORA:
    from peft import LoraConfig

# Action pattern for parsing
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)


# ============================================================================
# Configuration
# ============================================================================

MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Use instruct model for better instruction following!
VLLM_URL = "http://localhost:8000"  # TRL vLLM server (no /v1 prefix)
MINIWOB_URL = os.getenv("MINIWOB_URL", "http://172.17.0.1:8888/miniwob/")

TASKS = [
    "click-test",
    "click-button",
    "enter-text",
]

OUTPUT_DIR = "outputs/simple-vllm"


# ============================================================================
# Helper Functions (from browsergym_grpo_evalp.py)
# ============================================================================


def _as_scalar(x: Any) -> Any:
    """Convert tensor/array to scalar if possible."""
    try:
        return x.item()
    except Exception:
        return x


def _build_history_lines(history: List[str]) -> str:
    """Format history lines (last 4 steps)."""
    if not history:
        return "None"
    return "\n".join(history[-4:])


def _extract_goal_url_title(observation: Any) -> Tuple[str, str, str]:
    """Extract (goal, url, title) from observation."""
    goal = getattr(observation, "goal", "") or ""
    url = getattr(observation, "url", "") or ""
    title = ""

    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}

    if not goal:
        goal = obs_dict.get("goal") or ""
    if not goal:
        goal_object = obs_dict.get("goal_object")
        if isinstance(goal_object, (list, tuple)) and goal_object:
            for item in goal_object:
                if isinstance(item, dict) and item.get("type") == "text":
                    goal = str(item.get("text", "")).strip()
                    if goal:
                        break
    if not goal:
        chat = obs_dict.get("chat_messages")
        if isinstance(chat, (list, tuple)) and chat:
            for msg in reversed(chat):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    goal = str(msg.get("message", "")).strip()
                    if goal:
                        break

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


def _elapsed_time_str(obs_dict: dict) -> str:
    """Format elapsed time from observation."""
    et = obs_dict.get("elapsed_time")
    try:
        et = et.item() if hasattr(et, "item") else float(et)
        return f"{et:.2f}s"
    except Exception:
        return "-"


def _extract_clickable_elements(observation) -> List[str]:
    """Extract clickable BIDs with details."""
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    extra_props = obs_dict.get("extra_element_properties", {}) or {}
    axtree_object = obs_dict.get("axtree_object") or {}
    focused_bid = obs_dict.get("focused_element_bid")

    # Build BID -> (role, name) mapping
    bid_to_desc = {}
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


def _rank_clickables_by_goal(observation: Any, goal: str, top_n: int = 8) -> Tuple[List[str], Optional[str]]:
    """Rank clickable BIDs by relevance to goal."""
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    goal_lc = (goal or "").lower().strip()

    extra_props = obs_dict.get("extra_element_properties", {}) or {}
    axtree_object = obs_dict.get("axtree_object") or {}
    focused_bid = str(obs_dict.get("focused_element_bid") or "")

    bid_to_desc = {}
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

    scored: List[Tuple[float, str, str, str, str]] = []
    for bid_key in sorted(extra_props.keys(), key=lambda x: str(x)):
        props = extra_props[bid_key] or {}
        if not props.get("clickable"):
            continue
        role, name = bid_to_desc.get(str(bid_key), ("", ""))
        name_lc = (name or "").lower()

        # Scoring: substring match + role bonus + focused bonus + visibility
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


def build_prompt(obs, step, history):
    """Build detailed prompt from observation (from browsergym_grpo_evalp.py)."""
    goal, url, title = _extract_goal_url_title(obs)
    url = url or "(unknown)"
    error_note = "Yes" if getattr(obs, "last_action_error", False) else "No"

    # Clickable BIDs
    clickables = _extract_clickable_elements(obs)
    clickable_block = "\n".join(clickables) if clickables else "(none detected)"
    ranked_clickables, recommended_bid = _rank_clickables_by_goal(obs, goal, top_n=10)
    ranked_block = "\n".join(ranked_clickables) if ranked_clickables else "(none)"

    # Build textual prompt
    text = getattr(obs, "text", "") or ""
    text = text[:3000]  # Limit size

    metadata = getattr(obs, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    axtree_text = (
        getattr(obs, "axtree_txt", None)
        or getattr(obs, "ax_tree_txt", None)
        or obs_dict.get("axtree_txt")
        or obs_dict.get("ax_tree_txt")
        or ""
    )
    pruned_html = getattr(obs, "pruned_html", None) or obs_dict.get("pruned_html") or ""
    axtree_text = str(axtree_text)[:2000]
    pruned_html = str(pruned_html)[:2000]

    focused_bid = obs_dict.get("focused_element_bid") or ""
    elapsed_str = _elapsed_time_str(obs_dict)
    last_action = obs_dict.get("last_action") or ""

    user_prompt = (
        f"Step: {step}\n"
        f"Goal: {goal}\n"
        f"Current URL: {url}\n"
        f"Title: {title}\n"
        f"Elapsed: {elapsed_str}\n"
        f"Previous steps:\n{_build_history_lines(history)}\n"
        f"Last action: {last_action}\n"
        f"Last action error: {error_note}\n"
        f"Focused BID: {focused_bid}\n\n"
        f"Clickable elements (BID: role | name | bbox | visibility):\n{clickable_block}\n\n"
        f"Ranked clickable candidates (best first):\n{ranked_block}\n"
        f"Recommended BID: {recommended_bid or '(none)'}\n\n"
        "Instructions:\n"
        "- Choose the most relevant clickable BID to achieve the goal.\n"
        "- Prefer role=button or elements whose name matches the goal.\n"
        "- Reply with a single action, e.g., click('13') or noop().\n\n"
        f"Page excerpt:\n{text}\n\n"
        f"AXTree excerpt:\n{axtree_text}\n\n"
        f"Pruned HTML excerpt:\n{pruned_html}\n\n"
        "Reply with exactly one BrowserGym action string."
    ).strip()

    return user_prompt


def parse_action(response_text: str) -> BrowserGymAction:
    """Parse BrowserGym action from LLM response (from browsergym_grpo_evalp.py)."""
    if not response_text:
        return BrowserGymAction(action_str="noop()")

    # Prefer first line that matches the action pattern
    for raw in response_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = ACTION_PATTERN.search(line)
        if m:
            return BrowserGymAction(action_str=re.sub(r"\s+", " ", m.group(0)))

    # Fallback: search whole response
    m = ACTION_PATTERN.search(response_text)
    if m:
        return BrowserGymAction(action_str=re.sub(r"\s+", " ", m.group(0)))

    return BrowserGymAction(action_str="noop()")


# ============================================================================
# Reward Function
# ============================================================================


def reward_func(completions, **kwargs):
    """
    Reward function (uses environment rewards).

    Returns total reward per episode.
    step_rewards is now a 1D list: [1.0, 1.0, 0.5, ...] (one per episode)
    """
    step_rewards = kwargs.get("step_rewards", [])
    if step_rewards:
        # Already summed per episode, return as-is
        return [float(r) for r in step_rewards]

    # Fallback
    return [0.0] * len(completions)


# ============================================================================
# Main Training
# ============================================================================


def main():
    print("Simple vLLM + OpenEnv training")
    print(f"  Model: {MODEL}")
    print(f"  vLLM server: {VLLM_URL}")
    print(f"  Tasks: {TASKS}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # NOTE: We let OpenEnvRolloutProcessor construct environments using env_client_cls
    # and handle task rotation via `tasks` and `num_generations`.
    rollout_func = create_openenv_vllm_rollout_func(
        env_factory=None,  # Use internal env_factory with task rotation
        env_client_cls=BrowserGymEnv,  # Generic HTTPEnvClient class
        prompt_builder=build_prompt,
        action_parser=parse_action,
        vllm_base_url=VLLM_URL,
        vllm_model=MODEL,  # Model name on vLLM server
        # Task rotation parameters (BrowserGym-style)
        tasks=TASKS,  # Rotate through these MiniWoB tasks
        miniwob_url=MINIWOB_URL,
        docker_image="browsergym-env:latest",
        benchmark="miniwob",
        headless=True,
        viewport_width=1280,
        viewport_height=720,
        timeout_ms=10000,
        # Rollout / vLLM sampling parameters
        max_steps=6,
        completion_params={
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        concurrency=2,
    )

    # Create dataset
    dataset = Dataset.from_dict(
        {
            "prompt": ["Start task"] * 6,
        }
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Training config
    # Set use_vllm=True and vllm_mode="server" to trigger rollout_func!
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_generations=2,  # Must divide evenly into batch_size
        max_steps=3,
        learning_rate=5e-6,
        temperature=0.7,
        max_completion_length=100,
        logging_steps=1,
        save_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        # vLLM server configuration - REQUIRED for rollout_func to be called!
        use_vllm=True,
        vllm_mode="server",  # Use separate vLLM server
        vllm_server_base_url=VLLM_URL,  # Point to vLLM server (correct param name!)
    )

    print("\nTraining configuration")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Generations per prompt: {training_args.num_generations}")
    print(f"  Max GRPO steps: {training_args.max_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  use_vllm: {training_args.use_vllm}")
    print(f"  vllm_mode: {training_args.vllm_mode}")
    print(f"  vllm_server_base_url: {training_args.vllm_server_base_url}")

    # Optional: Configure LoRA
    peft_config = None
    if USE_LORA:
        peft_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=16,  # LoRA alpha
            target_modules="all-linear",  # Apply to all linear layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"  Using LoRA: rank={peft_config.r}, alpha={peft_config.lora_alpha}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = GRPOTrainer(
        model=MODEL,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        rollout_func=rollout_func,
        peft_config=peft_config,  # Pass LoRA config
    )

    # Train
    print("\nStarting training...\n")
    trainer.train()

    # Save
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print(f"Training complete. Model saved to: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
