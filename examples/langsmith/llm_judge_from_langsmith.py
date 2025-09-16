"""Run a quick LLM-as-judge evaluation using LangSmith datasets and evaluators.

This mirrors our Langfuse example: we define a tiny dataset, a trivial target,
and run a rubric-based LLM judge via LangSmith's evaluation API.

Requirements:
  pip install -U langsmith langchain-openai

Env Vars:
  export LANGSMITH_API_KEY=...         # required
  export OPENAI_API_KEY=...            # optional; if absent uses heuristic judge
  export LANGSMITH_TRACING=true        # optional, to record runs

Run:
  python python-sdk/examples/langsmith/llm_judge_from_langsmith.py
"""

from __future__ import annotations

import os
from typing import Any, Dict
import importlib


def _ensure_env() -> None:
    os.environ.setdefault("LANGCHAIN_PROJECT", "ep-langgraph-examples")
    # Enable tracing so target runs + evaluator runs are visible in the UI
    os.environ.setdefault("LANGSMITH_TRACING", "true")


def main() -> None:
    _ensure_env()

    if not os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        raise SystemExit("Please set LANGSMITH_API_KEY (or LANGCHAIN_API_KEY).")
    use_openai = bool(os.getenv("OPENAI_API_KEY"))

    # Import here to allow the script to print clearer errors if deps are missing.
    try:
        ls = importlib.import_module("langsmith")
        eval_mod = importlib.import_module("langsmith.evaluation")
    except ImportError as e:
        raise SystemExit("Missing dependency. Please `pip install -U langsmith`. ") from e

    Client = getattr(ls, "Client")
    evaluate = getattr(eval_mod, "evaluate")

    client = Client()

    dataset_name = "ep_langsmith_demo_ds"
    # Create or get dataset
    try:
        dataset = client.create_dataset(dataset_name, description="EP demo dataset for LLM-as-judge")
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)

    # Seed examples (idempotent-ish: try to insert; duplicates are okay for demo)
    examples = [
        ({"prompt": "Say hello to Bob."}, {"answer": "Hello Bob!"}),
        ({"prompt": "What is 2+2?"}, {"answer": "4"}),
        (
            {"prompt": "Respond with a haiku about spring."},
            {"answer": "Gentle rains arrive\nBuds whisper to warming winds\nEarth breathes life anew"},
        ),
    ]
    for inputs, outputs in examples:
        try:
            client.create_example(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
        except Exception:
            # Ignore duplicate errors in throwaway demo
            pass

    # Define the target function: pretend model that returns uppercase
    def target_func(example_inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = example_inputs.get("prompt", "")
        return {"answer": str(text).upper()}

    # Define an evaluator that either uses OpenAI (if available) or a heuristic fallback
    import json
    import re
    from typing import cast

    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def heuristic_score(pred: str, ref: str) -> float:
        if not ref:
            return 0.0 if not pred else 0.5
        p = _normalize_text(pred)
        r = _normalize_text(ref)
        if p == r:
            return 1.0
        if r in p:
            return 0.8
        return 0.0

    def llm_as_judge(run, example):  # type: ignore[no-untyped-def]
        # Extract strings
        pred = ""
        try:
            out = run.outputs or {}
            pred = cast(str, out.get("answer") or out.get("output") or "")
        except Exception:
            pred = ""
        ref = ""
        try:
            ex_out = example.outputs or {}
            ref = cast(str, ex_out.get("answer") or ex_out.get("output") or "")
        except Exception:
            ref = ""

        if not use_openai:
            score = heuristic_score(pred, ref)
            return {"key": "llm_judge", "score": float(score), "comment": "heuristic"}

        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception:
            score = heuristic_score(pred, ref)
            return {"key": "llm_judge", "score": float(score), "comment": "heuristic (no openai)"}

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        system = (
            "You are an impartial grader. Compare the candidate answer to the reference answer. "
            "Return a JSON object with fields 'score' (float 0.0-1.0) and 'reason' (short string). "
            "Award 1.0 for semantic equivalence, 0.8 for close paraphrase, else 0.0."
        )
        user = json.dumps({"reference": ref, "candidate": pred})
        try:
            resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            content = getattr(resp, "content", "")
            data = {}
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    # langchain message content may be a list of dicts
                    data = json.loads(content[0].get("text", "{}"))  # type: ignore[index]
            except Exception:
                data = {"score": heuristic_score(pred, ref), "reason": "fallback parse"}
            score = float(max(0.0, min(1.0, float(data.get("score", 0.0)))))
            reason = str(data.get("reason", ""))[:500]
            return {"key": "llm_judge", "score": score, "comment": reason}
        except Exception as e:
            score = heuristic_score(pred, ref)
            return {"key": "llm_judge", "score": float(score), "comment": f"heuristic (error: {e})"}

    print("Running evaluation... this will create an experiment in LangSmith.")
    results = evaluate(
        target_func,
        data=dataset_name,
        evaluators=[llm_as_judge],
        experiment_prefix="ep-llm-judge-demo",
        max_concurrency=4,
        metadata={"source": "examples/langsmith"},
    )

    print("Experiment URL:")
    try:
        print(results.get("url"))  # type: ignore[reportUnknownMemberType]
    except Exception:
        pass

    print("Done. Visit LangSmith to review scores and details.")


if __name__ == "__main__":
    main()
