"""
Minimal CLI runner for exported benchmarks.

Usage:

  python -m eval_protocol.benchmarks.run aime25_low \
    --model fireworks_ai/accounts/fireworks/models/gpt-oss-120b \
    --print-summary \
    --out artifacts/aime25_low.json \
    --max-rows 50 \
    --reasoning-effort low
"""

from __future__ import annotations

import argparse
from typing import Any

from importlib import import_module
import pkgutil
import eval_protocol.benchmarks.suites as suites_pkg
from eval_protocol.benchmarks.registry import get_benchmark_runner, list_benchmarks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an exported eval-protocol benchmark")
    parser.add_argument("name", help=f"Benchmark name. Known: {', '.join(list_benchmarks()) or '(none)'}")
    parser.add_argument("--model", required=True, help="Model identifier (provider/model)")
    parser.add_argument("--print-summary", action="store_true", help="Print concise EP summary line")
    parser.add_argument("--out", help="Write JSON summary artifact to path or directory")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Sets extra_body.reasoning.effort via EP_INPUT_PARAMS_JSON",
    )
    parser.add_argument(
        "--max-rows",
        help="Limit rows: integer or 'all' for no limit (maps to EP_MAX_DATASET_ROWS)",
    )
    parser.add_argument("--num-runs", type=int, help="Override num_runs if provided")
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens for generation requests")
    parser.add_argument("--max-concurrency", type=int, help="Override max concurrent rollouts")
    # Allow overriding reasoning effort explicitly (low/medium/high). If omitted, suite default is used.
    # Already mapped by --reasoning-effort above.
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # Auto-import all suite modules so their @export_benchmark decorators register
    # Import all suite modules so their @export_benchmark decorators register
    import sys, traceback
    for modinfo in pkgutil.iter_modules(suites_pkg.__path__):
        mod_name = f"{suites_pkg.__name__}.{modinfo.name}"
        try:
            import_module(mod_name)
        except Exception as e:
            print(f"[bench] failed to import suite module: {mod_name}: {e}", file=sys.stderr)
            traceback.print_exc()
    # Fallback: if nothing registered yet and a known suite was requested, try explicit import
    if not list_benchmarks():
        known_map = {
            "aime25_low": "eval_protocol.benchmarks.suites.aime25",
        }
        forced = known_map.get(args.name)
        if forced:
            try:
                import_module(forced)
            except Exception as e:
                print(f"[bench] explicit import failed for {forced}: {e}", file=sys.stderr)
    runner = get_benchmark_runner(args.name)
    max_rows: int | str | None = None
    if args.max_rows is not None:
        try:
            max_rows = int(args.max_rows)
        except Exception:
            max_rows = str(args.max_rows)
    # Build input params override if needed
    ip_override = {}
    if args.max_tokens is not None:
        ip_override["max_tokens"] = int(args.max_tokens)

    _ = runner(
        model=args.model,
        print_summary=args.print_summary,
        out=args.out,
        reasoning_effort=args.reasoning_effort,
        max_rows=max_rows,
        num_runs=args.num_runs,
        input_params_override=(ip_override or None),
        max_concurrency=args.max_concurrency,
    )
    # Non-zero exit on failure gate is handled within the runner via assertions
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


