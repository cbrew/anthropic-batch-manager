#!/usr/bin/env python3
"""Tuning harness for batch_threshold (k).

Sweeps across task counts and threshold values to measure the wall-clock and
token-cost tradeoffs between the individual-call path and the batch-API path.

Requires ANTHROPIC_API_KEY in the environment.

Usage:
    python scripts/tune_batch_threshold.py
    python scripts/tune_batch_threshold.py --task-counts 5 10 20 --thresholds 5 15 25
    python scripts/tune_batch_threshold.py --model claude-haiku-4-5-20251001  # cheaper for tire-kicking
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass

import anthropic

from batch_compiler import LLMTask, TaskGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Cheap, deterministic prompt that still exercises the full path.
PROMPT = "Reply with exactly one word: the color of the sky."


@dataclass
class TrialResult:
    task_count: int
    threshold: int
    path: str  # "individual" or "batch"
    wall_seconds: float
    succeeded: int
    errored: int
    total_input_tokens: int
    total_output_tokens: int


async def run_trial(
    n_tasks: int,
    threshold: int,
    model: str,
    client: anthropic.AsyncAnthropic,
) -> TrialResult:
    """Build a flat fan-out of n_tasks, run with the given threshold."""
    graph = TaskGraph()
    for i in range(n_tasks):
        graph.add(LLMTask(
            id=f"t{i}",
            model=model,
            max_tokens=16,
            prompt=PROMPT,
        ))

    t0 = time.monotonic()
    results = await graph.run(client=client, batch_threshold=threshold)
    elapsed = time.monotonic() - t0

    succeeded = sum(1 for r in results.values() if r.status == "succeeded")
    errored = sum(1 for r in results.values() if r.status == "errored")
    input_tok = sum(
        (r.usage or {}).get("input_tokens", 0) for r in results.values()
    )
    output_tok = sum(
        (r.usage or {}).get("output_tokens", 0) for r in results.values()
    )

    path = "individual" if n_tasks < threshold else "batch"
    return TrialResult(
        task_count=n_tasks,
        threshold=threshold,
        path=path,
        wall_seconds=elapsed,
        succeeded=succeeded,
        errored=errored,
        total_input_tokens=input_tok,
        total_output_tokens=output_tok,
    )


def print_table(rows: list[TrialResult]) -> None:
    header = (
        f"{'tasks':>5}  {'k':>3}  {'path':>10}  {'wall(s)':>8}  "
        f"{'ok':>4}  {'err':>4}  {'in_tok':>8}  {'out_tok':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.task_count:>5}  {r.threshold:>3}  {r.path:>10}  "
            f"{r.wall_seconds:>8.2f}  {r.succeeded:>4}  {r.errored:>4}  "
            f"{r.total_input_tokens:>8}  {r.total_output_tokens:>8}"
        )
    print()


def print_comparison(rows: list[TrialResult]) -> None:
    """For each task count that was tested on both paths, show the delta."""
    by_count: dict[int, dict[str, TrialResult]] = {}
    for r in rows:
        by_count.setdefault(r.task_count, {})[r.path] = r

    pairs = {n: d for n, d in by_count.items() if len(d) == 2}
    if not pairs:
        return

    print("=== Crossover analysis ===")
    print(f"{'tasks':>5}  {'indiv(s)':>9}  {'batch(s)':>9}  {'delta(s)':>9}  {'winner':>10}")
    print("-" * 50)
    for n in sorted(pairs):
        ind = pairs[n]["individual"].wall_seconds
        bat = pairs[n]["batch"].wall_seconds
        delta = bat - ind
        winner = "individual" if delta > 0 else "batch"
        print(f"{n:>5}  {ind:>9.2f}  {bat:>9.2f}  {delta:>+9.2f}  {winner:>10}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Tune batch_threshold (k)")
    parser.add_argument(
        "--task-counts",
        type=int,
        nargs="+",
        default=[3, 5, 8, 10, 15, 20, 50],
        help="Number of tasks per trial (default: 3 5 8 10 15 20 50)",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=None,
        help="Threshold values to test. Default: for each task count, "
        "force both paths by using k=task_count (batch) and k=task_count+1 (individual).",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model to use (default: claude-haiku-4-5-20251001 — cheapest)",
    )
    args = parser.parse_args()

    client = anthropic.AsyncAnthropic()
    rows: list[TrialResult] = []

    if args.thresholds:
        # Explicit grid: every (count, threshold) pair
        trials = [
            (n, k) for n in args.task_counts for k in args.thresholds
        ]
    else:
        # Default: for each count, force both the individual and batch paths
        trials = []
        for n in args.task_counts:
            trials.append((n, n + 1))  # individual path (n < k)
            trials.append((n, n))      # batch path     (n >= k)

    for n, k in trials:
        path_label = "individual" if n < k else "batch"
        logger.info("Trial: %d tasks, k=%d → %s path", n, k, path_label)
        try:
            row = await run_trial(n, k, args.model, client)
            rows.append(row)
        except Exception:
            logger.exception("Trial failed: %d tasks, k=%d", n, k)

    print_table(rows)
    print_comparison(rows)


if __name__ == "__main__":
    asyncio.run(main())
