#!/usr/bin/env python3
"""Latency scaling experiment for VRBDecode verification.

Measures verification latency with statistical rigor across a grid of
(K, N) configurations:

  - K in {16, 32, 64}  (candidate-set size)
  - N in {16, 32, 64, 128, 256}  (sequence length / steps)

For each (K, N): 100 verification passes (5 warmup + 95 measured).

Metrics per (K, N):
  - mean, std, min, max, p50, p95, p99 of verification latency
  - evidence artifact size (bytes)
  - verification throughput (transcripts/sec)
  - per-step latency (total / N)

Output:
  - ``eval/latency_scaling_results.json``
  - Printed formatted table

Usage:
  python3 eval/run_latency_scaling.py           # full run
  python3 eval/run_latency_scaling.py --quick   # reduced grid
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REF_PY = os.path.join(ROOT, "ref", "python")
sys.path.insert(0, REF_PY)

from forensic_verifier import verify_transcript
from receipt import (
    PolicyParams,
    generate_honest_transcript,
    serialize_transcript,
)


# ---------------------------------------------------------------------------
# Candidate generation (Zipf-like)
# ---------------------------------------------------------------------------

def _generate_candidates(
    rng: random.Random, K: int, n_steps: int,
) -> List[Tuple[List[int], List[int]]]:
    """Generate Zipf-like candidate sets."""
    sets: list = []
    for _ in range(n_steps):
        tids = rng.sample(range(1, 10_000_000), K)
        ranks = list(range(1, K + 1))
        rng.shuffle(ranks)
        logits = [
            int((2.0 - 5.0 * (r - 1) / max(K - 1, 1)) * (1 << 16))
            for r in ranks
        ]
        sets.append((tids, logits))
    return sets


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


# ---------------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------------

def run_latency_scaling(
    K_values: List[int],
    N_values: List[int],
    n_runs: int = 100,
    n_warmup: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run the latency scaling experiment.

    For each (K, N) configuration:
      1. Generate a single honest transcript.
      2. Verify it (n_warmup + n_runs) times.
      3. Discard the warmup runs.
      4. Compute statistics on the remaining n_runs latencies.
    """
    rng = random.Random(seed)
    all_results: List[Dict[str, Any]] = []
    n_measured = n_runs - n_warmup

    total_configs = len(K_values) * len(N_values)
    config_i = 0

    for K in K_values:
        for N in N_values:
            config_i += 1
            policy = PolicyParams(
                K=K,
                top_k=max(1, K // 4),
                top_p_q16=int(0.9 * (1 << 16)),
                T_q16=1 << 16,
                max_tokens=N,
            )
            tr_seed = rng.randbytes(32)
            request_id = rng.randbytes(32)
            candidates = _generate_candidates(rng, K, N)

            transcript = generate_honest_transcript(
                policy, tr_seed, request_id, candidates,
            )

            # Evidence artifact size
            evidence_bytes = len(serialize_transcript(transcript))

            # Run verification n_runs times (first n_warmup are discarded)
            latencies: List[float] = []
            for run_i in range(n_runs):
                t0 = time.perf_counter()
                results = verify_transcript(transcript, policy, tr_seed)
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)

            # Discard warmup
            measured = latencies[n_warmup:]

            # Compute statistics
            mean_lat = sum(measured) / len(measured)
            std_lat = math.sqrt(
                sum((x - mean_lat) ** 2 for x in measured) / len(measured)
            )
            min_lat = min(measured)
            max_lat = max(measured)
            p50 = _percentile(measured, 50)
            p95 = _percentile(measured, 95)
            p99 = _percentile(measured, 99)

            total_time = sum(measured)
            throughput = len(measured) / total_time if total_time > 0 else 0
            per_step_lat = mean_lat / N

            row = {
                "K": K,
                "N": N,
                "n_runs": n_measured,
                "n_warmup": n_warmup,
                "mean_ms": round(mean_lat * 1000, 4),
                "std_ms": round(std_lat * 1000, 4),
                "min_ms": round(min_lat * 1000, 4),
                "max_ms": round(max_lat * 1000, 4),
                "p50_ms": round(p50 * 1000, 4),
                "p95_ms": round(p95 * 1000, 4),
                "p99_ms": round(p99 * 1000, 4),
                "evidence_bytes": evidence_bytes,
                "throughput_transcripts_per_s": round(throughput, 2),
                "per_step_latency_ms": round(per_step_lat * 1000, 4),
            }
            all_results.append(row)

            print(
                f"  [{config_i}/{total_configs}] K={K:>3} N={N:>3}  "
                f"mean={row['mean_ms']:>8.3f}ms  "
                f"p50={row['p50_ms']:>8.3f}ms  "
                f"p95={row['p95_ms']:>8.3f}ms  "
                f"evidence={evidence_bytes:>7}B  "
                f"tput={row['throughput_transcripts_per_s']:>7.1f} t/s",
                flush=True,
            )

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Latency scaling experiment")
    ap.add_argument("--quick", action="store_true", help="Reduced grid")
    args = ap.parse_args()

    if args.quick:
        K_values = [16, 32]
        N_values = [16, 32, 64]
        n_runs = 30
        n_warmup = 5
    else:
        K_values = [16, 32, 64]
        N_values = [16, 32, 64, 128, 256]
        n_runs = 100
        n_warmup = 5

    print("=" * 78)
    print("LATENCY SCALING EXPERIMENT")
    print("=" * 78)
    print(f"  K values:  {K_values}")
    print(f"  N values:  {N_values}")
    print(f"  Runs/cfg:  {n_runs} ({n_warmup} warmup + {n_runs - n_warmup} measured)")
    print()

    t0 = time.perf_counter()
    results = run_latency_scaling(
        K_values, N_values,
        n_runs=n_runs, n_warmup=n_warmup,
    )
    total_elapsed = time.perf_counter() - t0

    # --- Output JSON ---
    output = {
        "config": {
            "K_values": K_values,
            "N_values": N_values,
            "n_runs": n_runs,
            "n_warmup": n_warmup,
            "n_measured": n_runs - n_warmup,
        },
        "results": results,
        "total_elapsed_s": round(total_elapsed, 3),
    }

    out_path = os.path.join(SCRIPT_DIR, "latency_scaling_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # --- Print formatted table ---
    print(f"\n{'='*78}")
    print("LATENCY SCALING RESULTS")
    print(f"{'='*78}")
    print(
        f"  {'K':>3} {'N':>4} "
        f"{'mean(ms)':>9} {'std(ms)':>8} "
        f"{'p50(ms)':>8} {'p95(ms)':>8} {'p99(ms)':>8} "
        f"{'min(ms)':>8} {'max(ms)':>8} "
        f"{'evid(B)':>8} {'tput(t/s)':>9} {'per-step':>9}"
    )
    print(f"  {'-'*103}")
    for r in results:
        print(
            f"  {r['K']:>3} {r['N']:>4} "
            f"{r['mean_ms']:>9.3f} {r['std_ms']:>8.3f} "
            f"{r['p50_ms']:>8.3f} {r['p95_ms']:>8.3f} {r['p99_ms']:>8.3f} "
            f"{r['min_ms']:>8.3f} {r['max_ms']:>8.3f} "
            f"{r['evidence_bytes']:>8} {r['throughput_transcripts_per_s']:>9.1f} "
            f"{r['per_step_latency_ms']:>8.3f}ms"
        )

    # --- Scaling analysis ---
    print(f"\n{'='*78}")
    print("SCALING ANALYSIS")
    print(f"{'='*78}")

    # Group by K to show N-scaling
    by_K: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_K.setdefault(r["K"], []).append(r)

    for K in sorted(by_K.keys()):
        rows = sorted(by_K[K], key=lambda r: r["N"])
        if len(rows) >= 2:
            base = rows[0]
            print(f"\n  K={K}: latency vs N (relative to N={base['N']})")
            for r in rows:
                ratio = r["mean_ms"] / base["mean_ms"] if base["mean_ms"] > 0 else 0
                n_ratio = r["N"] / base["N"]
                print(
                    f"    N={r['N']:>3}  mean={r['mean_ms']:>8.3f}ms  "
                    f"ratio={ratio:>5.2f}x  (N ratio={n_ratio:.1f}x)"
                )

    # Group by N to show K-scaling
    by_N: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_N.setdefault(r["N"], []).append(r)

    for N in sorted(by_N.keys()):
        rows = sorted(by_N[N], key=lambda r: r["K"])
        if len(rows) >= 2:
            base = rows[0]
            print(f"\n  N={N}: latency vs K (relative to K={base['K']})")
            for r in rows:
                ratio = r["mean_ms"] / base["mean_ms"] if base["mean_ms"] > 0 else 0
                k_ratio = r["K"] / base["K"]
                print(
                    f"    K={r['K']:>3}  mean={r['mean_ms']:>8.3f}ms  "
                    f"ratio={ratio:>5.2f}x  (K ratio={k_ratio:.1f}x)"
                )

    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Results written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
