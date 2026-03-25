#!/usr/bin/env python3
"""ICUFN evaluation runner.

Reproduces the experiments described in the ICUFN paper:
  Section 5.1 -- Security-facing metrics (detection rate, FP/FN,
                 attribution accuracy per attack class)
  Section 5.2 -- Operational metrics (verification latency, storage
                 overhead, audit throughput)

Usage
-----
Quick sanity run (small K, few transcripts):
    python3 eval/run_icufn.py --quick

Full paper-grade run:
    python3 eval/run_icufn.py

Output files (under eval/):
    icufn_detection.csv   -- per-attack detection/attribution results
    icufn_operational.csv -- per-config operational metrics
    icufn_results.json    -- consolidated results with metadata
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REF_PY = os.path.join(ROOT, "ref", "python")
sys.path.insert(0, REF_PY)

from attack_simulator import (
    attack_candidate_manipulation,
    attack_policy_mismatch,
    attack_randomness_replay,
    attack_transcript_drop,
    attack_transcript_reorder,
)
from forensic_verifier import VerifyCode, verify_transcript
from receipt import PolicyParams, Transcript, generate_honest_transcript, serialize_transcript

# Baselines
from baseline_merkle import MerkleBaseline, BaselineCode
from baseline_policy_commit import (
    PolicyCommitBaseline,
    PolicyCommitCode,
    evaluate_policy_commit_baseline,
)
from baseline_watermark import (
    WatermarkConfig,
    WatermarkCode,
    detect_watermark,
    generate_watermarked_transcript,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    K_values: List[int]
    n_steps_values: List[int]
    n_transcripts: int
    attack_fractions: List[float]
    seed: int = 1


QUICK_CONFIG = EvalConfig(
    K_values=[16],
    n_steps_values=[16, 32],
    n_transcripts=10,
    attack_fractions=[0.25, 0.5, 1.0],
)

FULL_CONFIG = EvalConfig(
    K_values=[16, 32, 64],
    n_steps_values=[32, 64, 128],
    n_transcripts=50,
    attack_fractions=[0.1, 0.25, 0.5, 1.0],
)


# ---------------------------------------------------------------------------
# Candidate-set generator
# ---------------------------------------------------------------------------

def _generate_candidates(
    rng: random.Random, K: int, n_steps: int,
    *, uniform_logits: bool = False,
) -> List[Tuple[List[int], List[int]]]:
    """Generate random candidate sets for *n_steps* decoding steps.

    By default, logits follow a Zipf-like (rank-based) distribution that
    mimics real LLM output: a few high-probability tokens and a long tail.
    Pass ``uniform_logits=True`` to use the legacy uniform-random logits.
    """
    sets = []
    for _ in range(n_steps):
        tids = rng.sample(range(1, 10_000_000), K)
        if uniform_logits:
            logits = [rng.randint(-(2 << 16), (2 << 16)) for _ in range(K)]
        else:
            # Rank-based logit assignment (Zipf-like)
            # Highest-ranked token gets logit ~2.0 (Q16), lowest gets ~-3.0 (Q16)
            ranks = list(range(1, K + 1))
            rng.shuffle(ranks)  # random assignment of ranks to token positions
            logits = [
                int((2.0 - 5.0 * (r - 1) / max(K - 1, 1)) * (1 << 16))
                for r in ranks
            ]
        sets.append((tids, logits))
    return sets


# ---------------------------------------------------------------------------
# Attack registry
# ---------------------------------------------------------------------------

ATTACK_CLASSES = {
    "policy_mismatch": {
        "code": VerifyCode.POLICY_MISMATCH,
        "description": "Alter temperature after commitment",
    },
    "randomness_replay": {
        "code": VerifyCode.RANDOMNESS_REPLAY,
        "description": "Reuse U_t from step 0 in later steps",
    },
    "candidate_manipulation": {
        "code": VerifyCode.CANDIDATE_MANIPULATION,
        "description": "Inject high-logit synthetic token",
    },
    "transcript_drop": {
        "code": VerifyCode.TRANSCRIPT_DISCONTINUITY,
        "description": "Drop random steps from transcript",
    },
    "transcript_reorder": {
        "code": VerifyCode.TRANSCRIPT_DISCONTINUITY,
        "description": "Swap adjacent step pairs",
    },
}


def _apply_attack(
    attack_name: str,
    transcript: Transcript,
    fraction: float,
    rng: random.Random,
) -> Transcript:
    """Apply the named attack at the given intensity."""
    if attack_name == "policy_mismatch":
        return attack_policy_mismatch(
            transcript, fraction,
            new_T_q16=transcript.policy.T_q16 // 2 or 1,
            rng=rng,
        )
    elif attack_name == "randomness_replay":
        return attack_randomness_replay(transcript, fraction, rng=rng)
    elif attack_name == "candidate_manipulation":
        return attack_candidate_manipulation(transcript, fraction, rng=rng)
    elif attack_name == "transcript_drop":
        return attack_transcript_drop(transcript, fraction, rng=rng)
    elif attack_name == "transcript_reorder":
        n_swaps = max(1, int(len(transcript.steps) * fraction))
        return attack_transcript_reorder(transcript, n_swaps=n_swaps, rng=rng)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    K: int
    n_steps: int
    attack: str
    fraction: float
    transcript_idx: int
    detected: bool
    attributed_correctly: bool
    n_findings: int
    # Baseline detection flags
    merkle_detected: bool = False
    pcommit_detected: bool = False
    watermark_detected: bool = False


@dataclass
class OperationalRecord:
    K: int
    n_steps: int
    transcript_idx: int
    verify_latency_s: float
    evidence_bytes: int
    n_steps_actual: int


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _run_detection_eval(
    cfg: EvalConfig,
    *,
    uniform_logits: bool = False,
) -> Tuple[List[DetectionRecord], List[OperationalRecord], float]:
    det_records: List[DetectionRecord] = []
    ops_records: List[OperationalRecord] = []
    rng = random.Random(cfg.seed)

    # -- FP tracking (Task A2) --
    n_honest_verified = 0
    n_false_positives = 0

    total_combos = (
        len(cfg.K_values)
        * len(cfg.n_steps_values)
        * cfg.n_transcripts
    )
    combo_i = 0

    for K in cfg.K_values:
        for n_steps in cfg.n_steps_values:
            policy = PolicyParams(
                K=K,
                top_k=max(1, K // 4),
                top_p_q16=int(0.9 * (1 << 16)),
                T_q16=1 << 16,  # T = 1.0
                max_tokens=n_steps,
            )
            seed = rng.randbytes(32)

            # Baseline instances (shared across transcripts in this config)
            merkle_bl = MerkleBaseline()
            pcommit_bl = PolicyCommitBaseline()
            wm_config = WatermarkConfig()

            for t_idx in range(cfg.n_transcripts):
                combo_i += 1
                request_id = rng.randbytes(32)
                candidates = _generate_candidates(
                    rng, K, n_steps, uniform_logits=uniform_logits,
                )

                transcript = generate_honest_transcript(
                    policy, seed, request_id, candidates,
                )

                # Also generate a watermarked version for watermark eval
                wm_transcript = generate_watermarked_transcript(
                    policy, seed, request_id, candidates, wm_config,
                )

                # -- Operational metrics on honest transcript --------
                t0 = time.perf_counter()
                honest_results = verify_transcript(
                    transcript, policy, seed,
                )
                elapsed = time.perf_counter() - t0

                # Full serialized evidence artifact size (Task A1)
                evidence_bytes = len(serialize_transcript(transcript))

                ops_records.append(OperationalRecord(
                    K=K,
                    n_steps=n_steps,
                    transcript_idx=t_idx,
                    verify_latency_s=elapsed,
                    evidence_bytes=evidence_bytes,
                    n_steps_actual=len(transcript.steps),
                ))

                # Track FP from honest transcript (Task A2)
                n_honest_verified += 1
                if any(r.code != VerifyCode.PASS for r in honest_results):
                    n_false_positives += 1

                # Honest transcript MUST pass
                assert all(
                    r.code == VerifyCode.PASS for r in honest_results
                ), f"Honest transcript failed: {honest_results}"

                # -- Attack evaluations ---------------------------------
                # Save ground-truth candidates for candidate manipulation
                gt_cands = [
                    (s.token_ids[:], s.logit_q16s[:])
                    for s in transcript.steps
                ]

                for attack_name, attack_info in ATTACK_CLASSES.items():
                    for frac in cfg.attack_fractions:
                        tampered = _apply_attack(
                            attack_name, transcript, frac, rng,
                        )

                        # --- Forensic verifier ---
                        gt = (
                            gt_cands
                            if attack_name == "candidate_manipulation"
                            else None
                        )
                        results = verify_transcript(
                            tampered, policy, seed,
                            ground_truth_candidates=gt,
                        )

                        detected = any(
                            r.code != VerifyCode.PASS for r in results
                        )
                        expected_code = attack_info["code"]
                        attributed = any(
                            r.code == expected_code for r in results
                        )

                        # --- Merkle baseline ---
                        merkle_signed = merkle_bl.sign_transcript(tampered)
                        merkle_results = merkle_bl.verify_transcript(merkle_signed)
                        merkle_det = not all(
                            r.code == BaselineCode.PASS for r in merkle_results
                        )

                        # --- Policy-commit baseline ---
                        pcommit_results = evaluate_policy_commit_baseline(tampered)
                        pcommit_det = not all(
                            r.code == PolicyCommitCode.PASS for r in pcommit_results
                        )

                        # --- Watermark baseline ---
                        # Apply same attack to the watermarked transcript
                        wm_tampered = _apply_attack(
                            attack_name, wm_transcript, frac, rng,
                        )
                        wm_result = detect_watermark(wm_tampered, wm_config)
                        # Detection: watermark is MISSING in tampered
                        wm_det = (wm_result.code == WatermarkCode.WATERMARK_MISSING)

                        det_records.append(DetectionRecord(
                            K=K,
                            n_steps=n_steps,
                            attack=attack_name,
                            fraction=frac,
                            transcript_idx=t_idx,
                            detected=detected,
                            attributed_correctly=attributed,
                            n_findings=len([
                                r for r in results
                                if r.code != VerifyCode.PASS
                            ]),
                            merkle_detected=merkle_det,
                            pcommit_detected=pcommit_det,
                            watermark_detected=wm_det,
                        ))

                if combo_i % 10 == 0 or combo_i == total_combos:
                    print(
                        f"  [{combo_i}/{total_combos}] "
                        f"K={K} N={n_steps} t={t_idx}",
                        flush=True,
                    )

    # -- Edge-case honest transcripts for FP measurement (Task A2) --
    print("\n  Running edge-case honest transcripts for FP measurement...")
    edge_rng = random.Random(cfg.seed + 7777)

    edge_cases = [
        # Near-boundary top_p (0.999 * Q16)
        {"label": "near_boundary_top_p",
         "policy": PolicyParams(K=16, top_k=4,
                                top_p_q16=int(0.999 * (1 << 16)),
                                T_q16=1 << 16, max_tokens=16)},
        # Extreme temperature: T_q16=1 (minimum possible)
        {"label": "extreme_low_temperature",
         "policy": PolicyParams(K=16, top_k=4,
                                top_p_q16=int(0.9 * (1 << 16)),
                                T_q16=1, max_tokens=16)},
        # All-ties logits (all candidates have the same logit)
        {"label": "all_ties_logits",
         "policy": PolicyParams(K=16, top_k=4,
                                top_p_q16=int(0.9 * (1 << 16)),
                                T_q16=1 << 16, max_tokens=16),
         "custom_logits": True},
        # Minimal K=1
        {"label": "minimal_K",
         "policy": PolicyParams(K=1, top_k=1,
                                top_p_q16=int(0.9 * (1 << 16)),
                                T_q16=1 << 16, max_tokens=16)},
    ]

    for ec in edge_cases:
        ec_policy = ec["policy"]
        ec_seed = edge_rng.randbytes(32)
        ec_reqid = edge_rng.randbytes(32)
        ec_K = ec_policy.K
        ec_n = ec_policy.max_tokens

        if ec.get("custom_logits"):
            # All-ties: every candidate has the same logit value
            ec_cands = []
            for _ in range(ec_n):
                tids = edge_rng.sample(range(1, 10_000_000), ec_K)
                logits = [1 << 16] * ec_K  # all equal: 1.0 in Q16
                ec_cands.append((tids, logits))
        else:
            ec_cands = _generate_candidates(
                edge_rng, ec_K, ec_n, uniform_logits=uniform_logits,
            )

        ec_transcript = generate_honest_transcript(
            ec_policy, ec_seed, ec_reqid, ec_cands,
        )
        ec_results = verify_transcript(ec_transcript, ec_policy, ec_seed)

        n_honest_verified += 1
        if any(r.code != VerifyCode.PASS for r in ec_results):
            n_false_positives += 1
            print(f"    WARNING: edge-case '{ec['label']}' triggered FP: {ec_results}")
        else:
            print(f"    edge-case '{ec['label']}': PASS")

    fp_rate = n_false_positives / n_honest_verified if n_honest_verified > 0 else 0.0
    print(f"  FP measurement: {n_false_positives}/{n_honest_verified} = {fp_rate:.4%}")

    return det_records, ops_records, fp_rate


# ---------------------------------------------------------------------------
# Dedicated FP measurement (10,000 honest transcripts)
# ---------------------------------------------------------------------------

def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds.  *z* = 1.96 gives a 95 % CI.
    """
    if n_total == 0:
        return (0.0, 1.0)
    p_hat = n_success / n_total
    denom = 1 + z * z / n_total
    centre = (p_hat + z * z / (2 * n_total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n_total)) / n_total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class FPResult:
    n_honest: int
    n_fp: int
    fp_rate: float
    wilson_lo: float
    wilson_hi: float


def _run_fp_measurement(
    *,
    n_total: int = 10_000,
    seed: int = 42_000,
    uniform_logits: bool = False,
    quick: bool = False,
) -> FPResult:
    """Generate *n_total* honest transcripts and measure false-positive rate.

    Transcripts are spread across K in {16, 32, 64}, N in {16, 32, 64},
    and four edge-case categories (near-boundary top_p, extreme temperature,
    all-ties logits, minimal K=1).

    In ``--quick`` mode only 200 transcripts are generated (enough for a
    smoke-test).
    """
    if quick:
        n_total = 200  # fast sanity check

    rng = random.Random(seed)

    # --- Configuration grid ---
    K_values = [16, 32, 64]
    N_values = [16, 32, 64]
    grid_configs: List[dict] = []
    for K in K_values:
        for N in N_values:
            grid_configs.append({
                "K": K,
                "N": N,
                "policy": PolicyParams(
                    K=K,
                    top_k=max(1, K // 4),
                    top_p_q16=int(0.9 * (1 << 16)),
                    T_q16=1 << 16,
                    max_tokens=N,
                ),
                "custom_logits": None,
            })

    # Edge-case configs (cycle through these as well)
    edge_configs: List[dict] = [
        # Near-boundary top_p (0.999)
        {"K": 16, "N": 32, "label": "near_boundary_top_p",
         "policy": PolicyParams(K=16, top_k=4,
                                top_p_q16=int(0.999 * (1 << 16)),
                                T_q16=1 << 16, max_tokens=32),
         "custom_logits": None},
        # Extreme low temperature
        {"K": 16, "N": 32, "label": "extreme_low_temperature",
         "policy": PolicyParams(K=16, top_k=4,
                                top_p_q16=int(0.9 * (1 << 16)),
                                T_q16=1, max_tokens=32),
         "custom_logits": None},
        # All-ties logits
        {"K": 16, "N": 32, "label": "all_ties_logits",
         "policy": PolicyParams(K=16, top_k=4,
                                top_p_q16=int(0.9 * (1 << 16)),
                                T_q16=1 << 16, max_tokens=32),
         "custom_logits": "all_ties"},
        # Minimal K=1
        {"K": 1, "N": 32, "label": "minimal_K",
         "policy": PolicyParams(K=1, top_k=1,
                                top_p_q16=int(0.9 * (1 << 16)),
                                T_q16=1 << 16, max_tokens=32),
         "custom_logits": None},
    ]

    # Build a round-robin schedule: grid (9 configs) + edge (4 configs) = 13
    all_configs = grid_configs + edge_configs

    n_fp = 0
    n_done = 0
    report_interval = max(1, n_total // 20)

    print(f"\n  FP measurement: generating {n_total} honest transcripts ...")

    for i in range(n_total):
        cfg = all_configs[i % len(all_configs)]
        policy = cfg["policy"]
        K = policy.K
        N = policy.max_tokens

        ec_seed = rng.randbytes(32)
        ec_reqid = rng.randbytes(32)

        if cfg.get("custom_logits") == "all_ties":
            cands = []
            for _ in range(N):
                tids = rng.sample(range(1, 10_000_000), K)
                logits = [1 << 16] * K
                cands.append((tids, logits))
        else:
            cands = _generate_candidates(rng, K, N, uniform_logits=uniform_logits)

        transcript = generate_honest_transcript(policy, ec_seed, ec_reqid, cands)
        results = verify_transcript(transcript, policy, ec_seed)

        n_done += 1
        if any(r.code != VerifyCode.PASS for r in results):
            n_fp += 1
            label = cfg.get("label", f"K={K},N={N}")
            print(f"    WARNING: FP at transcript {i} ({label}): {results}")

        if n_done % report_interval == 0 or n_done == n_total:
            print(f"    [{n_done}/{n_total}] FP so far: {n_fp}", flush=True)

    fp_rate = n_fp / n_done if n_done else 0.0
    lo, hi = _wilson_ci(n_fp, n_done)

    print(f"  FP result: {n_fp}/{n_done} = {fp_rate:.4%}")
    print(f"  95% Wilson CI: [{lo:.6f}, {hi:.6f}]")

    return FPResult(
        n_honest=n_done,
        n_fp=n_fp,
        fp_rate=fp_rate,
        wilson_lo=lo,
        wilson_hi=hi,
    )


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def _aggregate_detection(
    records: List[DetectionRecord],
) -> List[Dict[str, Any]]:
    """Aggregate detection records into per-(attack, fraction) summaries."""
    from collections import defaultdict

    buckets: Dict[tuple, list] = defaultdict(list)
    for r in records:
        buckets[(r.attack, r.fraction)].append(r)

    rows = []
    for (attack, frac), recs in sorted(buckets.items()):
        n = len(recs)
        n_detected = sum(1 for r in recs if r.detected)
        n_attributed = sum(1 for r in recs if r.attributed_correctly)
        n_merkle = sum(1 for r in recs if r.merkle_detected)
        n_pcommit = sum(1 for r in recs if r.pcommit_detected)
        n_watermark = sum(1 for r in recs if r.watermark_detected)
        rows.append({
            "attack": attack,
            "fraction": frac,
            "n_trials": n,
            "n_detected": n_detected,
            "detection_rate": n_detected / n if n else 0.0,
            "n_attributed": n_attributed,
            "attribution_rate": n_attributed / n if n else 0.0,
            "fn_rate": 1.0 - (n_detected / n) if n else 0.0,
            "merkle_detected": n_merkle,
            "merkle_rate": n_merkle / n if n else 0.0,
            "pcommit_detected": n_pcommit,
            "pcommit_rate": n_pcommit / n if n else 0.0,
            "watermark_detected": n_watermark,
            "watermark_rate": n_watermark / n if n else 0.0,
        })
    return rows


def _aggregate_operational(
    records: List[OperationalRecord],
) -> List[Dict[str, Any]]:
    """Aggregate operational records into per-(K, n_steps) summaries."""
    from collections import defaultdict

    buckets: Dict[tuple, list] = defaultdict(list)
    for r in records:
        buckets[(r.K, r.n_steps)].append(r)

    rows = []
    for (K, n_steps), recs in sorted(buckets.items()):
        latencies = [r.verify_latency_s for r in recs]
        sizes = [r.evidence_bytes for r in recs]
        rows.append({
            "K": K,
            "n_steps": n_steps,
            "n_transcripts": len(recs),
            "mean_verify_latency_s": sum(latencies) / len(latencies),
            "min_verify_latency_s": min(latencies),
            "max_verify_latency_s": max(latencies),
            "mean_evidence_bytes": sum(sizes) / len(sizes),
            "throughput_transcripts_per_s": (
                len(recs) / sum(latencies) if sum(latencies) > 0 else 0
            ),
        })
    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="ICUFN evaluation runner")
    ap.add_argument(
        "--quick", action="store_true",
        help="Run reduced configuration for quick sanity check",
    )
    ap.add_argument(
        "--uniform-logits", action="store_true",
        help="Use legacy uniform-random logits instead of Zipf-like distribution",
    )
    ap.add_argument(
        "--out-dir", default=os.path.join(SCRIPT_DIR),
        help="Output directory (default: eval/)",
    )
    args = ap.parse_args()

    cfg = QUICK_CONFIG if args.quick else FULL_CONFIG
    logit_mode = "uniform" if args.uniform_logits else "zipf-like"
    print(f"ICUFN evaluation: {'quick' if args.quick else 'full'} mode")
    print(f"  K values:     {cfg.K_values}")
    print(f"  N steps:      {cfg.n_steps_values}")
    print(f"  Transcripts:  {cfg.n_transcripts}")
    print(f"  Intensities:  {cfg.attack_fractions}")
    print(f"  Logit dist:   {logit_mode}")
    print()

    t0 = time.perf_counter()
    det_records, ops_records, fp_rate_inline = _run_detection_eval(
        cfg, uniform_logits=args.uniform_logits,
    )
    elapsed_det = time.perf_counter() - t0

    # --- Dedicated FP measurement (10,000 honest transcripts) ---
    t1 = time.perf_counter()
    fp_result = _run_fp_measurement(
        uniform_logits=args.uniform_logits,
        quick=args.quick,
    )
    elapsed_fp = time.perf_counter() - t1

    elapsed = elapsed_det + elapsed_fp

    # Aggregate
    det_summary = _aggregate_detection(det_records)
    ops_summary = _aggregate_operational(ops_records)

    # Write outputs
    out_dir = args.out_dir
    _write_csv(os.path.join(out_dir, "icufn_detection.csv"), det_summary)
    _write_csv(os.path.join(out_dir, "icufn_operational.csv"), ops_summary)

    consolidated = {
        "mode": "quick" if args.quick else "full",
        "logit_distribution": logit_mode,
        "config": {
            "K_values": cfg.K_values,
            "n_steps_values": cfg.n_steps_values,
            "n_transcripts": cfg.n_transcripts,
            "attack_fractions": cfg.attack_fractions,
            "seed": cfg.seed,
        },
        "elapsed_s": elapsed,
        "detection_summary": det_summary,
        "operational_summary": ops_summary,
        "false_positive_rate_inline": fp_rate_inline,
        "fp_measurement": {
            "n_honest": fp_result.n_honest,
            "n_fp": fp_result.n_fp,
            "fp_rate": fp_result.fp_rate,
            "wilson_ci_95_lo": fp_result.wilson_lo,
            "wilson_ci_95_hi": fp_result.wilson_hi,
        },
    }
    json_path = os.path.join(out_dir, "icufn_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    # Print summary
    print(f"\nCompleted in {elapsed:.1f}s "
          f"(detection: {elapsed_det:.1f}s, FP measurement: {elapsed_fp:.1f}s)")
    print(f"\n{'='*100}")
    print("DETECTION SUMMARY (Forensic vs. Baselines)")
    print(f"{'='*100}")
    print(
        f"{'Attack':<28} {'Frac':>5} "
        f"{'Forensic':>8} {'PCommit':>8} {'Wmark':>8} {'Merkle':>8} "
        f"{'Attr%':>6} {'FN%':>6}"
    )
    print(f"{'-'*100}")
    for row in det_summary:
        print(
            f"{row['attack']:<28} {row['fraction']:>5.0%} "
            f"{row['detection_rate']:>7.1%} "
            f"{row['pcommit_rate']:>7.1%} "
            f"{row['watermark_rate']:>7.1%} "
            f"{row['merkle_rate']:>7.1%} "
            f"{row['attribution_rate']:>5.1%} "
            f"{row['fn_rate']:>5.1%}"
        )

    # --- Per-attack-class aggregate (Table II style) ---
    from collections import defaultdict as _defaultdict
    _attack_buckets: Dict[str, List[Dict[str, Any]]] = _defaultdict(list)
    for row in det_summary:
        _attack_buckets[row["attack"]].append(row)

    print(f"\n{'='*80}")
    print("BASELINE COMPARISON (Table II: aggregate across all fractions)")
    print(f"{'='*80}")
    print(
        f"{'Attack class':<28} {'Forensic':>8} {'PCommit':>8} "
        f"{'Watermark':>10} {'Merkle':>8}"
    )
    print(f"{'-'*80}")
    _totals = {"forensic": 0, "pcommit": 0, "watermark": 0, "merkle": 0}
    _n_classes = 0
    for attack_name in ATTACK_CLASSES:
        rows_for_attack = _attack_buckets.get(attack_name, [])
        if not rows_for_attack:
            continue
        # A class is "detected" if avg detection rate > 50% across fractions
        forensic_avg = sum(r["detection_rate"] for r in rows_for_attack) / len(rows_for_attack)
        pcommit_avg = sum(r["pcommit_rate"] for r in rows_for_attack) / len(rows_for_attack)
        wmark_avg = sum(r["watermark_rate"] for r in rows_for_attack) / len(rows_for_attack)
        merkle_avg = sum(r["merkle_rate"] for r in rows_for_attack) / len(rows_for_attack)

        def _label(rate: float) -> str:
            if rate >= 0.99:
                return "Detected"
            elif rate >= 0.5:
                return f"Partial({rate:.0%})"
            else:
                return "Not det."

        print(
            f"{attack_name:<28} {_label(forensic_avg):>8} "
            f"{_label(pcommit_avg):>8} {_label(wmark_avg):>10} "
            f"{_label(merkle_avg):>8}"
        )
        _n_classes += 1
        _totals["forensic"] += (1 if forensic_avg >= 0.99 else 0)
        _totals["pcommit"] += (1 if pcommit_avg >= 0.99 else 0)
        _totals["watermark"] += (1 if wmark_avg >= 0.99 else 0)
        _totals["merkle"] += (1 if merkle_avg >= 0.99 else 0)

    print(f"{'-'*80}")
    print(
        f"{'Total':<28} "
        f"{_totals['forensic']}/{_n_classes:>5} "
        f"{_totals['pcommit']}/{_n_classes:>5} "
        f"  {_totals['watermark']}/{_n_classes:>7} "
        f"{_totals['merkle']}/{_n_classes:>5}"
    )

    print(f"\n{'='*60}")
    print("OPERATIONAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'K':>4} {'N':>5} {'Latency(ms)':>12} {'Evidence(B)':>12} {'Tput(t/s)':>10}")
    print(f"{'-'*60}")
    for row in ops_summary:
        print(
            f"{row['K']:>4} {row['n_steps']:>5} "
            f"{row['mean_verify_latency_s']*1000:>11.2f} "
            f"{row['mean_evidence_bytes']:>11.0f} "
            f"{row['throughput_transcripts_per_s']:>9.1f}"
        )

    print(f"\n{'='*60}")
    print("FALSE-POSITIVE MEASUREMENT")
    print(f"{'='*60}")
    print(f"  Honest transcripts tested: {fp_result.n_honest:,}")
    print(f"  False positives:           {fp_result.n_fp}")
    print(f"  FP rate:                   {fp_result.fp_rate:.4%}")
    print(f"  95% Wilson CI:             [{fp_result.wilson_lo:.6f}, {fp_result.wilson_hi:.6f}]")

    print(f"\nResults written to: {out_dir}/icufn_*.{{csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
