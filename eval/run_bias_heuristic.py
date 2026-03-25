#!/usr/bin/env python3
"""Bias heuristic characterization for VRBDecode.

Evaluates the ``_check_randomness_bias`` heuristic in forensic_verifier.py:

  1. **False positive rate** under honest transcripts with peaked (Zipf-like)
     logit distributions (K=16, N=32, 1000 transcripts).

  2. **Detection power (ROC-like curve)** under biased sampling where the
     attacker forces a target token to be selected with probability
     p in {0.1, 0.2, ..., 0.9}.  For each p, 200 transcripts are generated
     and the detection rate measured.

Output:
  - ``eval/bias_heuristic_results.json`` with FP rates and detection rates
  - Printed text table

Usage:
  python3 eval/run_bias_heuristic.py           # full run
  python3 eval/run_bias_heuristic.py --quick   # reduced (100 FP, 50 per p)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REF_PY = os.path.join(ROOT, "ref", "python")
sys.path.insert(0, REF_PY)

from decoding_ref import Q16, Q30, decode_step, _exp_poly5_q16_16_to_q30, _mul_q30, E_Q30
from forensic_verifier import VerifyCode, verify_transcript
from receipt import (
    PolicyParams,
    Transcript,
    TranscriptStep,
    canonical_sort,
    compute_candidate_hash,
    compute_policy_hash,
    compute_seed_commitment,
    derive_U_t,
    generate_honest_transcript,
    init_receipt,
    update_receipt,
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
# Weight computation (mirrors decode_step internals)
# ---------------------------------------------------------------------------
_T_MIN_Q16 = 1
_Z_MIN_Q16 = -(12 << 16)


def _compute_weights(
    token_ids: Sequence[int],
    logit_q16s: Sequence[int],
    policy: PolicyParams,
) -> Tuple[List[int], List[int], List[int], int]:
    """Compute the weight vector used by decode_step.

    Returns (sorted_tids, sorted_logits, weights, eligible_count) where
    eligible_count is the number of candidates after top-k and top-p
    filtering (= s in the decode_step code).
    """
    K = policy.K
    T_clamped = max(int(policy.T_q16), _T_MIN_Q16)

    # Scale logits
    scaled = []
    for l in logit_q16s:
        num = int(l) << 16
        s = num // T_clamped
        s = max(-0x8000000000000000, min(0x7FFFFFFFFFFFFFFF, s))
        scaled.append(s)

    # Sort
    items = list(zip(token_ids, logit_q16s, scaled))
    items.sort(key=lambda x: (-x[2], x[0]))

    sid = [x[0] for x in items]
    slog = [x[1] for x in items]
    sscaled = [x[2] for x in items]

    k = int(policy.top_k)
    m = sscaled[0]

    w = [0] * K
    for i in range(k):
        z = sscaled[i] - m
        if z < _Z_MIN_Q16:
            z = _Z_MIN_Q16

        neg_z = -z
        n = neg_z >> 16
        if n < 0:
            n = 0
        if n > 12:
            n = 12

        r = z + (n << 16)
        p = _exp_poly5_q16_16_to_q30(r)
        wi = _mul_q30(E_Q30[n], p)
        if wi < 0:
            wi = 0
        w[i] = int(wi)

    Wk = int(sum(w[:k]))
    TH = (int(policy.top_p_q16) * Wk) >> 16

    prefix = 0
    s = 1
    for i in range(k):
        prefix += w[i]
        if prefix >= TH:
            s = i + 1
            break

    return sid, slog, w, s


def _find_U_t_for_target(
    weights: List[int],
    s: int,
    target_idx: int,
) -> int:
    """Find a U_t value that will cause decode_step to select the token
    at position target_idx (within the eligible set 0..s-1).

    The selection logic: prefix2 accumulates weights[0..i]; the first i
    where prefix2 > R is selected, where R = (U_t * Ws) >> 64.

    We need R such that sum(weights[0..target_idx-1]) < R <= sum(weights[0..target_idx]).
    Then U_t = (R << 64) / Ws (approximately).
    """
    Ws = sum(weights[:s])
    if Ws == 0:
        return 0

    if target_idx >= s:
        # Target is outside the eligible set; can't force selection
        return 0

    # prefix2 up to (target_idx - 1)
    lower = sum(weights[:target_idx])
    # prefix2 up to target_idx
    upper = lower + weights[target_idx]

    # We need lower < R <= upper
    # Pick R in the middle of the range
    R_target = (lower + upper) // 2
    if R_target <= lower:
        R_target = lower + 1
    if R_target > upper:
        R_target = upper

    # R = (U_t * Ws) >> 64, so U_t = (R << 64) / Ws
    U_t = (R_target << 64) // Ws
    # Clamp to 64-bit unsigned
    U_t = min(U_t, (1 << 64) - 1)

    return U_t


# ---------------------------------------------------------------------------
# Section 1: False Positive Measurement
# ---------------------------------------------------------------------------

def _run_false_positive_measurement(
    n_transcripts: int, K: int, N: int, seed: int,
) -> Dict[str, Any]:
    """Generate honest transcripts and measure FP rate of bias heuristic."""
    rng = random.Random(seed)

    policy = PolicyParams(
        K=K,
        top_k=max(1, K // 4),
        top_p_q16=int(0.9 * (1 << 16)),
        T_q16=1 << 16,
        max_tokens=N,
    )

    n_fp = 0
    n_bias_fp = 0
    fp_details: List[Dict[str, Any]] = []

    for i in range(n_transcripts):
        tr_seed = rng.randbytes(32)
        request_id = rng.randbytes(32)
        candidates = _generate_candidates(rng, K, N)

        transcript = generate_honest_transcript(
            policy, tr_seed, request_id, candidates,
        )

        vresults = verify_transcript(transcript, policy, tr_seed)
        is_fp = any(r.code != VerifyCode.PASS for r in vresults)
        has_bias_flag = any(r.code == VerifyCode.RANDOMNESS_BIAS for r in vresults)

        if is_fp:
            n_fp += 1
        if has_bias_flag:
            n_bias_fp += 1
            fp_details.append({
                "transcript_idx": i,
                "findings": [
                    {"code": r.code.value, "details": r.details}
                    for r in vresults if r.code != VerifyCode.PASS
                ],
            })

        if (i + 1) % max(1, n_transcripts // 10) == 0:
            print(f"    FP measurement: {i + 1}/{n_transcripts}", flush=True)

    return {
        "n_transcripts": n_transcripts,
        "K": K,
        "N": N,
        "n_false_positives_any": n_fp,
        "fp_rate_any": round(n_fp / n_transcripts, 6),
        "n_bias_false_positives": n_bias_fp,
        "fp_rate_bias": round(n_bias_fp / n_transcripts, 6),
        "fp_details": fp_details[:10],  # first 10 for brevity
    }


# ---------------------------------------------------------------------------
# Section 2: Detection Power under Actual Bias
# ---------------------------------------------------------------------------

def _run_bias_detection_power(
    bias_levels: List[float],
    n_per_level: int,
    K: int,
    N: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """For each bias level p, generate transcripts where ~p fraction of
    steps are forced to select a specific *fixed* target token, and
    measure the detection rate of the bias heuristic.

    To isolate the bias heuristic's detection power we ensure the biased
    steps output the **same** token (by injecting it into every candidate
    set with the highest logit and manipulating U_t to select it).  The
    experiment reports two detection rates:

      - ``detection_rate_any``: any verification failure (PRF mismatch,
        chain break, or bias heuristic).
      - ``detection_rate_bias_heuristic``: specifically the
        RANDOMNESS_BIAS reason code.

    This yields an ROC-like curve showing at what bias fraction the
    statistical heuristic starts to fire.
    """
    rng = random.Random(seed)

    policy = PolicyParams(
        K=K,
        top_k=max(1, K // 4),
        top_p_q16=int(0.9 * (1 << 16)),
        T_q16=1 << 16,
        max_tokens=N,
    )

    # Fixed target token that will be injected into biased steps
    TARGET_TOKEN = 7_777_777
    TARGET_LOGIT = 5 << 16  # very high logit to guarantee rank-0

    level_results: List[Dict[str, Any]] = []

    for p in bias_levels:
        n_detected = 0
        n_bias_detected = 0

        for trial in range(n_per_level):
            tr_seed = rng.randbytes(32)
            request_id = rng.randbytes(32)
            candidates = _generate_candidates(rng, K, N)

            # Build the transcript step by step, biasing some steps
            policy_hash = compute_policy_hash(policy)
            seed_commit = compute_seed_commitment(tr_seed)
            h = init_receipt(request_id, policy_hash, seed_commit)

            transcript = Transcript(
                request_id=request_id,
                policy=policy,
                seed=tr_seed,
                policy_hash=policy_hash,
                seed_commit=seed_commit,
            )

            # Decide which steps are biased
            n_biased_steps = max(1, int(N * p))
            biased_steps = set(rng.sample(range(N), n_biased_steps))

            for t in range(N):
                raw_tids, raw_logits = list(candidates[t][0]), list(candidates[t][1])
                U_t = derive_U_t(request_id, policy_hash, seed_commit, t)

                if t in biased_steps:
                    # Inject the TARGET_TOKEN with a very high logit into
                    # the candidate set (replacing the last entry). This
                    # ensures that after sorting, TARGET_TOKEN will be
                    # rank 0 (highest scaled logit).
                    # Make sure TARGET_TOKEN is not already in the set.
                    if TARGET_TOKEN in raw_tids:
                        idx = raw_tids.index(TARGET_TOKEN)
                        raw_logits[idx] = TARGET_LOGIT
                    else:
                        raw_tids[-1] = TARGET_TOKEN
                        raw_logits[-1] = TARGET_LOGIT

                    # Compute weights with the manipulated candidates
                    sid, slog, weights, s = _compute_weights(
                        raw_tids, raw_logits, policy,
                    )

                    # TARGET_TOKEN should be at rank 0 after sorting
                    # (highest logit). Find its index in the sorted order.
                    target_idx = 0
                    for i_s, tid in enumerate(sid):
                        if tid == TARGET_TOKEN:
                            target_idx = i_s
                            break

                    # Force U_t so that TARGET_TOKEN is selected
                    forced_U_t = _find_U_t_for_target(
                        weights, s, target_idx,
                    )

                    sorted_tids, sorted_logits = canonical_sort(
                        raw_tids, raw_logits, policy.T_q16,
                    )
                    cand_hash = compute_candidate_hash(sorted_tids, sorted_logits)

                    res = decode_step(
                        K=policy.K,
                        top_k=policy.top_k,
                        top_p_q16=policy.top_p_q16,
                        T_q16=policy.T_q16,
                        token_id=raw_tids,
                        logit_q16=raw_logits,
                        U_t=forced_U_t,
                    )
                    actual_U_t = forced_U_t
                else:
                    sorted_tids, sorted_logits = canonical_sort(
                        raw_tids, raw_logits, policy.T_q16,
                    )
                    cand_hash = compute_candidate_hash(sorted_tids, sorted_logits)

                    res = decode_step(
                        K=policy.K,
                        top_k=policy.top_k,
                        top_p_q16=policy.top_p_q16,
                        T_q16=policy.T_q16,
                        token_id=raw_tids,
                        logit_q16=raw_logits,
                        U_t=U_t,
                    )
                    actual_U_t = U_t

                h = update_receipt(
                    h, request_id, policy_hash, seed_commit,
                    t, cand_hash, res.y, res.Ws, res.R,
                )

                transcript.steps.append(TranscriptStep(
                    step_index=t,
                    token_ids=sorted_tids,
                    logit_q16s=sorted_logits,
                    y=res.y,
                    Ws=res.Ws,
                    R=res.R,
                    U_t=actual_U_t,
                    cand_hash=cand_hash,
                    receipt_hash=h,
                ))

            # Verify -- the bias heuristic may or may not fire depending
            # on how concentrated the outputs are
            vresults = verify_transcript(transcript, policy, tr_seed)
            detected_any = any(r.code != VerifyCode.PASS for r in vresults)
            detected_bias = any(
                r.code == VerifyCode.RANDOMNESS_BIAS for r in vresults
            )

            if detected_any:
                n_detected += 1
            if detected_bias:
                n_bias_detected += 1

        level_results.append({
            "bias_level": p,
            "n_trials": n_per_level,
            "n_detected_any": n_detected,
            "detection_rate_any": round(n_detected / n_per_level, 4),
            "n_bias_detected": n_bias_detected,
            "detection_rate_bias_heuristic": round(n_bias_detected / n_per_level, 4),
        })

        print(
            f"    p={p:.1f}: detected(any)={n_detected}/{n_per_level} "
            f"({n_detected/n_per_level:.0%}), "
            f"bias_heuristic={n_bias_detected}/{n_per_level} "
            f"({n_bias_detected/n_per_level:.0%})",
            flush=True,
        )

    return level_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Bias heuristic characterization")
    ap.add_argument("--quick", action="store_true", help="Reduced run")
    args = ap.parse_args()

    K = 16
    N = 32

    if args.quick:
        n_fp_transcripts = 100
        n_per_level = 50
    else:
        n_fp_transcripts = 1000
        n_per_level = 200

    bias_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("=" * 70)
    print("BIAS HEURISTIC CHARACTERIZATION")
    print("=" * 70)
    print(f"  K={K}, N={N}")
    print(f"  FP transcripts: {n_fp_transcripts}")
    print(f"  Per bias level:  {n_per_level}")
    print(f"  Bias levels:     {bias_levels}")
    print()

    # --- Section 1: False positive measurement ---
    print("[1/2] False positive measurement on honest transcripts...")
    t0 = time.perf_counter()
    fp_results = _run_false_positive_measurement(
        n_fp_transcripts, K, N, seed=100,
    )
    t1 = time.perf_counter()
    fp_results["elapsed_s"] = round(t1 - t0, 3)

    print(f"\n  FP rate (any):   {fp_results['fp_rate_any']:.4%}")
    print(f"  FP rate (bias):  {fp_results['fp_rate_bias']:.4%}")
    print(f"  Elapsed: {fp_results['elapsed_s']:.1f}s\n")

    # --- Section 2: Detection power ---
    print("[2/2] Detection power under actual bias...")
    t0 = time.perf_counter()
    power_results = _run_bias_detection_power(
        bias_levels, n_per_level, K, N, seed=200,
    )
    t2 = time.perf_counter()
    power_elapsed = round(t2 - t0, 3)

    # --- Output ---
    output = {
        "config": {
            "K": K,
            "N": N,
            "n_fp_transcripts": n_fp_transcripts,
            "n_per_bias_level": n_per_level,
            "bias_levels": bias_levels,
        },
        "false_positive_measurement": fp_results,
        "detection_power": power_results,
        "total_elapsed_s": round(fp_results["elapsed_s"] + power_elapsed, 3),
    }

    out_path = os.path.join(SCRIPT_DIR, "bias_heuristic_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # --- Print summary table ---
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n  False Positive Rate (honest, N={N}, K={K}):")
    print(f"    Any flag:        {fp_results['fp_rate_any']:.4%}  ({fp_results['n_false_positives_any']}/{n_fp_transcripts})")
    print(f"    Bias heuristic:  {fp_results['fp_rate_bias']:.4%}  ({fp_results['n_bias_false_positives']}/{n_fp_transcripts})")

    print(f"\n  Detection Power (ROC-like curve):")
    print(f"  {'Bias p':>8} {'Det(any)':>10} {'Det(bias)':>10} {'n':>6}")
    print(f"  {'-'*38}")
    for row in power_results:
        print(
            f"  {row['bias_level']:>8.1f} "
            f"{row['detection_rate_any']:>9.1%} "
            f"{row['detection_rate_bias_heuristic']:>9.1%} "
            f"{row['n_trials']:>6}"
        )

    print(f"\n  Total elapsed: {output['total_elapsed_s']:.1f}s")
    print(f"  Results written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
