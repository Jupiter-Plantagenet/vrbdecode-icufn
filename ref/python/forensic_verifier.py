"""Forensic verification with reason-coded outcomes.

Implements the verification logic described in ICUFN paper Sections 3--4:
  - Policy commitment verification
  - Randomness derivation and replay detection
  - Candidate digest integrity
  - Receipt chain continuity
  - Decoding consistency re-execution

The verifier takes a transcript (evidence artifact) and the declared
policy/seed, then returns a list of VerificationResult with reason codes
suitable for triage and attribution.
"""
from __future__ import annotations

import enum
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(__file__))

from decoding_ref import decode_step
from receipt import (
    PolicyParams,
    Transcript,
    TranscriptStep,
    compute_candidate_hash,
    compute_policy_hash,
    compute_seed_commitment,
    derive_U_t,
    init_receipt,
    update_receipt,
)


# ---------------------------------------------------------------------------
# Reason codes (ICUFN Section 3.3)
# ---------------------------------------------------------------------------

class VerifyCode(enum.Enum):
    """Reason-coded verification outcomes."""

    PASS = "pass"
    POLICY_MISMATCH = "policy_mismatch"
    RANDOMNESS_REPLAY = "randomness_replay"
    RANDOMNESS_BIAS = "randomness_bias"
    CANDIDATE_MANIPULATION = "candidate_manipulation"
    TRANSCRIPT_DISCONTINUITY = "transcript_discontinuity"


@dataclass(frozen=True)
class VerificationResult:
    code: VerifyCode
    step_index: Optional[int]
    details: str


# ---------------------------------------------------------------------------
# Main verification entry point
# ---------------------------------------------------------------------------

def verify_transcript(
    transcript: Transcript,
    declared_policy: PolicyParams,
    seed: bytes,
    *,
    ground_truth_candidates: Optional[
        Sequence[tuple]  # list of (token_ids_sorted, logit_q16s_sorted)
    ] = None,
) -> List[VerificationResult]:
    """Verify a transcript against declared policy and seed.

    Parameters
    ----------
    transcript : Transcript
        The evidence artifact submitted by the provider.
    declared_policy : PolicyParams
        The policy the provider committed to.
    seed : bytes
        The 32-byte randomness seed.
    ground_truth_candidates : optional
        If provided, verifier-independent candidate sets for each step.
        Used to detect candidate-list manipulation (Section 4.3).

    Returns
    -------
    list[VerificationResult]
        One PASS entry if clean; otherwise one or more failure entries
        with reason codes for attribution.
    """
    results: List[VerificationResult] = []

    # ------------------------------------------------------------------
    # 1. Policy commitment check
    # ------------------------------------------------------------------
    expected_policy_hash = compute_policy_hash(declared_policy)
    if transcript.policy_hash != expected_policy_hash:
        results.append(VerificationResult(
            code=VerifyCode.POLICY_MISMATCH,
            step_index=None,
            details="Transcript policy_hash does not match declared policy",
        ))

    # ------------------------------------------------------------------
    # 2. Seed commitment check
    # ------------------------------------------------------------------
    expected_seed_commit = compute_seed_commitment(seed)
    if transcript.seed_commit != expected_seed_commit:
        results.append(VerificationResult(
            code=VerifyCode.POLICY_MISMATCH,
            step_index=None,
            details="Transcript seed_commit does not match provided seed",
        ))

    # ------------------------------------------------------------------
    # 3. Per-step checks
    # ------------------------------------------------------------------
    h = init_receipt(
        transcript.request_id, expected_policy_hash, expected_seed_commit,
    )

    seen_U_t: dict[int, int] = {}   # U_t value -> first step index
    prev_step_idx = -1

    for idx, step in enumerate(transcript.steps):
        # 3a. Step-index continuity
        if step.step_index != prev_step_idx + 1:
            results.append(VerificationResult(
                code=VerifyCode.TRANSCRIPT_DISCONTINUITY,
                step_index=step.step_index,
                details=(
                    f"Step index gap: expected {prev_step_idx + 1}, "
                    f"got {step.step_index}"
                ),
            ))
        prev_step_idx = step.step_index

        # 3b. Candidate-hash integrity (receipt-internal)
        expected_cand_hash = compute_candidate_hash(
            step.token_ids, step.logit_q16s,
        )
        if step.cand_hash != expected_cand_hash:
            results.append(VerificationResult(
                code=VerifyCode.CANDIDATE_MANIPULATION,
                step_index=step.step_index,
                details="cand_hash does not match provided candidates",
            ))

        # 3c. Ground-truth candidate comparison (if available)
        if ground_truth_candidates is not None:
            gt_tids, gt_logits = ground_truth_candidates[idx]
            gt_cand_hash = compute_candidate_hash(gt_tids, gt_logits)
            if step.cand_hash != gt_cand_hash:
                results.append(VerificationResult(
                    code=VerifyCode.CANDIDATE_MANIPULATION,
                    step_index=step.step_index,
                    details="cand_hash diverges from ground-truth candidates",
                ))

        # 3d. Randomness PRF derivation check
        expected_U_t = derive_U_t(
            transcript.request_id,
            expected_policy_hash,
            expected_seed_commit,
            step.step_index,
        )
        if step.U_t != expected_U_t:
            results.append(VerificationResult(
                code=VerifyCode.RANDOMNESS_REPLAY,
                step_index=step.step_index,
                details=(
                    f"U_t does not match PRF derivation "
                    f"(got {step.U_t:#x}, expected {expected_U_t:#x})"
                ),
            ))

        # 3e. Randomness collision check (replay across steps)
        if step.U_t in seen_U_t:
            results.append(VerificationResult(
                code=VerifyCode.RANDOMNESS_REPLAY,
                step_index=step.step_index,
                details=f"U_t reused from step {seen_U_t[step.U_t]}",
            ))
        seen_U_t[step.U_t] = step.step_index

        # 3f. Decoding consistency re-execution
        res = decode_step(
            K=declared_policy.K,
            top_k=declared_policy.top_k,
            top_p_q16=declared_policy.top_p_q16,
            T_q16=declared_policy.T_q16,
            token_id=step.token_ids,
            logit_q16=step.logit_q16s,
            U_t=step.U_t,
        )
        if res.y != step.y or res.Ws != step.Ws or res.R != step.R:
            results.append(VerificationResult(
                code=VerifyCode.POLICY_MISMATCH,
                step_index=step.step_index,
                details=(
                    f"Decoding re-execution mismatch: "
                    f"expected (y={res.y}, Ws={res.Ws}, R={res.R}), "
                    f"got (y={step.y}, Ws={step.Ws}, R={step.R})"
                ),
            ))

        # 3g. Receipt chain hash
        expected_h = update_receipt(
            h, transcript.request_id, expected_policy_hash,
            expected_seed_commit, step.step_index,
            step.cand_hash, step.y, step.Ws, step.R,
        )
        if step.receipt_hash != expected_h:
            results.append(VerificationResult(
                code=VerifyCode.TRANSCRIPT_DISCONTINUITY,
                step_index=step.step_index,
                details="Receipt hash chain broken at this step",
            ))
        h = expected_h  # continue with expected chain

    # ------------------------------------------------------------------
    # 4. Randomness bias heuristic (optional statistical check)
    # ------------------------------------------------------------------
    if len(transcript.steps) >= 16:
        _check_randomness_bias(transcript.steps, results)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    if not results:
        results.append(VerificationResult(
            code=VerifyCode.PASS,
            step_index=None,
            details="All checks passed",
        ))

    return results


# ---------------------------------------------------------------------------
# Randomness bias heuristic (Section 4.2)
# ---------------------------------------------------------------------------

def _check_randomness_bias(
    steps: List[TranscriptStep],
    results: List[VerificationResult],
) -> None:
    """Flag suspicious concentration in sampled token positions.

    A simple chi-squared-style test: if the same token appears far more
    often than expected under uniform sampling over the eligible set,
    flag it for analyst review.  This is a heuristic alarm, not a
    definitive detection.
    """
    from collections import Counter

    token_counts = Counter(step.y for step in steps)
    n = len(steps)
    # Flag if any single token accounts for >50% of outputs in a
    # sequence of 16+ steps (extremely unlikely under honest sampling
    # with diverse candidate sets).
    threshold = max(n // 2, 8)
    for token, count in token_counts.items():
        if count >= threshold:
            results.append(VerificationResult(
                code=VerifyCode.RANDOMNESS_BIAS,
                step_index=None,
                details=(
                    f"Token {token} selected {count}/{n} times "
                    f"(>{threshold - 1} threshold)"
                ),
            ))
