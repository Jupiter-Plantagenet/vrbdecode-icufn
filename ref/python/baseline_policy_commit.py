"""Baseline 2: Policy-Commitment Verifier (without re-execution).

This module implements a genuinely stronger baseline than the Merkle-log
approach.  It binds policy parameters to each decoding step via a hash
commitment and verifies that all steps reference the same commitment,
that step indices are contiguous, and that per-step HMAC signatures are
valid.

Crucially, this baseline does NOT re-execute decoding and does NOT check
intermediate values (Ws, R) or randomness derivation (U_t).  This is the
key architectural difference from the full forensic verifier.

Capabilities (detected):
  - Policy commitment changes between steps (different policy_hash)
  - Transcript drops / reorders (step-index gaps)
  - Post-hoc modification of signed step data (HMAC failure)

Limitations (not detected):
  - Policy mismatch where the attacker uses wrong parameters while
    referencing the correct commitment (no re-execution to check)
  - Randomness replay (no PRF derivation check)
  - Candidate manipulation (no candidate hash vs. ground truth)

This represents a realistic "policy binding without re-execution"
approach used in practice by systems that commit to configuration
but treat execution as a black box.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct
import sys
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from receipt import (
    PolicyParams,
    Transcript,
    TranscriptStep,
    compute_policy_hash,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class PolicyCommitCode:
    """Verification codes for the policy-commitment baseline."""
    PASS = "pcommit_pass"
    POLICY_HASH_INCONSISTENT = "pcommit_policy_hash_inconsistent"
    STEP_INDEX_GAP = "pcommit_step_index_gap"
    SIGNATURE_INVALID = "pcommit_signature_invalid"


@dataclass
class PolicyCommitResult:
    """A single verification finding from the policy-commitment baseline."""
    code: str
    step_index: Optional[int]
    details: str


# ---------------------------------------------------------------------------
# Signed step record
# ---------------------------------------------------------------------------

@dataclass
class PolicyCommitStepRecord:
    """Per-step record: policy_hash, token_id, step_index, HMAC."""
    step_index: int
    policy_hash: bytes       # 32 bytes -- commitment to (K, top_k, top_p, T)
    token_id: int            # selected token y
    hmac_sig: bytes          # HMAC-SHA256 over (step_index || policy_hash || token_id)


@dataclass
class PolicyCommitSignedTranscript:
    """A transcript signed by the policy-commitment baseline."""
    request_id: bytes
    declared_policy_hash: bytes   # the policy hash committed at generation time
    records: List[PolicyCommitStepRecord]


# ---------------------------------------------------------------------------
# PolicyCommitBaseline class
# ---------------------------------------------------------------------------

class PolicyCommitBaseline:
    """Policy-commitment verification baseline.

    At generation time, the provider commits to policy parameters via
    a hash and signs each step record (step_index, policy_hash, token_id)
    with HMAC.  At verification time, the verifier checks:

      (a) All steps reference the same policy_hash
      (b) Step indices are contiguous 0..N-1
      (c) Per-step HMAC signatures are valid

    The verifier does NOT re-execute decoding, does NOT check Ws/R,
    and does NOT verify U_t derivation.
    """

    def __init__(self, shared_secret: bytes = b"pcommit-secret-key-32bytes!!!!!"):
        self.shared_secret = shared_secret

    def _serialize_record(self, step_index: int, policy_hash: bytes, token_id: int) -> bytes:
        """Canonical serialization for HMAC signing."""
        return b"".join([
            b"PolicyCommit.Step.v1",
            struct.pack("<I", step_index),
            policy_hash,
            struct.pack("<I", token_id & 0xFFFFFFFF),
        ])

    def _hmac_sign(self, data: bytes) -> bytes:
        return hmac.new(self.shared_secret, data, hashlib.sha256).digest()

    def sign_transcript(self, transcript: Transcript) -> PolicyCommitSignedTranscript:
        """Sign a transcript at generation time.

        Computes a policy_hash from the transcript's declared policy
        and signs each step's (step_index, policy_hash, y) tuple.

        Note: the attacker can sign a tampered transcript.  The baseline
        can only check that the *signed data* is internally consistent --
        it cannot check that the signed data matches what honest
        execution would have produced.
        """
        policy_hash = compute_policy_hash(transcript.policy)
        records: List[PolicyCommitStepRecord] = []

        for step in transcript.steps:
            data = self._serialize_record(step.step_index, policy_hash, step.y)
            sig = self._hmac_sign(data)
            records.append(PolicyCommitStepRecord(
                step_index=step.step_index,
                policy_hash=policy_hash,
                token_id=step.y,
                hmac_sig=sig,
            ))

        return PolicyCommitSignedTranscript(
            request_id=transcript.request_id,
            declared_policy_hash=policy_hash,
            records=records,
        )

    def verify_transcript(
        self,
        signed: PolicyCommitSignedTranscript,
    ) -> List[PolicyCommitResult]:
        """Verify a policy-commitment signed transcript.

        Checks:
          (a) All steps reference the same policy_hash as declared
          (b) Step indices are contiguous 0..N-1
          (c) Per-step HMAC signatures are valid

        Returns
        -------
        list[PolicyCommitResult]
            Verification findings.  Single PASS entry if clean.
        """
        results: List[PolicyCommitResult] = []

        # (a) Policy hash consistency
        for rec in signed.records:
            if rec.policy_hash != signed.declared_policy_hash:
                results.append(PolicyCommitResult(
                    code=PolicyCommitCode.POLICY_HASH_INCONSISTENT,
                    step_index=rec.step_index,
                    details=(
                        f"Step {rec.step_index} policy_hash "
                        f"{rec.policy_hash.hex()[:16]}... differs from "
                        f"declared {signed.declared_policy_hash.hex()[:16]}..."
                    ),
                ))

        # (b) Step-index continuity
        indices = [rec.step_index for rec in signed.records]
        expected = list(range(len(indices)))
        if indices != expected:
            for i, (got, want) in enumerate(zip(indices, expected)):
                if got != want:
                    results.append(PolicyCommitResult(
                        code=PolicyCommitCode.STEP_INDEX_GAP,
                        step_index=got,
                        details=(
                            f"Step index discontinuity at position {i}: "
                            f"expected {want}, got {got}"
                        ),
                    ))
                    break
            # Also check if fewer steps than expected from max index
            if len(indices) > 0 and indices[-1] >= len(indices):
                results.append(PolicyCommitResult(
                    code=PolicyCommitCode.STEP_INDEX_GAP,
                    step_index=None,
                    details=(
                        f"Step count {len(indices)} inconsistent with "
                        f"max index {indices[-1]}"
                    ),
                ))

        # (c) HMAC signature validity
        for rec in signed.records:
            data = self._serialize_record(
                rec.step_index, rec.policy_hash, rec.token_id,
            )
            expected_sig = self._hmac_sign(data)
            if not hmac.compare_digest(rec.hmac_sig, expected_sig):
                results.append(PolicyCommitResult(
                    code=PolicyCommitCode.SIGNATURE_INVALID,
                    step_index=rec.step_index,
                    details=f"HMAC signature invalid at step {rec.step_index}",
                ))

        if not results:
            results.append(PolicyCommitResult(
                code=PolicyCommitCode.PASS,
                step_index=None,
                details="All policy-commitment checks passed",
            ))

        return results


# ---------------------------------------------------------------------------
# Convenience: run baseline on a transcript (for eval integration)
# ---------------------------------------------------------------------------

def evaluate_policy_commit_baseline(
    transcript: Transcript,
    shared_secret: bytes = b"pcommit-secret-key-32bytes!!!!!",
) -> List[PolicyCommitResult]:
    """One-shot: sign the transcript and verify it.

    This simulates the attacker providing a signed transcript of their
    (possibly tampered) computation.  The baseline signs whatever it
    receives, then verifies internal consistency.
    """
    baseline = PolicyCommitBaseline(shared_secret)
    signed = baseline.sign_transcript(transcript)
    return baseline.verify_transcript(signed)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _random
    from receipt import generate_honest_transcript
    from attack_simulator import (
        attack_policy_mismatch,
        attack_randomness_replay,
        attack_candidate_manipulation,
        attack_transcript_drop,
        attack_transcript_reorder,
    )

    K = 16
    N_STEPS = 8
    policy = PolicyParams(K=K, top_k=4, top_p_q16=int(0.9 * (1 << 16)),
                          T_q16=1 << 16, max_tokens=N_STEPS)
    seed = b"\x01" * 32
    request_id = b"\xaa" * 32
    rng = _random.Random(42)
    cands = []
    for _ in range(N_STEPS):
        tids = rng.sample(range(1, 10_000_000), K)
        logits = [rng.randint(-(2 << 16), 2 << 16) for _ in range(K)]
        cands.append((tids, logits))
    transcript = generate_honest_transcript(policy, seed, request_id, cands)

    attacks = [
        ("honest", lambda t: t),
        ("policy_mismatch", lambda t: attack_policy_mismatch(
            t, fraction=1.0, new_T_q16=max(1, policy.T_q16 // 2), rng=_random.Random(0))),
        ("randomness_replay", lambda t: attack_randomness_replay(
            t, fraction=1.0, rng=_random.Random(0))),
        ("candidate_manipulation", lambda t: attack_candidate_manipulation(
            t, fraction=1.0, rng=_random.Random(0))),
        ("transcript_drop", lambda t: attack_transcript_drop(
            t, fraction=0.25, rng=_random.Random(1))),
        ("transcript_reorder", lambda t: attack_transcript_reorder(
            t, n_swaps=2, rng=_random.Random(0))),
    ]

    print("=" * 70)
    print("Policy-Commitment Baseline Evaluation")
    print("=" * 70)
    for name, attack_fn in attacks:
        tampered = attack_fn(transcript)
        results = evaluate_policy_commit_baseline(tampered)
        detected = not all(r.code == PolicyCommitCode.PASS for r in results)
        codes = [r.code for r in results]
        status = "DETECT" if detected else "MISS"
        print(f"  {name:<28} {status:>8}  codes={codes}")
