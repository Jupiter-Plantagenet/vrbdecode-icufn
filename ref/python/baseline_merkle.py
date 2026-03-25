"""Baseline verification using signed logs with Merkle trees.

This module implements a simple alternative verification approach that
serves as a baseline comparison for the VRBDecode forensic verifier.
The baseline uses HMAC-SHA256 signatures and Merkle trees to provide
log integrity, but does NOT re-execute decoding or verify randomness
derivation.

Capabilities:
  - Detects transcript tampering (missing steps, reordering)
  - Detects post-hoc log modification (signature failure)

Limitations (by design):
  - Cannot detect policy mismatch (no decoding re-execution)
  - Cannot detect randomness replay (no PRF derivation check)
  - Cannot detect candidate manipulation (no candidate hash commitment
    against ground-truth)

These limitations are the key finding: Merkle-tree-based log signing
is insufficient for verifiable decoding because it treats the transcript
as opaque data rather than re-executing the computation.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from receipt import (
    PolicyParams,
    Transcript,
    TranscriptStep,
    generate_honest_transcript,
)
from forensic_verifier import VerifyCode, VerificationResult, verify_transcript
from attack_simulator import (
    attack_policy_mismatch,
    attack_randomness_replay,
    attack_candidate_manipulation,
    attack_transcript_drop,
    attack_transcript_reorder,
)


# ---------------------------------------------------------------------------
# Merkle tree utilities
# ---------------------------------------------------------------------------

def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def _merkle_parent(left: bytes, right: bytes) -> bytes:
    """Compute Merkle tree internal node: SHA-256(0x01 || left || right)."""
    return _sha256(b"\x01", left, right)


def _merkle_leaf(data: bytes) -> bytes:
    """Compute Merkle tree leaf node: SHA-256(0x00 || data)."""
    return _sha256(b"\x00", data)


def compute_merkle_root(leaves: Sequence[bytes]) -> bytes:
    """Compute the Merkle root of a list of leaf hashes.

    Uses a standard binary Merkle tree. If the number of leaves is odd,
    the last leaf is duplicated to fill the level.
    """
    if not leaves:
        return b"\x00" * 32

    # Compute leaf hashes
    current_level: List[bytes] = [_merkle_leaf(leaf) for leaf in leaves]

    while len(current_level) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            if i + 1 < len(current_level):
                right = current_level[i + 1]
            else:
                right = left  # duplicate last node
            next_level.append(_merkle_parent(left, right))
        current_level = next_level

    return current_level[0]


# ---------------------------------------------------------------------------
# Step serialization for signing
# ---------------------------------------------------------------------------

def _serialize_step(step: TranscriptStep) -> bytes:
    """Serialize a transcript step into a canonical byte string for signing."""
    import struct
    parts: List[bytes] = []
    parts.append(struct.pack("<I", step.step_index))
    parts.append(struct.pack("<I", step.y))
    parts.append(struct.pack("<Q", step.Ws & 0xFFFFFFFFFFFFFFFF))
    parts.append(struct.pack("<Q", step.R & 0xFFFFFFFFFFFFFFFF))
    parts.append(struct.pack("<Q", step.U_t & 0xFFFFFFFFFFFFFFFF))
    # Include candidate data
    for tid in step.token_ids:
        parts.append(struct.pack("<I", tid & 0xFFFFFFFF))
    for logit in step.logit_q16s:
        parts.append(struct.pack("<i", logit))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Baseline verification result codes
# ---------------------------------------------------------------------------

class BaselineCode:
    """Verification codes for the Merkle baseline."""
    PASS = "baseline_pass"
    SIGNATURE_INVALID = "baseline_signature_invalid"
    MERKLE_ROOT_MISMATCH = "baseline_merkle_root_mismatch"
    STEP_INDEX_GAP = "baseline_step_index_gap"


@dataclass
class BaselineResult:
    """A single verification finding from the Merkle baseline."""
    code: str
    step_index: Optional[int]
    details: str


# ---------------------------------------------------------------------------
# Signed transcript
# ---------------------------------------------------------------------------

@dataclass
class SignedTranscript:
    """A transcript with per-step HMAC signatures and a Merkle root."""
    request_id: bytes
    step_signatures: List[bytes]    # HMAC-SHA256 per step
    step_data: List[bytes]          # serialized step data
    merkle_root: bytes              # root of Merkle tree over step_data
    step_indices: List[int]         # step indices for completeness check
    n_steps: int                    # declared number of steps


# ---------------------------------------------------------------------------
# MerkleBaseline class
# ---------------------------------------------------------------------------

class MerkleBaseline:
    """Signed-log verification baseline using HMAC-SHA256 + Merkle tree.

    This class represents the strongest reasonable alternative to
    VRBDecode's forensic verifier that does NOT re-execute the decoding
    computation. It demonstrates that log-integrity checking alone is
    insufficient for detecting semantic attacks on decoding.
    """

    def __init__(self, shared_secret: bytes = b"baseline-secret-key-32bytes!!!!!"):
        """Initialize with a shared HMAC secret.

        Parameters
        ----------
        shared_secret : bytes
            Shared secret for HMAC-SHA256 signing. In a real deployment
            this would be established via a key exchange protocol.
        """
        self.shared_secret = shared_secret

    def _hmac_sign(self, data: bytes) -> bytes:
        """Compute HMAC-SHA256 over data using the shared secret."""
        return hmac.new(self.shared_secret, data, hashlib.sha256).digest()

    def sign_transcript(self, transcript: Transcript) -> SignedTranscript:
        """Sign each step and compute the Merkle root.

        Parameters
        ----------
        transcript : Transcript
            The transcript to sign (honest or tampered — the baseline
            cannot distinguish).

        Returns
        -------
        SignedTranscript
            The signed representation with per-step signatures and
            Merkle root.
        """
        step_data: List[bytes] = []
        signatures: List[bytes] = []
        step_indices: List[int] = []

        for step in transcript.steps:
            data = _serialize_step(step)
            step_data.append(data)
            signatures.append(self._hmac_sign(data))
            step_indices.append(step.step_index)

        merkle_root = compute_merkle_root(step_data)

        return SignedTranscript(
            request_id=transcript.request_id,
            step_signatures=signatures,
            step_data=step_data,
            merkle_root=merkle_root,
            step_indices=step_indices,
            n_steps=len(transcript.steps),
        )

    def verify_transcript(
        self,
        signed: SignedTranscript,
        expected_merkle_root: Optional[bytes] = None,
    ) -> List[BaselineResult]:
        """Verify a signed transcript.

        Checks:
          (a) Signature validity per step
          (b) Merkle root integrity (if expected root provided)
          (c) Step-index completeness (contiguous 0..N-1)

        Parameters
        ----------
        signed : SignedTranscript
            The signed transcript to verify.
        expected_merkle_root : bytes, optional
            If provided, the Merkle root is compared against this value.
            This simulates a scenario where the root was published to a
            bulletin board or blockchain.

        Returns
        -------
        list[BaselineResult]
            Verification findings. Empty list + PASS if all checks pass.
        """
        results: List[BaselineResult] = []

        # (a) Signature validity per step
        for i, (data, sig) in enumerate(
            zip(signed.step_data, signed.step_signatures)
        ):
            expected_sig = self._hmac_sign(data)
            if not hmac.compare_digest(sig, expected_sig):
                results.append(BaselineResult(
                    code=BaselineCode.SIGNATURE_INVALID,
                    step_index=signed.step_indices[i] if i < len(signed.step_indices) else i,
                    details=f"HMAC signature invalid at position {i}",
                ))

        # (b) Merkle root integrity
        recomputed_root = compute_merkle_root(signed.step_data)
        if expected_merkle_root is not None:
            if recomputed_root != expected_merkle_root:
                results.append(BaselineResult(
                    code=BaselineCode.MERKLE_ROOT_MISMATCH,
                    step_index=None,
                    details=(
                        f"Merkle root mismatch: expected "
                        f"{expected_merkle_root.hex()[:16]}..., "
                        f"got {recomputed_root.hex()[:16]}..."
                    ),
                ))

        # (c) Step-index completeness
        if signed.step_indices:
            expected_indices = list(range(len(signed.step_indices)))
            if signed.step_indices != expected_indices:
                # Check for gaps
                for i in range(len(signed.step_indices)):
                    if signed.step_indices[i] != i:
                        results.append(BaselineResult(
                            code=BaselineCode.STEP_INDEX_GAP,
                            step_index=signed.step_indices[i],
                            details=(
                                f"Step index discontinuity: expected {i}, "
                                f"got {signed.step_indices[i]}"
                            ),
                        ))
                        break

        if not results:
            results.append(BaselineResult(
                code=BaselineCode.PASS,
                step_index=None,
                details="All baseline checks passed",
            ))

        return results


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    """One row in the comparison table."""
    attack_class: str
    forensic_detected: bool
    forensic_codes: List[str]
    baseline_detected: bool
    baseline_codes: List[str]
    key_difference: str


def compare_approaches(
    transcript: Transcript,
    seed: bytes,
    policy: PolicyParams,
    ground_truth_candidates: Optional[List[Tuple[List[int], List[int]]]] = None,
) -> List[ComparisonRow]:
    """Run both verifiers on each attack and produce a comparison table.

    Parameters
    ----------
    transcript : Transcript
        An honest transcript to use as the basis for attacks.
    seed : bytes
        The randomness seed for the forensic verifier.
    policy : PolicyParams
        The declared policy.
    ground_truth_candidates : optional
        Ground-truth candidate sets for candidate manipulation detection.

    Returns
    -------
    list[ComparisonRow]
        One row per attack class showing detection capabilities.
    """
    import random as _random

    baseline = MerkleBaseline()
    rows: List[ComparisonRow] = []

    # Define attacks
    attacks = [
        (
            "policy_mismatch",
            lambda t: attack_policy_mismatch(
                t, fraction=1.0, new_T_q16=max(1, policy.T_q16 // 2),
                rng=_random.Random(0),
            ),
            "Baseline lacks decoding re-execution; cannot detect altered parameters",
        ),
        (
            "randomness_replay",
            lambda t: attack_randomness_replay(
                t, fraction=1.0, rng=_random.Random(0),
            ),
            "Baseline lacks PRF derivation check; cannot detect U_t manipulation",
        ),
        (
            "candidate_manipulation",
            lambda t: attack_candidate_manipulation(
                t, fraction=1.0, rng=_random.Random(0),
            ),
            "Baseline lacks ground-truth candidate comparison; signs whatever data it receives",
        ),
        (
            "transcript_drop",
            lambda t: attack_transcript_drop(
                t, fraction=0.25, rng=_random.Random(1),
            ),
            "Both detect: step-index gap is visible to any verifier checking completeness",
        ),
        (
            "transcript_reorder",
            lambda t: attack_transcript_reorder(
                t, n_swaps=2, rng=_random.Random(0),
            ),
            "Both detect: step ordering violation is visible to any verifier",
        ),
    ]

    for attack_name, attack_fn, key_diff in attacks:
        tampered = attack_fn(transcript)

        # --- Forensic verifier ---
        forensic_results = verify_transcript(
            tampered, policy, seed,
            ground_truth_candidates=ground_truth_candidates,
        )
        forensic_codes = [r.code.value for r in forensic_results]
        forensic_detected = not all(
            r.code == VerifyCode.PASS for r in forensic_results
        )

        # --- Merkle baseline ---
        # Sign the TAMPERED transcript (simulating the attacker providing
        # a signed log of their tampered computation)
        signed = baseline.sign_transcript(tampered)
        # The baseline verifier has no expected Merkle root to compare
        # against (the attacker computes a fresh root over their tampered
        # data). So we only check step-index completeness and signature
        # validity.
        baseline_results = baseline.verify_transcript(signed)
        baseline_codes = [r.code for r in baseline_results]
        baseline_detected = not all(
            r.code == BaselineCode.PASS for r in baseline_results
        )

        rows.append(ComparisonRow(
            attack_class=attack_name,
            forensic_detected=forensic_detected,
            forensic_codes=forensic_codes,
            baseline_detected=baseline_detected,
            baseline_codes=baseline_codes,
            key_difference=key_diff,
        ))

    return rows


def format_comparison_table(rows: Optional[List[ComparisonRow]] = None) -> str:
    """Format the comparison table as a human-readable string.

    If rows is None, generates a fresh comparison using default parameters.
    """
    import random as _random

    if rows is None:
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
        gt_cands = [(s.token_ids[:], s.logit_q16s[:]) for s in transcript.steps]
        rows = compare_approaches(transcript, seed, policy, gt_cands)

    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("Comparison: VRBDecode Forensic Verifier vs. Merkle Baseline")
    lines.append("=" * 90)
    lines.append("")

    hdr = (
        f"{'Attack Class':<26} {'Forensic':>10} {'Baseline':>10}   "
        f"{'Key Difference'}"
    )
    lines.append(hdr)
    lines.append("-" * 90)

    for row in rows:
        forensic_str = "DETECT" if row.forensic_detected else "MISS"
        baseline_str = "DETECT" if row.baseline_detected else "MISS"
        lines.append(
            f"{row.attack_class:<26} {forensic_str:>10} {baseline_str:>10}   "
            f"{row.key_difference}"
        )

    lines.append("-" * 90)
    lines.append("")

    # Summary
    forensic_detect = sum(1 for r in rows if r.forensic_detected)
    baseline_detect = sum(1 for r in rows if r.baseline_detected)
    lines.append(
        f"Detection rate: Forensic {forensic_detect}/{len(rows)}, "
        f"Baseline {baseline_detect}/{len(rows)}"
    )
    lines.append("")
    lines.append(
        "Conclusion: The Merkle baseline detects only structural tampering "
        "(missing/reordered steps) but is blind to semantic attacks "
        "(policy mismatch, randomness replay, candidate manipulation). "
        "VRBDecode's forensic verifier detects all attack classes by "
        "re-executing the deterministic decoding computation and verifying "
        "cryptographic commitments."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(format_comparison_table())
