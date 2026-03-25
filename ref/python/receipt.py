"""Receipt generation and chaining for VRBDecode forensic audit.

Implements the receipt model from ReceiptSpec_v1 and PublicInputsSpec_v1
using SHA-256 as the prototype hash function.  Production deployments
use Poseidon for in-circuit proofs; the domain-separation strings and
field layout are identical.
"""
from __future__ import annotations

import hashlib
import json as _json
import struct
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyParams:
    K: int
    top_k: int
    top_p_q16: int
    T_q16: int
    max_tokens: int


@dataclass
class TranscriptStep:
    step_index: int
    token_ids: List[int]       # candidate set in canonical order (K elems)
    logit_q16s: List[int]      # matching logits in canonical order (K elems)
    y: int                     # selected token
    Ws: int                    # weight sum (Q30)
    R: int                     # sampling threshold
    U_t: int                   # per-step randomness
    cand_hash: bytes           # candidate digest  (32 bytes)
    receipt_hash: bytes        # chain hash h_t    (32 bytes)


@dataclass
class Transcript:
    request_id: bytes          # 32 bytes
    policy: PolicyParams
    seed: bytes                # 32 bytes
    policy_hash: bytes         # 32 bytes
    seed_commit: bytes         # 32 bytes
    steps: List[TranscriptStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic encoding helpers (matches spec field packing)
# ---------------------------------------------------------------------------

def _u32_le(x: int) -> bytes:
    return struct.pack("<I", x & 0xFFFFFFFF)


def _u64_le(x: int) -> bytes:
    return struct.pack("<Q", x & 0xFFFFFFFFFFFFFFFF)


def _i32_as_u32_le(x: int) -> bytes:
    """Two's-complement bitcast i32 -> u32, little-endian."""
    return struct.pack("<i", x)


def _domain_hash(domain: str, *parts: bytes) -> bytes:
    """SHA-256 with domain separation (mirrors Poseidon layout)."""
    h = hashlib.sha256()
    h.update(domain.encode("ascii"))
    for p in parts:
        h.update(p)
    return h.digest()


# ---------------------------------------------------------------------------
# Canonical candidate ordering (DecodingSpec_v1 Section 2)
# ---------------------------------------------------------------------------

_T_MIN_Q16 = 1


def canonical_sort(
    token_ids: Sequence[int],
    logit_q16s: Sequence[int],
    T_q16: int,
) -> Tuple[List[int], List[int]]:
    """Return (sorted_token_ids, sorted_logit_q16s) in canonical order.

    Canonical order: scaled_logit DESC, token_id ASC.
    """
    T_clamped = max(int(T_q16), _T_MIN_Q16)
    items: list = []
    for tid, logit in zip(token_ids, logit_q16s):
        scaled = (int(logit) << 16) // T_clamped
        scaled = max(-0x8000000000000000, min(0x7FFFFFFFFFFFFFFF, scaled))
        items.append((tid, logit, scaled))
    items.sort(key=lambda x: (-x[2], x[0]))
    return [x[0] for x in items], [x[1] for x in items]


# ---------------------------------------------------------------------------
# Commitment functions (PublicInputsSpec_v1)
# ---------------------------------------------------------------------------

def compute_policy_hash(policy: PolicyParams) -> bytes:
    """Compute policy commitment (SHA-256 prototype of Poseidon commitment)."""
    return _domain_hash(
        "VRBDecode.Policy.v1",
        _u32_le(policy.K),
        _u32_le(policy.top_k),
        _u32_le(policy.top_p_q16),
        _u32_le(policy.T_q16),
        _u32_le(policy.max_tokens),
        _u32_le(1),  # hash_fn_id  (Poseidon)
        _u32_le(1),  # exp_approx_id (ExpPoly5)
    )


def compute_seed_commitment(seed: bytes) -> bytes:
    """Commit to the 32-byte randomness seed."""
    assert len(seed) == 32, "seed must be 32 bytes"
    return _domain_hash("VRBDecode.SeedCommit.v1", seed)


def compute_candidate_hash(
    token_ids_sorted: Sequence[int],
    logit_q16_sorted: Sequence[int],
) -> bytes:
    """Candidate-set digest over canonically sorted candidates."""
    parts: list = []
    for tid, logit in zip(token_ids_sorted, logit_q16_sorted):
        parts.append(_u32_le(tid))
        parts.append(_i32_as_u32_le(logit))
    return _domain_hash("VRBDecode.Candidates.v1", *parts)


# ---------------------------------------------------------------------------
# Receipt chain (ReceiptSpec_v1)
# ---------------------------------------------------------------------------

def init_receipt(
    request_id: bytes,
    policy_hash: bytes,
    seed_commit: bytes,
) -> bytes:
    """Compute initial receipt state h_0."""
    return _domain_hash(
        "VRBDecode.ReceiptInit.v1",
        request_id,
        policy_hash,
        seed_commit,
    )


def update_receipt(
    h_prev: bytes,
    request_id: bytes,
    policy_hash: bytes,
    seed_commit: bytes,
    step_idx: int,
    cand_hash: bytes,
    y: int,
    Ws: int,
    R: int,
) -> bytes:
    """Compute receipt state h_t from h_{t-1} and step data."""
    return _domain_hash(
        "VRBDecode.Receipt.v1",
        h_prev,
        request_id,
        policy_hash,
        seed_commit,
        _u32_le(step_idx),
        cand_hash,
        _u32_le(y),
        _u64_le(Ws),
        _u64_le(R),
    )


# ---------------------------------------------------------------------------
# Per-step randomness derivation (PublicInputsSpec_v1 Section 2.2)
# ---------------------------------------------------------------------------

def derive_U_t(
    request_id: bytes,
    policy_hash: bytes,
    seed_commit: bytes,
    step_idx: int,
) -> int:
    """Derive per-step randomness U_t via PRF."""
    raw = _domain_hash(
        "VRBDecode.U_t.v1",
        request_id,
        policy_hash,
        seed_commit,
        _u32_le(step_idx),
    )
    return struct.unpack("<Q", raw[:8])[0]


# ---------------------------------------------------------------------------
# Honest transcript generation
# ---------------------------------------------------------------------------

def generate_honest_transcript(
    policy: PolicyParams,
    seed: bytes,
    request_id: bytes,
    candidate_sets: Sequence[Tuple[Sequence[int], Sequence[int]]],
) -> Transcript:
    """Build a fully-chained honest transcript.

    Parameters
    ----------
    candidate_sets : list of (token_ids, logit_q16s) per step.
        Candidates may be in arbitrary order; they are canonically sorted
        before hashing.
    """
    import importlib, os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from decoding_ref import decode_step  # local import avoids circular dep

    policy_hash = compute_policy_hash(policy)
    seed_commit = compute_seed_commitment(seed)
    h = init_receipt(request_id, policy_hash, seed_commit)

    transcript = Transcript(
        request_id=request_id,
        policy=policy,
        seed=seed,
        policy_hash=policy_hash,
        seed_commit=seed_commit,
    )

    for t, (raw_tids, raw_logits) in enumerate(candidate_sets):
        U_t = derive_U_t(request_id, policy_hash, seed_commit, t)

        # Canonical sort for candidate hash
        sorted_tids, sorted_logits = canonical_sort(raw_tids, raw_logits, policy.T_q16)
        cand_hash = compute_candidate_hash(sorted_tids, sorted_logits)

        # Execute decoding (decode_step sorts internally)
        res = decode_step(
            K=policy.K,
            top_k=policy.top_k,
            top_p_q16=policy.top_p_q16,
            T_q16=policy.T_q16,
            token_id=list(raw_tids),
            logit_q16=list(raw_logits),
            U_t=U_t,
        )

        # Update receipt chain
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
            U_t=U_t,
            cand_hash=cand_hash,
            receipt_hash=h,
        ))

    return transcript


# ---------------------------------------------------------------------------
# Canonical JSON serialization of a full evidence artifact
# ---------------------------------------------------------------------------

def serialize_transcript(transcript: Transcript) -> bytes:
    """Serialize the full evidence artifact to canonical JSON bytes.

    The resulting JSON includes everything needed for independent
    verification: policy params, seed, request_id, and per-step data
    (token_ids, logit_q16s, y, Ws, R, U_t, cand_hash, receipt_hash).
    This matches the JSONL vector format used elsewhere in the repo.
    """
    obj = {
        "request_id": transcript.request_id.hex(),
        "seed": transcript.seed.hex(),
        "policy_hash": transcript.policy_hash.hex(),
        "seed_commit": transcript.seed_commit.hex(),
        "policy": {
            "K": transcript.policy.K,
            "top_k": transcript.policy.top_k,
            "top_p_q16": transcript.policy.top_p_q16,
            "T_q16": transcript.policy.T_q16,
            "max_tokens": transcript.policy.max_tokens,
        },
        "steps": [
            {
                "step_index": s.step_index,
                "token_ids": s.token_ids,
                "logit_q16s": s.logit_q16s,
                "y": s.y,
                "Ws": s.Ws,
                "R": s.R,
                "U_t": s.U_t,
                "cand_hash": s.cand_hash.hex(),
                "receipt_hash": s.receipt_hash.hex(),
            }
            for s in transcript.steps
        ],
    }
    # sort_keys for canonical ordering; no extra whitespace
    return _json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
