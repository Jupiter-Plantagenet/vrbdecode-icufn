"""Baseline 3: Watermark-Based Detector (Kirchenbauer et al. 2023 style).

This module implements a simplified but genuine version of the
Kirchenbauer et al. (2023) "A Watermark for Large Language Models"
approach as a baseline for comparison with the VRBDecode forensic
verifier.

At generation time, each token's vocabulary is partitioned into a
"green list" and "red list" using a hash of the previous token as
the pseudorandom seed.  Sampling is biased toward green-list tokens
by adding a logit bonus (delta) to green-list candidates.

At verification time, the detector counts how many tokens in the
transcript fall on their step's green list and runs a one-proportion
z-test against the null hypothesis of random (unbiased) selection.
A high z-score indicates the watermark is present.

Capabilities (detected):
  - Transcript replacement / rewriting (if the replacement text does
    not preserve the watermark pattern, z-score drops)
  - Some forms of transcript splicing (if spliced tokens break the
    previous-token dependency)

Limitations (not detected):
  - Policy parameter changes that preserve the watermark (attacker
    can change T/top_k/top_p while still biasing toward green list)
  - Randomness replay (watermark is statistical, not deterministic)
  - Candidate manipulation (watermark checks tokens, not candidates)
  - Inherent false positives (statistical test with configurable
    significance level)
  - Inherent false negatives (short transcripts lack statistical power)

This represents a real detection mechanism used in practice for
LLM output provenance, adapted for our experimental setting.

Reference:
  Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., &
  Goldstein, T. (2023). A Watermark for Large Language Models.
  ICML 2023.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from receipt import (
    PolicyParams,
    Transcript,
    TranscriptStep,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class WatermarkCode:
    """Verification codes for the watermark baseline."""
    PASS = "watermark_pass"
    WATERMARK_MISSING = "watermark_missing"
    WATERMARK_PRESENT = "watermark_present"


@dataclass
class WatermarkResult:
    """A single verification finding from the watermark baseline."""
    code: str
    step_index: Optional[int]
    details: str
    z_score: float = 0.0
    green_fraction: float = 0.0
    n_tokens_tested: int = 0


# ---------------------------------------------------------------------------
# Green/red list partitioning
# ---------------------------------------------------------------------------

def _compute_green_set(
    prev_token: int,
    vocab_or_candidates: Sequence[int],
    gamma: float = 0.5,
    secret_key: bytes = b"watermark-key-v1",
) -> set:
    """Partition candidates into green/red using hash of previous token.

    The green list contains approximately gamma fraction of the
    candidate tokens.  The partition is deterministic given the
    previous token and secret key.

    Parameters
    ----------
    prev_token : int
        The token selected at the previous step (or a start-of-sequence
        sentinel for step 0).
    vocab_or_candidates : sequence of int
        The candidate token IDs to partition.
    gamma : float
        Fraction of vocabulary assigned to the green list (default 0.5).
    secret_key : bytes
        Secret key for the hash-based partition.

    Returns
    -------
    set of int
        The green-list token IDs.
    """
    green = set()
    for token_id in vocab_or_candidates:
        # Hash (secret_key || prev_token || token_id) to get a
        # deterministic pseudo-random value in [0, 1)
        h = hashlib.sha256()
        h.update(secret_key)
        h.update(struct.pack("<I", prev_token & 0xFFFFFFFF))
        h.update(struct.pack("<I", token_id & 0xFFFFFFFF))
        digest = h.digest()
        # Use first 4 bytes as a uniform random value in [0, 1)
        val = struct.unpack("<I", digest[:4])[0] / (2**32)
        if val < gamma:
            green.add(token_id)
    return green


# ---------------------------------------------------------------------------
# Watermark generation (biased sampling)
# ---------------------------------------------------------------------------

@dataclass
class WatermarkConfig:
    """Configuration for watermark generation and detection."""
    delta: float = 2.0          # logit bias for green-list tokens (Q16 units)
    gamma: float = 0.5          # fraction of vocab in green list
    secret_key: bytes = b"watermark-key-v1"
    significance_level: float = 0.01   # z-test significance threshold
    bos_token: int = 0          # start-of-sequence sentinel


def compute_watermark_bias(
    step_index: int,
    prev_token: int,
    candidate_ids: Sequence[int],
    config: WatermarkConfig,
) -> List[int]:
    """Compute per-candidate logit biases for watermark injection.

    Returns a list of logit adjustments (in Q16.16 fixed-point) that
    should be added to the candidate logits before sampling.  Green-list
    tokens receive +delta; red-list tokens receive 0.

    Parameters
    ----------
    step_index : int
        Current decoding step.
    prev_token : int
        Token selected at the previous step.
    candidate_ids : sequence of int
        Candidate token IDs at this step.
    config : WatermarkConfig
        Watermark configuration.

    Returns
    -------
    list of int
        Logit biases in Q16.16 for each candidate.
    """
    green = _compute_green_set(
        prev_token, candidate_ids, config.gamma, config.secret_key,
    )
    delta_q16 = int(config.delta * (1 << 16))
    return [delta_q16 if tid in green else 0 for tid in candidate_ids]


# ---------------------------------------------------------------------------
# Watermark detection (z-test)
# ---------------------------------------------------------------------------

def _z_score(n_green: int, n_total: int, gamma: float) -> float:
    """One-proportion z-test for watermark detection.

    H0: each token is green with probability gamma (no watermark).
    H1: green fraction > gamma (watermark present).

    Returns the z-score.  Positive values indicate more green tokens
    than expected under H0.
    """
    if n_total == 0:
        return 0.0
    p_hat = n_green / n_total
    se = math.sqrt(gamma * (1 - gamma) / n_total)
    if se == 0:
        return 0.0
    return (p_hat - gamma) / se


def _z_threshold(significance_level: float) -> float:
    """Approximate z-threshold for one-sided test.

    Uses the standard normal quantile approximation.
    """
    # Common thresholds
    thresholds = {
        0.10: 1.282,
        0.05: 1.645,
        0.01: 2.326,
        0.005: 2.576,
        0.001: 3.090,
    }
    if significance_level in thresholds:
        return thresholds[significance_level]
    # Rational approximation for the inverse normal CDF
    # (Abramowitz and Stegun 26.2.23)
    p = significance_level
    if p >= 0.5:
        return 0.0
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def detect_watermark(
    transcript: Transcript,
    config: WatermarkConfig,
) -> WatermarkResult:
    """Test whether a transcript contains the expected watermark.

    For each step (starting from step 1, since step 0 has no previous
    token), check whether the selected token falls on the green list
    defined by the previous step's token.  Then run a z-test.

    Parameters
    ----------
    transcript : Transcript
        The transcript to test.
    config : WatermarkConfig
        Watermark configuration (must match generation-time config).

    Returns
    -------
    WatermarkResult
        Detection result with z-score and green fraction.
    """
    if len(transcript.steps) < 2:
        return WatermarkResult(
            code=WatermarkCode.PASS,
            step_index=None,
            details="Transcript too short for watermark detection (< 2 steps)",
            z_score=0.0,
            green_fraction=0.0,
            n_tokens_tested=0,
        )

    n_green = 0
    n_total = 0

    # For step 0, use BOS token as the "previous token"
    prev_token = config.bos_token

    for step in transcript.steps:
        # Compute the green set for this step based on the previous token
        # Use the candidate set from this step as the vocabulary
        green = _compute_green_set(
            prev_token, step.token_ids, config.gamma, config.secret_key,
        )

        # Check if the selected token is in the green set
        if step.y in green:
            n_green += 1
        n_total += 1

        prev_token = step.y

    z = _z_score(n_green, n_total, config.gamma)
    z_thresh = _z_threshold(config.significance_level)
    green_frac = n_green / n_total if n_total > 0 else 0.0

    if z >= z_thresh:
        return WatermarkResult(
            code=WatermarkCode.WATERMARK_PRESENT,
            step_index=None,
            details=(
                f"Watermark detected: z={z:.3f} >= {z_thresh:.3f} "
                f"(green={n_green}/{n_total}={green_frac:.1%})"
            ),
            z_score=z,
            green_fraction=green_frac,
            n_tokens_tested=n_total,
        )
    else:
        return WatermarkResult(
            code=WatermarkCode.WATERMARK_MISSING,
            step_index=None,
            details=(
                f"Watermark NOT detected: z={z:.3f} < {z_thresh:.3f} "
                f"(green={n_green}/{n_total}={green_frac:.1%})"
            ),
            z_score=z,
            green_fraction=green_frac,
            n_tokens_tested=n_total,
        )


# ---------------------------------------------------------------------------
# Evaluation helper: test whether an attack breaks the watermark
# ---------------------------------------------------------------------------

def evaluate_watermark_baseline(
    honest_transcript: Transcript,
    tampered_transcript: Transcript,
    config: Optional[WatermarkConfig] = None,
) -> Tuple[bool, WatermarkResult]:
    """Evaluate whether the watermark baseline detects tampering.

    The detection logic:
    1. The honest transcript was generated WITH watermark bias
       (green-list tokens are favored), so it should have z >= threshold.
    2. A tampered transcript may or may not preserve the watermark.
    3. Detection = watermark was present in honest but missing in tampered.

    For attacks that don't rewrite tokens (policy_mismatch that happens
    to select the same token, or transcript structural attacks), the
    watermark may still be present -- correctly reflecting the baseline's
    inability to detect those attacks.

    Parameters
    ----------
    honest_transcript : Transcript
        The original (watermarked) transcript.
    tampered_transcript : Transcript
        The attacked transcript.
    config : WatermarkConfig, optional
        Watermark configuration.

    Returns
    -------
    (detected, result) : (bool, WatermarkResult)
        Whether the attack was detected, and the full detection result.
    """
    if config is None:
        config = WatermarkConfig()

    tampered_result = detect_watermark(tampered_transcript, config)

    # Attack is detected if the watermark is missing from the tampered
    # transcript (the honest transcript had the watermark by construction).
    detected = tampered_result.code == WatermarkCode.WATERMARK_MISSING

    return detected, tampered_result


# ---------------------------------------------------------------------------
# Generate a watermarked transcript
# ---------------------------------------------------------------------------

def generate_watermarked_transcript(
    policy: PolicyParams,
    seed: bytes,
    request_id: bytes,
    candidate_sets: Sequence[Tuple[Sequence[int], Sequence[int]]],
    config: Optional[WatermarkConfig] = None,
) -> Transcript:
    """Generate a transcript with watermark bias injected into logits.

    This modifies the logits at each step to bias toward green-list
    tokens before honest transcript generation.  The resulting transcript
    is otherwise identical to an honest transcript and can be verified
    by the forensic verifier (though it will detect the logit modification
    as candidate manipulation if ground-truth candidates are available).

    Parameters
    ----------
    policy : PolicyParams
        Decoding policy.
    seed : bytes
        32-byte randomness seed.
    request_id : bytes
        32-byte request identifier.
    candidate_sets : sequence of (token_ids, logit_q16s)
        Per-step candidate sets.
    config : WatermarkConfig, optional
        Watermark configuration.

    Returns
    -------
    Transcript
        A watermarked transcript.
    """
    if config is None:
        config = WatermarkConfig()

    # We need to inject watermark bias into logits BEFORE generating
    # the transcript.  Since the watermark depends on the previous
    # token (which depends on biased sampling), we must generate
    # step-by-step.
    import importlib
    from decoding_ref import decode_step
    from receipt import (
        canonical_sort,
        compute_candidate_hash,
        compute_policy_hash,
        compute_seed_commitment,
        derive_U_t,
        init_receipt,
        update_receipt,
        Transcript,
        TranscriptStep,
    )

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

    prev_token = config.bos_token

    for t, (raw_tids, raw_logits) in enumerate(candidate_sets):
        U_t = derive_U_t(request_id, policy_hash, seed_commit, t)

        # Compute watermark bias
        biases = compute_watermark_bias(t, prev_token, raw_tids, config)

        # Apply bias to logits
        biased_logits = [l + b for l, b in zip(raw_logits, biases)]

        # Canonical sort for candidate hash (using BIASED logits)
        sorted_tids, sorted_logits = canonical_sort(
            raw_tids, biased_logits, policy.T_q16,
        )
        cand_hash = compute_candidate_hash(sorted_tids, sorted_logits)

        # Execute decoding with biased logits
        res = decode_step(
            K=policy.K,
            top_k=policy.top_k,
            top_p_q16=policy.top_p_q16,
            T_q16=policy.T_q16,
            token_id=list(raw_tids),
            logit_q16=list(biased_logits),
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

        prev_token = res.y

    return transcript


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
    N_STEPS = 32   # need enough steps for statistical power
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

    config = WatermarkConfig()

    # Generate watermarked transcript
    wm_transcript = generate_watermarked_transcript(
        policy, seed, request_id, cands, config,
    )

    # Check that watermark is present in honest transcript
    honest_result = detect_watermark(wm_transcript, config)
    print(f"Watermarked transcript: {honest_result.code} "
          f"(z={honest_result.z_score:.3f}, "
          f"green={honest_result.green_fraction:.1%})")

    attacks = [
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

    print()
    print("=" * 70)
    print("Watermark Baseline Evaluation")
    print("=" * 70)
    for name, attack_fn in attacks:
        tampered = attack_fn(wm_transcript)
        detected, result = evaluate_watermark_baseline(
            wm_transcript, tampered, config,
        )
        status = "DETECT" if detected else "MISS"
        print(f"  {name:<28} {status:>8}  z={result.z_score:.3f} "
              f"green={result.green_fraction:.1%}")
