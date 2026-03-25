"""Attack simulation for four adversarial classes (ICUFN Section 4).

Each function takes an honest transcript and returns a tampered copy.
The tampered transcript is self-consistent in the fields the attacker
controls, but the forensic verifier can detect the tampering through
the checks described in the paper.

Attack classes
--------------
1. Policy mismatch       -- alter T / top_k / top_p for selected steps
2. Randomness replay     -- reuse U_t from another step
3. Candidate manipulation-- inject / remove shortlist entries
4. Transcript tampering  -- drop or reorder steps
"""
from __future__ import annotations

import copy
import os
import random as _random
import sys
from typing import List, Optional, Set

sys.path.insert(0, os.path.dirname(__file__))

from decoding_ref import Q16, decode_step
from receipt import (
    PolicyParams,
    Transcript,
    TranscriptStep,
    canonical_sort,
    compute_candidate_hash,
    compute_policy_hash,
    compute_seed_commitment,
    derive_U_t,
    init_receipt,
    update_receipt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rechain(transcript: Transcript) -> None:
    """Recompute the receipt hash chain in-place after mutations."""
    h = init_receipt(
        transcript.request_id,
        transcript.policy_hash,
        transcript.seed_commit,
    )
    for step in transcript.steps:
        h = update_receipt(
            h, transcript.request_id,
            transcript.policy_hash, transcript.seed_commit,
            step.step_index, step.cand_hash,
            step.y, step.Ws, step.R,
        )
        step.receipt_hash = h


def _select_affected(n_steps: int, fraction: float, rng: _random.Random) -> Set[int]:
    """Pick a random subset of step indices to attack."""
    count = max(1, int(n_steps * fraction))
    return set(rng.sample(range(n_steps), min(count, n_steps)))


# ---------------------------------------------------------------------------
# Attack 1 -- Policy mismatch (Section 4.1)
# ---------------------------------------------------------------------------

def attack_policy_mismatch(
    transcript: Transcript,
    fraction: float = 0.5,
    *,
    new_T_q16: Optional[int] = None,
    new_top_p_q16: Optional[int] = None,
    new_top_k: Optional[int] = None,
    rng: Optional[_random.Random] = None,
) -> Transcript:
    """Re-execute affected steps with altered policy parameters.

    The attacker changes decoding behavior but keeps the original
    policy commitment.  The receipt chain is recomputed so it is
    internally consistent with the altered outputs.  Detection:
    decoding re-execution with declared policy produces different
    (y, Ws, R).
    """
    if rng is None:
        rng = _random.Random(42)

    tampered = copy.deepcopy(transcript)
    affected = _select_affected(len(tampered.steps), fraction, rng)

    altered_policy = PolicyParams(
        K=transcript.policy.K,
        top_k=new_top_k if new_top_k is not None else transcript.policy.top_k,
        top_p_q16=(new_top_p_q16 if new_top_p_q16 is not None
                   else transcript.policy.top_p_q16),
        T_q16=new_T_q16 if new_T_q16 is not None else transcript.policy.T_q16,
        max_tokens=transcript.policy.max_tokens,
    )

    for step in tampered.steps:
        if step.step_index in affected:
            res = decode_step(
                K=altered_policy.K,
                top_k=altered_policy.top_k,
                top_p_q16=altered_policy.top_p_q16,
                T_q16=altered_policy.T_q16,
                token_id=step.token_ids,
                logit_q16=step.logit_q16s,
                U_t=step.U_t,
            )
            step.y = res.y
            step.Ws = res.Ws
            step.R = res.R

    _rechain(tampered)
    return tampered


# ---------------------------------------------------------------------------
# Attack 2 -- Randomness replay / bias (Section 4.2)
# ---------------------------------------------------------------------------

def attack_randomness_replay(
    transcript: Transcript,
    fraction: float = 0.5,
    *,
    rng: Optional[_random.Random] = None,
) -> Transcript:
    """Replay U_t from step 0 into other steps.

    The attacker reuses the randomness from step 0 in affected steps,
    then re-executes decoding and rechains.  Detection: PRF derivation
    check (U_t != expected) and collision check.
    """
    if rng is None:
        rng = _random.Random(42)

    tampered = copy.deepcopy(transcript)
    if len(tampered.steps) < 2:
        return tampered

    source_U_t = tampered.steps[0].U_t
    # Affect steps 1..N (never step 0 itself)
    pool = list(range(1, len(tampered.steps)))
    count = max(1, int(len(pool) * fraction))
    affected = set(rng.sample(pool, min(count, len(pool))))

    for step in tampered.steps:
        if step.step_index in affected:
            step.U_t = source_U_t
            res = decode_step(
                K=transcript.policy.K,
                top_k=transcript.policy.top_k,
                top_p_q16=transcript.policy.top_p_q16,
                T_q16=transcript.policy.T_q16,
                token_id=step.token_ids,
                logit_q16=step.logit_q16s,
                U_t=step.U_t,
            )
            step.y = res.y
            step.Ws = res.Ws
            step.R = res.R

    _rechain(tampered)
    return tampered


# ---------------------------------------------------------------------------
# Attack 3 -- Candidate-list manipulation (Section 4.3)
# ---------------------------------------------------------------------------

def attack_candidate_manipulation(
    transcript: Transcript,
    fraction: float = 0.5,
    *,
    rng: Optional[_random.Random] = None,
) -> Transcript:
    """Inject a high-logit token into the candidate set.

    The attacker replaces the last candidate with a synthetic token
    that has a very high logit, biasing the sampling.  The candidate
    hash is recomputed for the modified set but will diverge from
    ground-truth candidates.  Detection: ground-truth candidate
    comparison.
    """
    if rng is None:
        rng = _random.Random(42)

    tampered = copy.deepcopy(transcript)
    affected = _select_affected(len(tampered.steps), fraction, rng)

    injected_token_id = 9_999_999  # synthetic token
    injected_logit = 5 << 16       # large positive logit (Q16.16)

    for step in tampered.steps:
        if step.step_index in affected:
            # Replace the last candidate with the injected one
            step.token_ids[-1] = injected_token_id
            step.logit_q16s[-1] = injected_logit

            # Re-sort and recompute candidate hash
            sorted_tids, sorted_logits = canonical_sort(
                step.token_ids, step.logit_q16s, transcript.policy.T_q16,
            )
            step.token_ids = sorted_tids
            step.logit_q16s = sorted_logits
            step.cand_hash = compute_candidate_hash(sorted_tids, sorted_logits)

            # Re-execute decoding with manipulated candidates
            res = decode_step(
                K=transcript.policy.K,
                top_k=transcript.policy.top_k,
                top_p_q16=transcript.policy.top_p_q16,
                T_q16=transcript.policy.T_q16,
                token_id=step.token_ids,
                logit_q16=step.logit_q16s,
                U_t=step.U_t,
            )
            step.y = res.y
            step.Ws = res.Ws
            step.R = res.R

    _rechain(tampered)
    return tampered


# ---------------------------------------------------------------------------
# Attack 4 -- Transcript truncation / reordering (Section 4.4)
# ---------------------------------------------------------------------------

def attack_transcript_drop(
    transcript: Transcript,
    fraction: float = 0.25,
    *,
    rng: Optional[_random.Random] = None,
) -> Transcript:
    """Drop steps from the transcript.

    Detection: step-index gaps and receipt chain breaks.
    At least one step is always retained (an empty transcript is not
    a realistic attack).
    """
    if rng is None:
        rng = _random.Random(42)

    tampered = copy.deepcopy(transcript)
    if len(tampered.steps) < 2:
        return tampered

    # Cap drop count so at least 1 step survives
    max_drop = len(tampered.steps) - 1
    count = max(1, int(len(tampered.steps) * fraction))
    count = min(count, max_drop)
    drop = set(rng.sample(range(len(tampered.steps)), count))
    tampered.steps = [s for s in tampered.steps if s.step_index not in drop]

    # Attacker does NOT renumber steps (that would be a different attack);
    # the gaps are the evidence of truncation.
    return tampered


def attack_transcript_reorder(
    transcript: Transcript,
    n_swaps: int = 3,
    *,
    rng: Optional[_random.Random] = None,
) -> Transcript:
    """Swap adjacent step pairs in the transcript.

    Detection: step-index ordering violation and receipt chain breaks.
    """
    if rng is None:
        rng = _random.Random(42)

    tampered = copy.deepcopy(transcript)
    if len(tampered.steps) < 2:
        return tampered

    for _ in range(n_swaps):
        i = rng.randint(0, len(tampered.steps) - 2)
        tampered.steps[i], tampered.steps[i + 1] = (
            tampered.steps[i + 1],
            tampered.steps[i],
        )

    # Receipt hashes are left as-is (from honest chain); verifier will
    # detect the chain discontinuity.
    return tampered
