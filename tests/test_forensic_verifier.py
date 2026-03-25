"""Tests for the forensic verification pipeline (ICUFN).

Covers:
  - Honest transcripts pass verification cleanly
  - Each of the four attack classes is detected
  - Attribution reason codes are correct
  - False-positive rate is zero on honest data
"""
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
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
from receipt import PolicyParams, generate_honest_transcript


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

K = 16
N_STEPS = 16
POLICY = PolicyParams(K=K, top_k=4, top_p_q16=int(0.9 * (1 << 16)),
                      T_q16=1 << 16, max_tokens=N_STEPS)
SEED = b"\x01" * 32
REQUEST_ID = b"\xaa" * 32


def _make_candidates(rng: random.Random, K: int, n: int):
    sets = []
    for _ in range(n):
        tids = rng.sample(range(1, 10_000_000), K)
        logits = [rng.randint(-(2 << 16), 2 << 16) for _ in range(K)]
        sets.append((tids, logits))
    return sets


@pytest.fixture
def honest_transcript():
    rng = random.Random(42)
    cands = _make_candidates(rng, K, N_STEPS)
    return generate_honest_transcript(POLICY, SEED, REQUEST_ID, cands)


@pytest.fixture
def ground_truth_cands(honest_transcript):
    return [(s.token_ids[:], s.logit_q16s[:]) for s in honest_transcript.steps]


# ---------------------------------------------------------------------------
# Honest transcript tests
# ---------------------------------------------------------------------------

class TestHonestVerification:

    def test_honest_passes(self, honest_transcript):
        results = verify_transcript(honest_transcript, POLICY, SEED)
        assert len(results) == 1
        assert results[0].code == VerifyCode.PASS

    def test_honest_no_false_positives(self):
        """Run 20 independent honest transcripts; all must pass."""
        rng = random.Random(123)
        for i in range(20):
            seed = rng.randbytes(32)
            req_id = rng.randbytes(32)
            cands = _make_candidates(rng, K, N_STEPS)
            t = generate_honest_transcript(POLICY, seed, req_id, cands)
            results = verify_transcript(t, POLICY, seed)
            assert all(r.code == VerifyCode.PASS for r in results), (
                f"Transcript {i} false positive: {results}"
            )


# ---------------------------------------------------------------------------
# Attack detection tests
# ---------------------------------------------------------------------------

class TestPolicyMismatch:

    def test_detected(self, honest_transcript):
        tampered = attack_policy_mismatch(
            honest_transcript, fraction=1.0,
            new_T_q16=POLICY.T_q16 // 2 or 1,
        )
        results = verify_transcript(tampered, POLICY, SEED)
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.POLICY_MISMATCH in codes

    def test_partial_attack_detected(self, honest_transcript):
        tampered = attack_policy_mismatch(
            honest_transcript, fraction=0.25,
            new_top_p_q16=int(0.5 * (1 << 16)),
            rng=random.Random(7),
        )
        results = verify_transcript(tampered, POLICY, SEED)
        assert any(r.code == VerifyCode.POLICY_MISMATCH for r in results)


class TestRandomnessReplay:

    def test_detected(self, honest_transcript):
        tampered = attack_randomness_replay(
            honest_transcript, fraction=1.0,
        )
        results = verify_transcript(tampered, POLICY, SEED)
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.RANDOMNESS_REPLAY in codes

    def test_partial_detected(self, honest_transcript):
        tampered = attack_randomness_replay(
            honest_transcript, fraction=0.25,
            rng=random.Random(99),
        )
        results = verify_transcript(tampered, POLICY, SEED)
        assert any(r.code == VerifyCode.RANDOMNESS_REPLAY for r in results)


class TestCandidateManipulation:

    def test_detected_with_ground_truth(self, honest_transcript, ground_truth_cands):
        tampered = attack_candidate_manipulation(
            honest_transcript, fraction=1.0,
        )
        results = verify_transcript(
            tampered, POLICY, SEED,
            ground_truth_candidates=ground_truth_cands,
        )
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.CANDIDATE_MANIPULATION in codes

    def test_partial_detected(self, honest_transcript, ground_truth_cands):
        tampered = attack_candidate_manipulation(
            honest_transcript, fraction=0.25,
            rng=random.Random(55),
        )
        results = verify_transcript(
            tampered, POLICY, SEED,
            ground_truth_candidates=ground_truth_cands,
        )
        assert any(r.code == VerifyCode.CANDIDATE_MANIPULATION for r in results)


class TestTranscriptTampering:

    def test_drop_detected(self, honest_transcript):
        tampered = attack_transcript_drop(
            honest_transcript, fraction=0.25,
        )
        results = verify_transcript(tampered, POLICY, SEED)
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.TRANSCRIPT_DISCONTINUITY in codes

    def test_reorder_detected(self, honest_transcript):
        tampered = attack_transcript_reorder(
            honest_transcript, n_swaps=3,
        )
        results = verify_transcript(tampered, POLICY, SEED)
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.TRANSCRIPT_DISCONTINUITY in codes


# ---------------------------------------------------------------------------
# Attribution accuracy tests
# ---------------------------------------------------------------------------

class TestAttribution:
    """Verify that the *primary* reason code matches the attack class."""

    @pytest.mark.parametrize("frac", [0.25, 0.5, 1.0])
    def test_policy_mismatch_attribution(self, honest_transcript, frac):
        tampered = attack_policy_mismatch(
            honest_transcript, fraction=frac,
            new_T_q16=POLICY.T_q16 * 2,
            rng=random.Random(frac),
        )
        results = verify_transcript(tampered, POLICY, SEED)
        assert any(r.code == VerifyCode.POLICY_MISMATCH for r in results)

    @pytest.mark.parametrize("frac", [0.25, 0.5, 1.0])
    def test_randomness_replay_attribution(self, honest_transcript, frac):
        tampered = attack_randomness_replay(
            honest_transcript, fraction=frac,
            rng=random.Random(frac),
        )
        results = verify_transcript(tampered, POLICY, SEED)
        assert any(r.code == VerifyCode.RANDOMNESS_REPLAY for r in results)

    @pytest.mark.parametrize("frac", [0.25, 0.5, 1.0])
    def test_candidate_manipulation_attribution(
        self, honest_transcript, ground_truth_cands, frac,
    ):
        tampered = attack_candidate_manipulation(
            honest_transcript, fraction=frac,
            rng=random.Random(frac),
        )
        results = verify_transcript(
            tampered, POLICY, SEED,
            ground_truth_candidates=ground_truth_cands,
        )
        assert any(r.code == VerifyCode.CANDIDATE_MANIPULATION for r in results)
