"""Tests for the Merkle baseline and comparison with forensic verifier.

Demonstrates:
  - Merkle baseline passes honest transcripts
  - Merkle baseline detects transcript tampering (step drop / reorder)
  - Merkle baseline FAILS to detect policy mismatch
  - Merkle baseline FAILS to detect randomness replay
  - Merkle baseline FAILS to detect candidate manipulation
  - Forensic verifier detects ALL attack types
  - Security analysis proofs all produce valid results
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
from baseline_merkle import (
    BaselineCode,
    ComparisonRow,
    MerkleBaseline,
    compare_approaches,
    compute_merkle_root,
    format_comparison_table,
)
from forensic_verifier import VerifyCode, verify_transcript
from receipt import PolicyParams, generate_honest_transcript
from security_analysis import (
    SecurityProofResult,
    format_proof_report,
    prove_candidate_manipulation_detection,
    prove_policy_mismatch_detection,
    prove_randomness_replay_detection,
    prove_transcript_tampering_detection,
    run_all_proofs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

K = 16
N_STEPS = 8
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


@pytest.fixture
def baseline():
    return MerkleBaseline()


# ---------------------------------------------------------------------------
# Merkle tree utility tests
# ---------------------------------------------------------------------------

class TestMerkleTree:

    def test_single_leaf(self):
        root = compute_merkle_root([b"hello"])
        assert len(root) == 32
        assert root != b"\x00" * 32

    def test_two_leaves(self):
        root = compute_merkle_root([b"hello", b"world"])
        assert len(root) == 32

    def test_empty(self):
        root = compute_merkle_root([])
        assert root == b"\x00" * 32

    def test_deterministic(self):
        leaves = [b"a", b"b", b"c", b"d"]
        root1 = compute_merkle_root(leaves)
        root2 = compute_merkle_root(leaves)
        assert root1 == root2

    def test_order_sensitive(self):
        root1 = compute_merkle_root([b"a", b"b"])
        root2 = compute_merkle_root([b"b", b"a"])
        assert root1 != root2

    def test_odd_number_of_leaves(self):
        """Odd leaf count should still produce a valid root."""
        root = compute_merkle_root([b"a", b"b", b"c"])
        assert len(root) == 32


# ---------------------------------------------------------------------------
# Merkle baseline: honest transcript
# ---------------------------------------------------------------------------

class TestBaselineHonest:

    def test_honest_passes(self, honest_transcript, baseline):
        """An honestly-generated transcript must pass baseline verification."""
        signed = baseline.sign_transcript(honest_transcript)
        results = baseline.verify_transcript(signed)
        assert len(results) == 1
        assert results[0].code == BaselineCode.PASS

    def test_honest_with_merkle_root_check(self, honest_transcript, baseline):
        """Honest transcript with matching expected Merkle root passes."""
        signed = baseline.sign_transcript(honest_transcript)
        results = baseline.verify_transcript(
            signed, expected_merkle_root=signed.merkle_root,
        )
        assert len(results) == 1
        assert results[0].code == BaselineCode.PASS

    def test_wrong_merkle_root_detected(self, honest_transcript, baseline):
        """Providing an incorrect expected Merkle root is detected."""
        signed = baseline.sign_transcript(honest_transcript)
        results = baseline.verify_transcript(
            signed, expected_merkle_root=b"\xff" * 32,
        )
        codes = [r.code for r in results]
        assert BaselineCode.MERKLE_ROOT_MISMATCH in codes


# ---------------------------------------------------------------------------
# Merkle baseline: detects transcript tampering
# ---------------------------------------------------------------------------

class TestBaselineDetectsTranscriptTampering:

    def test_drop_detected(self, honest_transcript, baseline):
        """Merkle baseline detects dropped steps via step-index gap."""
        tampered = attack_transcript_drop(
            honest_transcript, fraction=0.25,
            rng=random.Random(1),
        )
        signed = baseline.sign_transcript(tampered)
        results = baseline.verify_transcript(signed)
        codes = [r.code for r in results]
        assert BaselineCode.PASS not in codes
        assert BaselineCode.STEP_INDEX_GAP in codes

    def test_reorder_detected(self, honest_transcript, baseline):
        """Merkle baseline detects reordered steps via step-index check."""
        tampered = attack_transcript_reorder(
            honest_transcript, n_swaps=2,
            rng=random.Random(0),
        )
        signed = baseline.sign_transcript(tampered)
        results = baseline.verify_transcript(signed)
        codes = [r.code for r in results]
        assert BaselineCode.PASS not in codes
        assert BaselineCode.STEP_INDEX_GAP in codes

    def test_signature_tampering_detected(self, honest_transcript, baseline):
        """Post-signing modification of step data is detected."""
        signed = baseline.sign_transcript(honest_transcript)
        # Tamper with step data after signing
        if signed.step_data:
            signed.step_data[0] = b"\x00" * len(signed.step_data[0])
        results = baseline.verify_transcript(signed)
        codes = [r.code for r in results]
        assert BaselineCode.SIGNATURE_INVALID in codes


# ---------------------------------------------------------------------------
# Merkle baseline: FAILS to detect semantic attacks (key finding)
# ---------------------------------------------------------------------------

class TestBaselineFailsSemanticAttacks:

    def test_fails_policy_mismatch(self, honest_transcript, baseline):
        """KEY FINDING: Baseline cannot detect policy mismatch.

        The attacker re-executes decoding with altered parameters and
        signs the resulting (different) transcript. The baseline sees
        valid signatures over the tampered data and passes it.
        """
        tampered = attack_policy_mismatch(
            honest_transcript, fraction=1.0,
            new_T_q16=max(1, POLICY.T_q16 // 2),
            rng=random.Random(0),
        )
        signed = baseline.sign_transcript(tampered)
        results = baseline.verify_transcript(signed)
        # Baseline should PASS — it cannot see the policy violation
        assert any(r.code == BaselineCode.PASS for r in results), (
            f"Expected baseline to PASS (miss policy mismatch), got: "
            f"{[r.code for r in results]}"
        )

    def test_fails_randomness_replay(self, honest_transcript, baseline):
        """KEY FINDING: Baseline cannot detect randomness replay.

        The attacker replays U_t values and re-signs. The baseline
        cannot distinguish replayed randomness from legitimate values.
        """
        tampered = attack_randomness_replay(
            honest_transcript, fraction=1.0,
            rng=random.Random(0),
        )
        signed = baseline.sign_transcript(tampered)
        results = baseline.verify_transcript(signed)
        assert any(r.code == BaselineCode.PASS for r in results), (
            f"Expected baseline to PASS (miss randomness replay), got: "
            f"{[r.code for r in results]}"
        )

    def test_fails_candidate_manipulation(self, honest_transcript, baseline):
        """KEY FINDING: Baseline cannot detect candidate manipulation.

        The attacker modifies candidates, re-executes decoding, and
        re-signs. The baseline signs whatever data it receives.
        """
        tampered = attack_candidate_manipulation(
            honest_transcript, fraction=1.0,
            rng=random.Random(0),
        )
        signed = baseline.sign_transcript(tampered)
        results = baseline.verify_transcript(signed)
        assert any(r.code == BaselineCode.PASS for r in results), (
            f"Expected baseline to PASS (miss candidate manipulation), got: "
            f"{[r.code for r in results]}"
        )


# ---------------------------------------------------------------------------
# Forensic verifier: detects ALL attack types (contrast)
# ---------------------------------------------------------------------------

class TestForensicDetectsAll:

    def test_detects_policy_mismatch(self, honest_transcript):
        tampered = attack_policy_mismatch(
            honest_transcript, fraction=1.0,
            new_T_q16=max(1, POLICY.T_q16 // 2),
            rng=random.Random(0),
        )
        results = verify_transcript(tampered, POLICY, SEED)
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.POLICY_MISMATCH in codes

    def test_detects_randomness_replay(self, honest_transcript):
        tampered = attack_randomness_replay(
            honest_transcript, fraction=1.0,
            rng=random.Random(0),
        )
        results = verify_transcript(tampered, POLICY, SEED)
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.RANDOMNESS_REPLAY in codes

    def test_detects_candidate_manipulation(self, honest_transcript, ground_truth_cands):
        tampered = attack_candidate_manipulation(
            honest_transcript, fraction=1.0,
            rng=random.Random(0),
        )
        results = verify_transcript(
            tampered, POLICY, SEED,
            ground_truth_candidates=ground_truth_cands,
        )
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.CANDIDATE_MANIPULATION in codes

    def test_detects_transcript_drop(self, honest_transcript):
        tampered = attack_transcript_drop(
            honest_transcript, fraction=0.25,
            rng=random.Random(1),
        )
        results = verify_transcript(tampered, POLICY, SEED)
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.TRANSCRIPT_DISCONTINUITY in codes

    def test_detects_transcript_reorder(self, honest_transcript):
        tampered = attack_transcript_reorder(
            honest_transcript, n_swaps=2,
            rng=random.Random(0),
        )
        results = verify_transcript(tampered, POLICY, SEED)
        codes = {r.code for r in results}
        assert VerifyCode.PASS not in codes
        assert VerifyCode.TRANSCRIPT_DISCONTINUITY in codes


# ---------------------------------------------------------------------------
# Comparison function tests
# ---------------------------------------------------------------------------

class TestComparisonFunction:

    def test_comparison_runs(self, honest_transcript, ground_truth_cands):
        """compare_approaches produces a valid table."""
        rows = compare_approaches(
            honest_transcript, SEED, POLICY,
            ground_truth_candidates=ground_truth_cands,
        )
        assert len(rows) == 5  # 5 attack types
        for row in rows:
            assert isinstance(row, ComparisonRow)
            assert isinstance(row.forensic_detected, bool)
            assert isinstance(row.baseline_detected, bool)

    def test_forensic_detects_all(self, honest_transcript, ground_truth_cands):
        """Forensic verifier detects every attack in the comparison."""
        rows = compare_approaches(
            honest_transcript, SEED, POLICY,
            ground_truth_candidates=ground_truth_cands,
        )
        for row in rows:
            assert row.forensic_detected, (
                f"Forensic verifier missed {row.attack_class}"
            )

    def test_baseline_misses_semantic(self, honest_transcript, ground_truth_cands):
        """Baseline misses policy mismatch, randomness replay, and candidate manipulation."""
        rows = compare_approaches(
            honest_transcript, SEED, POLICY,
            ground_truth_candidates=ground_truth_cands,
        )
        row_map = {r.attack_class: r for r in rows}
        assert not row_map["policy_mismatch"].baseline_detected
        assert not row_map["randomness_replay"].baseline_detected
        assert not row_map["candidate_manipulation"].baseline_detected

    def test_baseline_detects_structural(self, honest_transcript, ground_truth_cands):
        """Baseline detects structural attacks (drop and reorder)."""
        rows = compare_approaches(
            honest_transcript, SEED, POLICY,
            ground_truth_candidates=ground_truth_cands,
        )
        row_map = {r.attack_class: r for r in rows}
        assert row_map["transcript_drop"].baseline_detected
        assert row_map["transcript_reorder"].baseline_detected

    def test_format_comparison_table(self):
        """format_comparison_table produces non-empty output."""
        table = format_comparison_table()
        assert len(table) > 100
        assert "Forensic" in table
        assert "Baseline" in table
        assert "policy_mismatch" in table


# ---------------------------------------------------------------------------
# Security analysis proof tests
# ---------------------------------------------------------------------------

class TestSecurityAnalysis:

    def test_policy_mismatch_proof(self):
        result = prove_policy_mismatch_detection()
        assert isinstance(result, SecurityProofResult)
        assert result.attack_class == "policy_mismatch"
        assert len(result.concrete_example.detection_codes) > 0
        assert "policy_mismatch" in result.concrete_example.detection_codes
        assert "determinism" in result.security_reduction.lower() or \
               "deterministic" in result.security_reduction.lower()

    def test_randomness_replay_proof(self):
        result = prove_randomness_replay_detection()
        assert isinstance(result, SecurityProofResult)
        assert result.attack_class == "randomness_replay"
        assert len(result.concrete_example.detection_codes) > 0
        assert "randomness_replay" in result.concrete_example.detection_codes
        assert "sha-256" in result.security_reduction.lower() or \
               "sha256" in result.security_reduction.lower() or \
               "collision" in result.security_reduction.lower()

    def test_candidate_manipulation_proof(self):
        result = prove_candidate_manipulation_detection()
        assert isinstance(result, SecurityProofResult)
        assert result.attack_class == "candidate_manipulation"
        assert len(result.concrete_example.detection_codes) > 0
        assert "candidate_manipulation" in result.concrete_example.detection_codes
        assert "collision" in result.security_reduction.lower()

    def test_transcript_tampering_proof(self):
        result = prove_transcript_tampering_detection()
        assert isinstance(result, SecurityProofResult)
        assert result.attack_class == "transcript_tampering"
        assert len(result.concrete_example.detection_codes) > 0
        assert "transcript_discontinuity" in result.concrete_example.detection_codes

    def test_run_all_proofs(self):
        proofs = run_all_proofs()
        assert len(proofs) == 4
        classes = {p.attack_class for p in proofs}
        assert classes == {
            "policy_mismatch",
            "randomness_replay",
            "candidate_manipulation",
            "transcript_tampering",
        }
        # Every proof must have detected something
        for proof in proofs:
            assert len(proof.concrete_example.detection_codes) > 0, (
                f"Proof for {proof.attack_class} has no detection codes"
            )

    def test_format_proof_report(self):
        report = format_proof_report()
        assert len(report) > 200
        assert "Proposition 1" in report
        assert "Proposition 4" in report
        assert "Summary Table" in report
        assert "YES" in report
