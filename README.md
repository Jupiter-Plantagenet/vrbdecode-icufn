# Attack-Aware Forensic Receipts for Accountable Large Language Model Decoding Services

**Authors:** George Chidera Akor, Love Allen Chijioke Ahakonye, Jae Min Lee, Dong-Seong Kim

**Affiliation:** IT Convergence Engineering and NSLab Co. Ltd., Kumoh National Institute of Technology, Gumi, South Korea; ICT Convergence Research Center, Kumoh National Institute of Technology

**Venue:** ICTC 2026

Research artifact accompanying the ICTC 2026 manuscript "Attack-Aware Forensic Receipts for Accountable Large Language Model Decoding Services."

This repository implements a deterministic forensic-auditing framework for LLM decoding services, combining fixed-point decoding, policy commitments, per-step receipts, tamper-evident chaining, forensic re-execution, attack simulation, baseline implementations, and GPT-2 validation.

The artifact is designed to test whether recorded decoding behavior is consistent with a declared decoding policy and recorded randomness. Post-generation transcript integrity additionally depends on an externally authenticated root. Source-side fabrication of candidate lists remains outside the core verifier's trust boundary.

## Status of this repository

This repository preserves the implementation associated with the ICTC 2026 manuscript. Substantive future protocol corrections or extensions should be developed in successor artifacts rather than silently rewriting this experimental record.

The code, tests, experiment scripts, and result files are the versions the manuscript refers to. This README has been revised to describe the artifact's verification model and evaluation scope more precisely; no implementation, test, experiment, or result file was altered in that revision.

## What this artifact demonstrates

- **Python research engineering** — a layered codebase separating the reference implementation (`ref/python/`) from evaluation harnesses (`eval/`) and tests (`tests/`), with the core pipeline written against the standard library only.
- **Deterministic numerical implementation** — an integer-only, fixed-point decoding step (Q16.16 / Q30 arithmetic, a fixed exponential table, canonical candidate ordering) with no floating-point arithmetic anywhere in the decode or receipt path.
- **Adversarial evaluation** — a simulated attack suite over four attack classes, plus a separate adaptive-adversary harness that searches for policy perturbations producing outputs indistinguishable from honest execution.
- **Reproducibility tooling** — a single-entry reproduction script, `--quick` and full modes for each experiment, deterministic seeding, pre-computed results committed alongside the scripts that generate them, and Wilson confidence intervals on headline counts.
- **LLM-security experimentation** — validation on real GPT-2 (117M) top-K logit distributions across 100 prompts in four stylistic categories, with per-category entropy characterization.
- **Forensic verification design** — reason-coded verification outcomes intended for triage, a documented trust boundary, and explicitly stated limitations (seed grinding, source-side fabrication) rather than an unqualified security claim.

## Verification model

The verifier re-executes the recorded decoding computation and compares it against the transcript. It is useful to separate three distinct layers, because they rest on different assumptions:

**1. Local self-consistency (what `verify_transcript()` checks today).**
Given a declared policy and seed, the verifier recomputes the policy commitment, the per-step randomness, the candidate digests, the receipt chain, and the decoding step itself, then compares each against the transcript. This catches a recorded transcript that is internally inconsistent with the declared policy — including a provider that generated under a policy it did not declare, which is the case re-execution addresses and a signed log cannot. Detection here is a property of deterministic recomputation plus hash collision-resistance.

**2. Externally authenticated root (protection against post-generation rewriting).**
`chain_root()` in `ref/python/receipt.py` recomputes the receipt-chain head from step contents. If a provider authenticates that root at generation time — by signing or anchoring it — a downstream relay or store that edits a transcript and re-chains it produces a root that diverges from the authenticated one, so post-generation rewriting is detectable without ground-truth candidate lists. This protection depends on the external authentication step, which is outside this repository. Note that `verify_transcript()` does not currently take an authenticated root as input; the root comparison is exercised in `eval/run_review_upgrades.py` (experiment U2), not inside the verifier's reason-code pipeline. A transcript that has been rewritten and then made fully self-consistent is therefore not distinguishable by local checks alone.

**3. Source-side fabrication (unresolved).**
A receipt generator that fabricates a candidate shortlist at source, runs the declared policy honestly over it, and authenticates the resulting root produces an artifact that is consistent at every layer above. This is outside the core verifier's trust boundary and is not addressed by this implementation. The manuscript discusses client co-signing, an attested enclave, and a verifiable-forward-pass anchor as directions for closing it.

Separately, a provider that controls the randomness seed can search over seeds to steer an output, and the resulting transcript verifies as clean. This limitation is measured directly by experiment U5 in `eval/run_review_upgrades.py` and motivates VRF- or beacon-supplied randomness.

### Terminology

Per-step randomness `U_t` is produced by `derive_U_t()` as a **deterministic domain-separated hash derivation** over `(request_id, policy_hash, seed_commit, step_index)`. It is a randomness-binding derivation: it binds each step's randomness to the request, the policy commitment, the seed commitment, and the step index, so a verifier can recompute it exactly. All of its inputs are public within the evidence artifact, so it should not be read as providing unpredictability, and this repository does not claim pseudorandom-function security for it.

`ref/python/security_analysis.py` contains **security analyses**, not machine-checked proofs: for each attack class it states a proposition, runs an **executable attack check** on a concrete instance and records which reason codes fire, and supplies a **worked reduction argument** in prose. These are structured arguments and worked examples intended to support the manuscript's security discussion; they are not formal proofs, and they do not quantify over all inputs or adversaries.

## Repository Structure

```
ref/python/              Core implementation
  decoding_ref.py          Fixed-point decoding step (DecodeStep)
  receipt.py               Receipt generation, chaining, chain_root()
  forensic_verifier.py     Verification with reason codes
  attack_simulator.py      Simulated attacks (four classes)
  adaptive_attacker.py     Adaptive adversary with evasion search
  baseline_merkle.py       Baseline 1: Merkle log signing
  baseline_policy_commit.py Baseline 2: Policy-commitment verifier
  baseline_watermark.py    Baseline 3: Kirchenbauer-style watermark detector
  security_analysis.py     Security analyses: executable attack checks
                           and worked reduction arguments

eval/                    Evaluation scripts and pre-computed results
  run_ictc.py              Main detection + operational evaluation
  run_review_upgrades.py   Reason-code coverage, authenticated-root
                           ablation, determinism, inline overhead,
                           seed-grinding limitation, Wilson CIs
  run_latency_scaling.py   Latency scaling vs. (K, N)
  run_bias_heuristic.py    Supplementary bias-heuristic characterization
  extract_gpt2_logits.py   GPT-2 logit validation
  *.json, *.csv            Pre-computed results

tests/                   Unit tests (50 tests)
  test_forensic_verifier.py   Verification pipeline tests
  test_baseline_comparison.py Baseline comparison and security-analysis tests
```

The manuscript source (LaTeX) is maintained separately; this repository is the code artifact and is referenced from the manuscript.

## Requirements

- **Python 3.12+** (the core pipeline and verifier use only the standard library)
- **pytest** (for unit tests)
- **torch + transformers + numpy** (only for the GPT-2 validation experiment)

### Install

```bash
# Minimal (core experiments only -- no external packages needed)
pip install pytest

# Full (including the GPT-2 validation experiment)
pip install -r requirements.txt
```

## Reproducing Results

### Portability note

`reproduce_all.sh` invokes the interpreter as `python3`. On systems where `python3` is not on `PATH` — notably Windows, where it may resolve to a non-functional stub — either expose a `python3` entry point on `PATH` or run the experiments individually with whichever interpreter name is available, for example `python eval/run_ictc.py --quick`. The individual commands below are equivalent to the stages the script runs.

### Quick sanity check

```bash
bash reproduce_all.sh --quick
```

Runs the unit tests plus reduced configurations of each experiment. Note that `--quick` is not propagated to `eval/run_review_upgrades.py` or `eval/extract_gpt2_logits.py`; if `torch` and `transformers` are installed, the GPT-2 stage runs at full size and dominates the runtime. Without those packages the GPT-2 stage is skipped and the quick path completes in roughly a minute.

### Full reproduction

```bash
bash reproduce_all.sh
```

The non-GPT-2 stages complete in a few minutes on a modern desktop; GPT-2 logit extraction adds several minutes on CPU and requires a one-time model download.

### Individual experiments

#### 1. Unit tests

```bash
pytest tests/ -v
```

50 tests covering honest-transcript verification, detection of each simulated attack class, reason codes, Merkle-tree utilities, the documented limitations of the Merkle baseline, and the security-analysis entry points.

#### 2. Main detection and operational evaluation

```bash
python3 eval/run_ictc.py           # Full: K in {16,32,64}, N in {32,64,128}
python3 eval/run_ictc.py --quick   # Quick: K=16, N in {16,32}
```

**Output:** `eval/ictc_results.json`, `eval/ictc_detection.csv`, `eval/ictc_operational.csv`

**Recorded results:** per-attack, per-intensity detection and reason-code rates for the simulated attack suite are written to `eval/ictc_detection.csv`; readers should consult that file rather than a single summary figure. Detection is complete for every simulated configuration except transcript-drop at intensity 1.0, where the attack retains a single step and the resulting artifact can be a well-formed short transcript (`detection_rate` 0.9778, `fn_rate` 0.0222 in the committed CSV). The false-positive measurement, operational latency, and evidence-artifact sizes are recorded in `eval/ictc_results.json` and `eval/ictc_operational.csv`.

#### 3. Reviewer-evidence experiments

```bash
python3 eval/run_review_upgrades.py
```

**Output:** `eval/review_upgrades_results.json`

Six experiments: reason-code coverage across the tested attack classes (U1); the authenticated-root ablation, comparing detection of downstream candidate rewriting with and without a generation-time root (U2); cross-process determinism of the receipt root (U3); inline receipt-generation overhead (U4); the seed-grinding limitation (U5); and Wilson 95% confidence intervals for the headline counts (U6).

U1 reports **reason-code recall**: for each simulated attack class, whether the expected reason code appears among the codes raised. The same file also records a `code_cofire` distribution showing that codes are not one-to-one with attack classes — several attacks raise more than one code, and transcript-drop and transcript-reorder share a single code. U1 should not be read as one-to-one causal attribution.

U6 derives its counts from the committed result files rather than from the current run, so its confidence intervals reflect whichever result files are present when it executes.

#### 4. Latency scaling

```bash
python3 eval/run_latency_scaling.py           # Full: 15 (K,N) configs
python3 eval/run_latency_scaling.py --quick   # Quick: 6 configs
```

**Output:** `eval/latency_scaling_results.json` — per-configuration verification latency (5 warmup runs discarded, remainder measured), evidence-artifact size, and throughput. Latency grows approximately linearly in the number of steps N. Each configuration is measured by repeatedly verifying one generated transcript, so the reported dispersion reflects timing variation rather than variation across transcripts.

#### 5. Adaptive adversary

```bash
python3 ref/python/adaptive_attacker.py           # Full run
python3 ref/python/adaptive_attacker.py --quick   # Quick run
```

**Output:** `eval/adaptive_adversary_results.json`

Searches for policy perturbations `P' != P` that yield outputs identical to honest execution, across four candidate-entropy regimes. The substantive recorded finding is that such perturbations are common: `policy_evasion_fraction_mean` is high at every entropy level, meaning re-execution constrains the *output* rather than pinning the exact policy parameters. The perturbation grid mixes very small and very large parameter changes, so this fraction should be read alongside the per-perturbation rates in `results_by_entropy`. The companion `output_evasion` counter is defined as an output change with identical `(Ws, R)`, a condition the sampling logic cannot produce, so it is not a meaningful evasion measurement.

#### 6. Bias-heuristic characterization (supplementary)

```bash
python3 eval/run_bias_heuristic.py           # Full: 1000 FP transcripts, 200 per bias level
python3 eval/run_bias_heuristic.py --quick   # Quick: 100 FP, 50 per level
```

**Output:** `eval/bias_heuristic_results.json`

Characterizes the optional `RANDOMNESS_BIAS` heuristic in `forensic_verifier.py`. The heuristic is a fixed output-frequency threshold, not a distributional test: it fires when one token accounts for at least half the steps, and the recorded detection curve is correspondingly a step function at that threshold. In this experiment the biased transcripts are constructed by overriding `U_t`, so they are additionally caught by the randomness-binding check regardless of the heuristic.

#### 7. GPT-2 validation

Requires `torch` and `transformers`:

```bash
pip install torch transformers
python3 eval/extract_gpt2_logits.py
```

**Output:** `eval/gpt2_validation_results.json` — verification against real GPT-2 top-K logit distributions over 100 prompts (4 categories × 25), with per-category entropy statistics, per-attack detection rates and Wilson intervals, verification latency, and evidence-artifact sizes. All 100 honest transcripts verify cleanly. Candidate-list rewriting is detected in every case when a ground-truth shortlist is available and in 53/100 cases without one, which marks the trust boundary described in the verification model above. Context is advanced by greedy selection during extraction, so the recorded candidate trajectories condition on a greedy continuation rather than on the sampled tokens.

## Evaluation scope

The reported numbers characterize this implementation under the specific generators and adversaries implemented here. Three scope boundaries are worth stating explicitly:

- **False positives.** The published synthetic evaluation reports 0/10,000 false positives under its tested honest-generator distribution (95% Wilson CI [0.0, 0.038]%). That generator draws fresh random token identifiers at every step, so honest transcripts effectively never repeat a token. This is not a universal property of the verifier: the optional output-frequency heuristic can flag honest transcripts whose distribution is strongly peaked and whose context repeats.
- **Simulated adversaries.** The attack suite in `ref/python/attack_simulator.py` models specific adversaries, documented in that file. In particular the transcript-drop attack deliberately leaves step indices un-renumbered and the reorder attack leaves receipt hashes stale, so that the structural evidence of tampering is present. Results for these attacks characterize those adversaries, not the strongest adversary the threat model admits — see layer 2 of the verification model for why post-generation rewriting ultimately depends on an externally authenticated root.
- **Baselines.** The repository includes Merkle-log, policy-commitment, and watermark baselines for comparative experiments, in `ref/python/baseline_merkle.py`, `baseline_policy_commit.py`, and `baseline_watermark.py`. Their behavior is sensitive to what each verifier is assumed to hold at verification time — in particular whether it is given a generation-time root to compare against — so the comparative counts should be read together with the protocol each baseline is evaluated under, as implemented in `eval/run_ictc.py` and documented in the baseline modules.

## Reproducibility

All result files in `eval/` are committed so readers can inspect the data without re-running anything, and each is produced by the script named in the section above. Experiment seeding is deterministic, so the security-facing outputs are stable across runs; timing-dependent fields naturally vary with hardware.

The `--quick` flag on each script that supports it produces directionally similar results with smaller sample sizes for fast verification.

Determinism is measured directly rather than asserted: experiment U3 in `eval/run_review_upgrades.py` generates the same transcript in 12 independent OS processes under varied `PYTHONHASHSEED` values, plus 1000 in-process repeats, and checks that every receipt root is bit-identical. This follows from the decode and receipt paths being integer-only.

> **Independent-audit observation (not a manuscript result).** An independent reproducibility audit of commit `526ab28` re-ran the test suite and every experiment on different hardware and a different operating system from the one that produced the committed results. All 50 tests passed, and the committed result files reproduced with all non-timing content identical, including the U3 receipt root reproducing bit-for-bit. The same audit is the source of several of the scope statements above.

## License

MIT License

Copyright (c) 2026 George Chidera Akor, Love Allen Chijioke Ahakonye, Jae Min Lee, Dong-Seong Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
