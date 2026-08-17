# Attack-Aware Forensic Receipts for Accountable Large Language Model Decoding Services

**Authors:** George Chidera Akor, Love Allen Chijioke Ahakonye, Jae Min Lee, Dong-Seong Kim

**Affiliation:** IT Convergence Engineering and NSLab Co. Ltd., Kumoh National Institute of Technology, Gumi, South Korea; ICT Convergence Research Center, Kumoh National Institute of Technology

**Venue:** ICTC 2026

Research artifact accompanying the ICTC 2026 manuscript of the same title.

This repository implements a deterministic forensic-auditing framework for LLM decoding services: fixed-point decoding, policy commitments, per-step receipts, tamper-evident chaining, forensic re-execution, attack simulation, three comparison baselines, and GPT-2 validation.

The artifact tests whether recorded decoding behavior is consistent with a declared decoding policy and recorded randomness. Post-generation transcript integrity additionally depends on an externally authenticated root. Source-side fabrication of candidate lists remains outside the core verifier's trust boundary.

## Status of this repository

This repository preserves the implementation associated with the ICTC 2026 manuscript. Substantive future protocol corrections or extensions should be developed in successor artifacts rather than silently rewriting this experimental record.

This README describes the artifact's verification model and evaluation scope; no implementation, test, experiment, or result file was altered in revising it.

## What this artifact demonstrates

- **Python research engineering** — layered separation of reference implementation, evaluation harnesses, and tests; core pipeline is standard-library only.
- **Deterministic numerical implementation** — integer-only fixed-point decoding (Q16.16 / Q30, fixed exponential table, canonical ordering); no floating point in the decode or receipt path.
- **Adversarial evaluation** — a simulated attack suite over four attack classes, plus an adaptive-adversary search for policy perturbations that leave the output unchanged.
- **Reproducibility tooling** — one-command reproduction, `--quick` and full modes, deterministic seeding, results committed beside their generating scripts.
- **LLM-security experimentation** — validation on real GPT-2 (117M) top-K logits over 100 prompts in four stylistic categories, with entropy characterization.
- **Forensic verification design** — reason-coded outcomes for triage, a documented trust boundary, and stated limitations rather than an unqualified security claim.

## Verification model

Three layers, resting on different assumptions.

**1. Local self-consistency — what `verify_transcript()` checks.** Given a declared policy and seed, the verifier recomputes the policy commitment, per-step randomness, candidate digests, receipt chain, and the decoding step itself, and compares each against the transcript. This catches a transcript inconsistent with the declared policy — including a provider that generated under a policy it did not declare, the case re-execution addresses and a signed log cannot. It rests on deterministic recomputation plus hash collision-resistance.

**2. Externally authenticated root — post-generation rewriting.** `chain_root()` (`ref/python/receipt.py`) recomputes the chain head from step contents. If a provider authenticates that root at generation time, a downstream relay that edits a transcript and re-chains it produces a diverging root, so rewriting is detectable without ground-truth candidate lists. This depends on the external authentication step, which is outside this repository. `verify_transcript()` does not currently take an authenticated root as input — the comparison is exercised in `eval/run_review_upgrades.py` (U2), not in the verifier's reason-code pipeline. A rewritten transcript made fully self-consistent is therefore not distinguishable by the local checks alone.

**3. Source-side fabrication — unresolved.** A generator that fabricates a candidate shortlist at source, runs the declared policy honestly over it, and authenticates the resulting root produces an artifact consistent at every layer above. This is outside the core verifier's trust boundary; the manuscript discusses client co-signing, an attested enclave, and a verifiable-forward-pass anchor as directions.

Separately, a provider controlling the randomness seed can search over seeds to steer an output, and the resulting transcript verifies as clean — measured directly by experiment U5.

### Terminology

Per-step randomness `U_t` comes from `derive_U_t()`, a **deterministic domain-separated hash derivation** over `(request_id, policy_hash, seed_commit, step_index)`, so a verifier can recompute it exactly. All of its inputs are public within the evidence artifact, so it provides no unpredictability, and this repository does not claim pseudorandom-function security for it.

`ref/python/security_analysis.py` contains **security analyses**, not formal proofs: per attack class it states a proposition, runs an **executable attack check** on a concrete instance, and gives a **worked reduction argument** in prose. These are not machine-checked and do not quantify over all inputs or adversaries.

## Repository structure

```
ref/python/     Core implementation
  decoding_ref.py        Fixed-point decoding step (DecodeStep)
  receipt.py             Receipts, chaining, chain_root()
  forensic_verifier.py   Verification with reason codes
  attack_simulator.py    Simulated attacks (four classes)
  adaptive_attacker.py   Adaptive adversary with evasion search
  baseline_merkle.py         Baseline 1: Merkle log signing
  baseline_policy_commit.py  Baseline 2: policy-commitment verifier
  baseline_watermark.py      Baseline 3: Kirchenbauer-style watermark
  security_analysis.py   Security analyses (see Terminology)

eval/           Experiment scripts and committed results (*.json, *.csv)
tests/          50 unit tests
```

The manuscript source (LaTeX) is maintained separately.

## Requirements

**Python 3.12+** (core pipeline and verifier: standard library only) · **pytest** for tests · **torch + transformers + numpy** only for the GPT-2 experiment.

```bash
pip install pytest               # core experiments only
pip install -r requirements.txt  # including GPT-2 validation
```

## Reproducing results

```bash
pytest tests/ -v                # 50 unit tests
bash reproduce_all.sh --quick   # reduced configurations
bash reproduce_all.sh           # full
```

**Portability.** `reproduce_all.sh` invokes `python3`. Where that is not on `PATH` — notably Windows, where it may resolve to a non-functional stub — expose a `python3` entry point or run scripts individually with the available interpreter, e.g. `python eval/run_ictc.py --quick`. `--quick` is not propagated to `run_review_upgrades.py` or `extract_gpt2_logits.py`; with `torch` installed the GPT-2 stage runs at full size and dominates runtime.

### Experiments

Each accepts `--quick` unless noted, and writes to `eval/`.

| Script | Output | What it measures, and how to read it |
|---|---|---|
| `run_ictc.py` | `ictc_results.json`, `ictc_detection.csv`, `ictc_operational.csv` | Per-attack, per-intensity detection and reason-code rates for the simulated attack suite; the false-positive measurement; verification latency and evidence size. Read `ictc_detection.csv` rather than any single summary figure — the committed evaluation contains non-zero false negatives for some transcript-drop settings. |
| `run_review_upgrades.py` *(no `--quick`)* | `review_upgrades_results.json` | Reason-code coverage (U1), authenticated-root ablation (U2), cross-process determinism (U3), receipt overhead (U4), the seed-grinding limitation (U5), Wilson 95% CIs (U6). U1 reports **reason-code recall** — whether the expected code appears among those raised. Its `code_cofire` distribution shows codes are not one-to-one with attack classes: several attacks raise more than one, and transcript-drop and transcript-reorder share a code. U1 is coverage, not one-to-one causal attribution. |
| `run_latency_scaling.py` | `latency_scaling_results.json` | Verification latency, evidence size and throughput across a (K, N) grid; latency grows approximately linearly in N. Each configuration repeatedly verifies one transcript, so dispersion reflects timing variation, not variation across transcripts. |
| `../ref/python/adaptive_attacker.py` | `adaptive_adversary_results.json` | Searches for policy perturbations `P' != P` yielding outputs identical to honest execution. The substantive finding: such perturbations are common at every candidate-entropy level, so re-execution constrains the *output* rather than pinning exact policy parameters. The `output_evasion` counter is not used as a headline security result in this README. |
| `run_bias_heuristic.py` | `bias_heuristic_results.json` | Characterizes the optional `RANDOMNESS_BIAS` heuristic, a fixed output-frequency threshold rather than a distributional test. Its biased transcripts are built by overriding `U_t`, so they are additionally caught by the randomness-binding check. |
| `extract_gpt2_logits.py` *(needs torch; no `--quick`)* | `gpt2_validation_results.json` | Verification against real GPT-2 top-K logits: 100 prompts (4 × 25), entropy statistics, per-attack detection with Wilson intervals, latency, evidence size. All 100 honest transcripts verify cleanly. Candidate rewriting is detected in every case with a ground-truth shortlist and in 53/100 without one, marking the trust boundary above. Context is advanced greedily during extraction, so trajectories condition on a greedy continuation rather than the sampled tokens. |

## Evaluation scope

Reported numbers characterize this implementation under the generators and adversaries implemented here.

- **False positives.** The published synthetic evaluation reports 0/10,000 false positives under its tested honest-generator distribution (95% Wilson CI [0.0, 0.038]%). That generator draws fresh random token identifiers every step, so honest transcripts effectively never repeat a token. This is not a universal property of the verifier: the optional frequency heuristic can flag honest transcripts whose output distribution is strongly peaked and whose context repeats.
- **Simulated adversaries.** `attack_simulator.py` models specific adversaries, documented in that file: the drop attack leaves step indices un-renumbered and the reorder attack leaves receipt hashes stale, so structural evidence of tampering is present. Results characterize those adversaries, not the strongest adversary the threat model admits — see layer 2 for why post-generation rewriting ultimately depends on an externally authenticated root.
- **Baselines.** Merkle-log, policy-commitment, and watermark baselines are included for comparative experiments. Their behavior is sensitive to what each verifier is assumed to hold at verification time — in particular whether it is given a generation-time root — so comparative counts should be read together with the protocol implemented in `run_ictc.py`.

## Reproducibility

All result files in `eval/` are committed so readers can inspect the data without re-running anything, each produced by the script named above. Seeding is deterministic, so security-facing outputs are stable across runs; timing fields vary with hardware.

Determinism is measured rather than asserted: U3 generates the same transcript in 12 independent OS processes under varied `PYTHONHASHSEED`, plus 1000 in-process repeats, and checks every receipt root is bit-identical. This follows from the decode and receipt paths being integer-only.

> **Independent-audit observation (not a manuscript result).** An independent reproducibility audit of commit `526ab28` re-ran the test suite and every experiment on different hardware and a different operating system. All 50 tests passed, and the committed result files reproduced with all non-timing content identical, including the U3 receipt root reproducing bit-for-bit.

## License

MIT — see [`LICENSE`](LICENSE) for the full text.

Copyright (c) 2026 George Chidera Akor, Love Allen Chijioke Ahakonye, Jae Min Lee, Dong-Seong Kim
