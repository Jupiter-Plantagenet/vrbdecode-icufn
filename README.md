# Attack-Aware Forensic Receipts for Accountable LLM Decoding Services

**Authors:** George Chidera Akor, Love Allen Chijioke Ahakonye, Jae Min Lee, Dong-Seong Kim

**Affiliation:** IT Convergence Engineering and NSLab Co. Ltd., Kumoh National Institute of Technology, Gumi, South Korea; ICT Convergence Research Center, Kumoh National Institute of Technology

**Venue:** ICUFN 2026

## Description

This repository provides the complete source code and evaluation scripts to reproduce the results in the ICUFN 2025 paper "Attack-Aware Forensic Receipts for Accountable LLM Decoding Services." The paper presents a forensic audit architecture for LLM decoding services that binds policy commitments, per-step receipts, tamper-evident chaining, and deterministic re-execution into chain-of-custody evidence artifacts. The prototype achieves sub-millisecond per-step latency (0.021--0.067 ms/step), compact evidence artifacts (8--307 KB), 0.0% false positives across 10,000 honest transcripts, and 100% detection across five attack classes. A three-baseline comparison (Merkle log signing, policy-commitment verification, watermark detection) demonstrates that re-execution is necessary for reliable semantic attack detection (5/5 vs. 2/5 for the strongest alternative).

## Repository Structure

```
ref/python/              Core implementation
  decoding_ref.py          Fixed-point decoding step (DecodeStep)
  receipt.py               Receipt generation, chaining, and serialization
  forensic_verifier.py     Verification algorithm with reason-coded outcomes
  attack_simulator.py      Five attack implementations
  adaptive_attacker.py     Adaptive adversary with evasion search
  baseline_merkle.py       Baseline 1: Merkle log signing
  baseline_policy_commit.py Baseline 2: Policy-commitment verifier
  baseline_watermark.py    Baseline 3: Kirchenbauer-style watermark detector
  security_analysis.py     Constructive security proofs

eval/                    Evaluation scripts and results
  run_icufn.py             Main evaluation (Tables I & II)
  run_latency_scaling.py   Latency scaling experiment (Figure 2)
  run_bias_heuristic.py    Bias heuristic characterization (Figure 3)
  extract_gpt2_logits.py   GPT-2 logit validation (Section 5.4)
  *.json, *.csv            Pre-computed results

paper/icufn/             Paper source and figures
  main.tex                 LaTeX source
  refs.bib                 Bibliography
  main.pdf                 Compiled paper
  generate_figures.py      Figure generation from result data
  figures/                 Pre-generated figures

tests/                   Unit tests
  test_forensic_verifier.py  Verification pipeline tests
  test_baseline_comparison.py Baseline comparison and security proof tests
```

## Requirements

- **Python 3.12+** (all core experiments use only the standard library)
- **pytest** (for unit tests)
- **torch + transformers** (only for GPT-2 validation experiment)
- **matplotlib + numpy** (only for figure regeneration)

### Install

```bash
# Minimal (core experiments only -- no external packages needed)
pip install pytest

# Full (including GPT-2 experiment and figure generation)
pip install -r requirements.txt
```

## Reproducing Results

### Quick sanity check (~2 minutes)

```bash
bash reproduce_all.sh --quick
```

### Full paper-grade reproduction (~30 minutes without GPT-2, ~60 minutes with)

```bash
bash reproduce_all.sh
```

### Individual experiments

#### 1. Unit tests

```bash
pytest tests/ -v
```

Expected: All tests pass. Verifies correct detection of all five attack classes, baseline limitations (Merkle misses 3/5, policy-commit misses 3/5), and constructive security proofs.

#### 2. Main evaluation -- Tables I and II (Section 5)

```bash
python3 eval/run_icufn.py           # Full: K in {16,32,64}, N in {32,64,128}, 50 transcripts
python3 eval/run_icufn.py --quick   # Quick: K=16, N in {16,32}, 10 transcripts
```

**Output files:** `eval/icufn_results.json`, `eval/icufn_detection.csv`, `eval/icufn_operational.csv`

**Expected results:**
- Detection rate: 100% for all five attack classes across all configurations
- Attribution accuracy: 100% (correct reason code assigned)
- False-positive rate: 0.0% (0/10,000 honest transcripts; 95% Wilson CI [0.000000, 0.000369])
- Baseline comparison: Forensic 5/5, Policy-Commit 2/5, Watermark 0/5, Merkle 2/5
- Verification latency: 0.7--8.6 ms per transcript (sub-ms per step)
- Evidence size: 8--307 KB depending on (K, N) configuration

#### 3. Latency scaling -- Figure 2 (Section 5.2)

```bash
python3 eval/run_latency_scaling.py           # Full: 15 (K,N) configs, 100 runs each
python3 eval/run_latency_scaling.py --quick   # Quick: 6 configs, 30 runs each
```

**Output:** `eval/latency_scaling_results.json`

**Expected:** Linear scaling with N (sequence length). Per-step latency 0.021--0.067 ms/step. Throughput 100--1400 transcripts/sec.

#### 4. Bias heuristic -- Figure 3 (Section 5.3)

```bash
python3 eval/run_bias_heuristic.py           # Full: 1000 FP transcripts, 200 per bias level
python3 eval/run_bias_heuristic.py --quick   # Quick: 100 FP, 50 per level
```

**Output:** `eval/bias_heuristic_results.json`

**Expected:** 0.0% false-positive rate on honest transcripts. Detection power increases monotonically with attacker bias fraction (from ~100% at p=0.1 via PRF mismatch to 100% at all levels).

#### 5. Adaptive adversary (Section 5.3)

```bash
python3 ref/python/adaptive_attacker.py           # Full run
python3 ref/python/adaptive_attacker.py --quick   # Quick run
```

**Output:** `eval/adaptive_adversary_results.json`

**Expected:** 0% output-change evasion rate. Degenerate-case evasion (identical output despite different policy) occurs only when the policy change has no effect on the selected token -- the verifier cannot detect these because there is nothing to detect (the output is correct).

#### 6. GPT-2 validation (Section 5.4)

Requires `torch` and `transformers`:

```bash
pip install torch transformers
python3 eval/extract_gpt2_logits.py
```

**Output:** `eval/gpt2_validation_results.json`

**Expected:** 100/100 honest transcripts pass. 100% detection rate for all attack types across 100 diverse prompts (4 categories x 25). Zero false positives. Sub-millisecond per-step verification latency on real GPT-2 logit distributions.

#### 7. Regenerate figures

```bash
pip install matplotlib numpy
python3 paper/icufn/generate_figures.py
```

Regenerates `paper/icufn/figures/latency_vs_n.{png,pdf}` and `paper/icufn/figures/bias_detection.{png,pdf}` from the result JSON files.

## Pre-Computed Results

All result files are included in `eval/` so that readers can inspect the data without re-running experiments. The `--quick` flag on each script produces directionally identical results with smaller sample sizes for fast verification.

## License

MIT License

Copyright (c) 2025 George Chidera Akor, Love Allen Chijioke Ahakonye, Jae Min Lee, Dong-Seong Kim

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
