#!/usr/bin/env bash
# reproduce_all.sh -- Reproduce all experiments from the ICUFN paper.
#
# Usage:
#   bash reproduce_all.sh          # Full paper-grade runs
#   bash reproduce_all.sh --quick  # Quick sanity check (~2 min)
#
# Requirements:
#   - Python 3.12+
#   - For GPT-2 experiment: torch, transformers (pip install torch transformers)
#   - For figure regeneration: matplotlib, numpy
#
# All outputs are written under eval/.

set -euo pipefail

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
    QUICK="--quick"
    echo "========================================"
    echo "  QUICK MODE (reduced configurations)"
    echo "========================================"
else
    echo "========================================"
    echo "  FULL PAPER-GRADE REPRODUCTION"
    echo "========================================"
fi
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
FAIL=0
SKIP=0

run_experiment() {
    local name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  Experiment: $name"
    echo "================================================================"
    if "$@"; then
        echo "  >> $name: PASSED"
        PASS=$((PASS + 1))
    else
        echo "  >> $name: FAILED (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

# -------------------------------------------------------------------
# 1. Unit tests
# -------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Running unit tests"
echo "================================================================"
if command -v pytest &>/dev/null; then
    if pytest tests/ -v; then
        echo "  >> Unit tests: PASSED"
        PASS=$((PASS + 1))
    else
        echo "  >> Unit tests: FAILED"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  >> pytest not installed, skipping unit tests"
    SKIP=$((SKIP + 1))
fi

# -------------------------------------------------------------------
# 2. Main ICUFN evaluation (Tables I & II, Section 5)
# -------------------------------------------------------------------
run_experiment "ICUFN main evaluation (Tables I & II)" \
    python3 eval/run_icufn.py $QUICK

# -------------------------------------------------------------------
# 3. Latency scaling (Figure 2)
# -------------------------------------------------------------------
run_experiment "Latency scaling (Figure 2)" \
    python3 eval/run_latency_scaling.py $QUICK

# -------------------------------------------------------------------
# 4. Bias heuristic characterization (Figure 3)
# -------------------------------------------------------------------
run_experiment "Bias heuristic (Figure 3)" \
    python3 eval/run_bias_heuristic.py $QUICK

# -------------------------------------------------------------------
# 5. Adaptive adversary analysis (Section 5.3)
# -------------------------------------------------------------------
run_experiment "Adaptive adversary (Section 5.3)" \
    python3 ref/python/adaptive_attacker.py $QUICK

# -------------------------------------------------------------------
# 6. GPT-2 validation (Section 5.4)
# -------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Experiment: GPT-2 validation (Section 5.4)"
echo "================================================================"
if python3 -c "import torch; import transformers" 2>/dev/null; then
    if python3 eval/extract_gpt2_logits.py; then
        echo "  >> GPT-2 validation: PASSED"
        PASS=$((PASS + 1))
    else
        echo "  >> GPT-2 validation: FAILED"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  >> torch/transformers not installed, skipping GPT-2 experiment"
    echo "  >> Install with: pip install torch transformers"
    echo "  >> Pre-computed results available in eval/gpt2_validation_results.json"
    SKIP=$((SKIP + 1))
fi

# -------------------------------------------------------------------
# 7. Regenerate figures (optional)
# -------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Regenerating figures"
echo "================================================================"
if python3 -c "import matplotlib" 2>/dev/null; then
    if python3 paper/icufn/generate_figures.py; then
        echo "  >> Figure generation: PASSED"
        PASS=$((PASS + 1))
    else
        echo "  >> Figure generation: FAILED"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  >> matplotlib not installed, skipping figure generation"
    echo "  >> Pre-generated figures available in paper/icufn/figures/"
    SKIP=$((SKIP + 1))
fi

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  REPRODUCTION SUMMARY"
echo "================================================================"
echo "  Passed:  $PASS"
echo "  Failed:  $FAIL"
echo "  Skipped: $SKIP"
echo ""
echo "  Output files:"
echo "    eval/icufn_results.json           -- Main detection & operational results"
echo "    eval/icufn_detection.csv          -- Per-attack detection rates"
echo "    eval/icufn_operational.csv        -- Per-config latency & storage"
echo "    eval/latency_scaling_results.json -- Latency scaling data (Figure 2)"
echo "    eval/bias_heuristic_results.json  -- Bias heuristic data (Figure 3)"
echo "    eval/adaptive_adversary_results.json -- Adaptive adversary analysis"
echo "    eval/gpt2_validation_results.json -- GPT-2 logit validation"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo "  ** $FAIL experiment(s) FAILED **"
    exit 1
else
    echo "  All experiments completed successfully."
    exit 0
fi
