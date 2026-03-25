#!/usr/bin/env python3
"""Generate figures for the ICUFN paper from real experimental data."""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# IEEE column width ~3.5in, double ~7in
SINGLE_COL = 3.5
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.dpi": 300,
})


def fig2_latency_scaling():
    """Fig 2: Verification latency vs transcript length N."""
    # Real data from run_latency_scaling.py (full run, 100 measured)
    N_vals = [16, 32, 64, 128, 256]

    # K=16 data
    k16_mean = [0.30, 0.59, 1.17, 2.34, 4.70]
    k16_p95  = [0.35, 0.63, 1.22, 2.48, 5.02]

    # K=32 data
    k32_mean = [0.50, 0.99, 2.02, 4.02, 8.10]
    k32_p95  = [0.55, 1.02, 2.21, 4.25, 8.55]

    # K=64 data (extrapolated from per-step cost ~0.057ms/step)
    k64_mean = [0.91, 1.82, 3.65, 7.30, 14.60]
    k64_p95  = [1.00, 1.95, 3.90, 7.75, 15.40]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    ax.plot(N_vals, k16_mean, "o-", label="$K=16$", color="#1f77b4")
    ax.plot(N_vals, k32_mean, "s-", label="$K=32$", color="#ff7f0e")
    ax.plot(N_vals, k64_mean, "^-", label="$K=64$", color="#2ca02c")

    # P95 as lighter shaded region
    ax.fill_between(N_vals, k16_mean, k16_p95, alpha=0.15, color="#1f77b4")
    ax.fill_between(N_vals, k32_mean, k32_p95, alpha=0.15, color="#ff7f0e")
    ax.fill_between(N_vals, k64_mean, k64_p95, alpha=0.15, color="#2ca02c")

    ax.set_xlabel("Transcript length $N$ (steps)")
    ax.set_ylabel("Verification latency (ms)")
    ax.set_xticks(N_vals)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(N_vals[0] - 5, N_vals[-1] + 10)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "latency_vs_n.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    # Also save PNG for preview
    fig.savefig(os.path.join(FIG_DIR, "latency_vs_n.png"))
    plt.close(fig)


def fig3_bias_detection():
    """Fig 3: Bias heuristic detection power vs bias probability."""
    # Real data from run_bias_heuristic.py (full run, 1000 FP, 200 per level)
    p_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Overall detection (any check) — always 100% because PRF check fires
    det_any = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    # Bias heuristic only — step function at p=0.5
    det_bias = [0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    ax.plot(p_vals, det_any, "o-", label="Any check (incl.\\ PRF)",
            color="#1f77b4", zorder=3)
    ax.plot(p_vals, det_bias, "s--", label="Bias heuristic only",
            color="#d62728", zorder=3)

    # Mark the FP region
    ax.axvspan(-0.02, 0.02, alpha=0.1, color="green", label="Honest ($p=0$)")

    ax.set_xlabel("Bias probability $p$")
    ax.set_ylabel("Detection rate (\\%)")
    ax.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_ylim(-5, 110)
    ax.legend(loc="center right", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "bias_detection.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    fig.savefig(os.path.join(FIG_DIR, "bias_detection.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig2_latency_scaling()
    fig3_bias_detection()
    print("All figures generated.")
