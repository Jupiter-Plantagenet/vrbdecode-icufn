#!/usr/bin/env python3
"""Generate figures for the ICUFN paper from experimental result files.

Reads data from eval/*.json -- never uses hardcoded values.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EVAL_DIR = os.path.join(ROOT, "eval")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

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
    """Fig 2: Verification latency vs transcript length N.

    Reads from eval/latency_scaling_results.json.
    """
    json_path = os.path.join(EVAL_DIR, "latency_scaling_results.json")
    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]

    # Group by K
    by_K = {}
    for r in results:
        by_K.setdefault(r["K"], []).append(r)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    colors = {16: "#1f77b4", 32: "#ff7f0e", 64: "#2ca02c"}
    markers = {16: "o", 32: "s", 64: "^"}

    for K in sorted(by_K.keys()):
        rows = sorted(by_K[K], key=lambda r: r["N"])
        Ns = [r["N"] for r in rows]
        means = [r["mean_ms"] for r in rows]
        p95s = [r["p95_ms"] for r in rows]
        ax.plot(Ns, means, f"{markers[K]}-", label=f"$K = {K}$",
                color=colors[K])
        ax.fill_between(Ns, means, p95s, alpha=0.15, color=colors[K])

    ax.set_xlabel("Transcript length $N$ (steps)")
    ax.set_ylabel("Verification latency (ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256])
    ax.set_xticklabels(["16", "32", "64", "128", "256"])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"latency_vs_n.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def fig3_bias_detection():
    """Fig 3: Bias heuristic detection power vs bias probability.

    Reads from eval/bias_heuristic_results.json.
    """
    json_path = os.path.join(EVAL_DIR, "bias_heuristic_results.json")
    with open(json_path) as f:
        data = json.load(f)

    levels = data["detection_power"]

    p_vals = []
    det_any = []
    det_bias = []

    for level in levels:
        p_vals.append(level["bias_level"])
        det_any.append(level["detection_rate_any"] * 100)
        det_bias.append(level["detection_rate_bias_heuristic"] * 100)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    ax.plot(p_vals, det_any, "o-", label="Any check (incl.\\ PRF)",
            color="#1f77b4", zorder=3)
    ax.plot(p_vals, det_bias, "s--", label="Bias heuristic only",
            color="#d62728", zorder=3)
    ax.axvspan(-0.02, 0.02, alpha=0.1, color="green", label="Honest ($p=0$)")

    ax.set_xlabel("Bias probability $p$")
    ax.set_ylabel("Detection rate (\\%)")
    ax.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_ylim(-5, 110)
    ax.legend(loc="center right", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"bias_detection.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    fig2_latency_scaling()
    fig3_bias_detection()
    print("All figures generated from experimental data.")
