#!/usr/bin/env python3
"""Full GPT-2 forensic verification experiment (100 prompts, 4 attack types).

Extracts top-K logit distributions from GPT-2 (117M) across 100 diverse
prompts (4 categories x 25), generates honest transcripts and four attack
variants, then runs forensic verification with statistical analysis.

Output: eval/gpt2_validation_results.json
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REF_PY = os.path.join(ROOT, "ref", "python")
sys.path.insert(0, REF_PY)

from forensic_verifier import VerifyCode, verify_transcript
from receipt import PolicyParams, generate_honest_transcript, serialize_transcript
from attack_simulator import (
    attack_policy_mismatch,
    attack_randomness_replay,
    attack_candidate_manipulation,
    attack_transcript_drop,
)


# ---------------------------------------------------------------------------
# Prompt bank: 4 categories x 25 prompts = 100 total
# ---------------------------------------------------------------------------

PROMPTS_NEWS_FORMAL = [
    "The president announced today that the new policy would",
    "According to the latest financial report, the company's revenue",
    "In a press conference held earlier this morning, officials stated",
    "The central bank raised interest rates by 25 basis points on",
    "A new report from the World Health Organization suggests that",
    "The United Nations Security Council voted unanimously to impose",
    "Stocks rallied sharply on Wall Street after the Federal Reserve",
    "The prime minister addressed parliament today regarding the ongoing",
    "A major earthquake measuring 7.1 on the Richter scale struck",
    "The European Commission proposed new regulations that would require",
    "Following months of negotiations, the two countries finally agreed",
    "The Supreme Court issued a landmark ruling today that effectively",
    "A bipartisan group of senators introduced legislation that would",
    "The International Monetary Fund warned that global economic growth",
    "Government officials confirmed that the infrastructure spending bill",
    "The trade deficit widened to a record level in the third quarter",
    "Scientists at the National Institutes of Health published findings",
    "The defense department announced a new contract worth approximately",
    "Election results from the closely watched swing state showed that",
    "The housing market experienced its sharpest decline in a decade",
    "Diplomatic talks between the two nations broke down after the",
    "The technology sector led the market downturn as investors reacted",
    "A congressional committee launched an investigation into allegations",
    "The labor department reported that unemployment claims rose by",
    "Environmental regulators issued new guidelines for emissions from",
]

PROMPTS_CREATIVE_NARRATIVE = [
    "Once upon a time in a land far away, there lived a",
    "The old lighthouse keeper stared out at the stormy sea and",
    "She opened the dusty leather journal and began to read the",
    "In the depths of the enchanted forest, a mysterious light",
    "The detective examined the crime scene carefully, noting that the",
    "As the spaceship descended through the thick atmosphere of the",
    "The musician picked up the violin and played a melody that",
    "Rain pattered against the window as she sat alone in the",
    "The ancient map revealed a hidden passage beneath the castle",
    "He woke up in a world where everything was made of",
    "The last dragon in the kingdom had been sleeping for centuries",
    "A letter arrived at midnight, sealed with red wax and addressed",
    "The robot looked at its reflection and wondered if it could",
    "Deep beneath the ocean surface, the submarine crew discovered an",
    "The magician waved his wand and the entire audience gasped as",
    "Through the fog, a figure emerged walking slowly toward the",
    "The princess refused to be rescued and instead drew her sword",
    "In the year 2147, humanity had finally mastered the art of",
    "The baker woke before dawn every morning to prepare the sourdough",
    "A stray cat wandered into the bookshop and made itself comfortable",
    "The time traveler arrived in ancient Rome and immediately realized",
    "Under the pale moonlight, the garden seemed to come alive with",
    "The pirate captain studied the treasure map, comparing it to the",
    "An unexpected knock on the door interrupted her quiet evening of",
    "The painter stepped back from the canvas and studied what she had",
]

PROMPTS_CODE_TECHNICAL = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The chemical formula for water is H2O, and the formula for",
    "In the context of machine learning, gradient descent is used to",
    "import numpy as np\n\ndef matrix_multiply(A, B):\n    return",
    "The TCP three-way handshake begins when the client sends a",
    "In distributed systems, the CAP theorem states that it is impossible",
    "class BinarySearchTree:\n    def __init__(self):\n        self.root =",
    "The time complexity of merge sort is O(n log n) because the",
    "According to the second law of thermodynamics, the entropy of an",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "In quantum computing, a qubit can exist in a superposition of",
    "The RSA encryption algorithm relies on the computational difficulty of",
    "SELECT users.name, orders.total FROM users INNER JOIN orders ON",
    "A convolutional neural network typically consists of convolutional layers",
    "The Fourier transform decomposes a function into its constituent frequency",
    "In compiler design, the lexical analyzer converts the source code into",
    "async def fetch_data(url):\n    async with aiohttp.ClientSession() as",
    "The halting problem, proven undecidable by Turing, states that no",
    "When implementing a hash table, collision resolution can be handled by",
    "The backpropagation algorithm computes gradients by applying the chain rule",
    "In operating systems, virtual memory allows processes to use more memory",
    "The Dijkstra algorithm finds the shortest path in a weighted graph by",
    "A B-tree of order m has the property that each node can",
    "In cryptography, a hash function must satisfy three properties: preimage",
    "The HTTP protocol is stateless, meaning that each request from the",
]

PROMPTS_CONVERSATIONAL = [
    "Hey, have you heard about the new restaurant that just opened",
    "I was thinking about what you said yesterday, and honestly I",
    "So the thing is, I really need your help with something",
    "Can you believe what happened at the meeting today? They actually",
    "I just got back from vacation and you would not believe",
    "Do you remember that time we went camping and it started",
    "I have been meaning to tell you something important about the",
    "What do you think about getting a dog? I saw this",
    "My neighbor just told me the craziest story about what happened",
    "I tried cooking that recipe you sent me, but I think",
    "So I finally watched that show everyone keeps talking about and",
    "I am not sure if I should take the job offer",
    "The kids have been driving me crazy all day because they",
    "Have you seen the latest episode? I cannot believe they killed",
    "I ran into our old friend from college yesterday at the",
    "You know what really bothers me about the whole situation is",
    "I was walking home from work when I noticed something strange",
    "Let me tell you about my experience at the airport last",
    "I think we should plan a trip somewhere this summer, maybe",
    "The funniest thing happened to me at the grocery store today",
    "I need some advice about my car because it has been",
    "Did you hear the news about the company? Apparently they are",
    "I have been learning to play guitar and it is much",
    "So my doctor told me I need to start exercising more",
    "I was looking at old photos and I found the one",
]

CATEGORY_NAMES = ["news_formal", "creative_narrative", "code_technical", "conversational"]
ALL_PROMPTS = (
    PROMPTS_NEWS_FORMAL
    + PROMPTS_CREATIVE_NARRATIVE
    + PROMPTS_CODE_TECHNICAL
    + PROMPTS_CONVERSATIONAL
)
PROMPT_CATEGORIES = (
    ["news_formal"] * 25
    + ["creative_narrative"] * 25
    + ["code_technical"] * 25
    + ["conversational"] * 25
)

assert len(ALL_PROMPTS) == 100, f"Expected 100 prompts, got {len(ALL_PROMPTS)}"
assert len(PROMPT_CATEGORIES) == 100


# ---------------------------------------------------------------------------
# GPT-2 logit extraction
# ---------------------------------------------------------------------------

def extract_gpt2_topk(prompts: list[str], K: int = 32, max_steps: int = 32):
    """Run GPT-2 and extract top-K logits at each autoregressive step.

    Returns list of dicts with prompt, K, n_steps, steps[{token_ids, logits_q16}].
    """
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("  Loading GPT-2 (117M)...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    all_extractions = []
    t_start = time.perf_counter()

    for pi, prompt in enumerate(prompts):
        elapsed = time.perf_counter() - t_start
        eta = (elapsed / max(pi, 1)) * (len(prompts) - pi) if pi > 0 else 0
        print(
            f"  [{pi+1:3d}/{len(prompts)}] "
            f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s) "
            f"{prompt[:50]}...",
            flush=True,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        steps = []

        with torch.no_grad():
            for step in range(max_steps):
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]  # last position

                # Get top-K
                topk_values, topk_indices = torch.topk(logits, K)

                # Convert to Q16.16 fixed-point (matching our pipeline)
                token_ids = topk_indices.tolist()
                logits_q16 = [int(v * (1 << 16)) for v in topk_values.tolist()]

                steps.append({
                    "token_ids": token_ids,
                    "logits_q16": logits_q16,
                })

                # Greedy decode for next step
                next_token = topk_indices[0].unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

        all_extractions.append({
            "prompt": prompt,
            "K": K,
            "n_steps": len(steps),
            "steps": steps,
        })

    return all_extractions


# ---------------------------------------------------------------------------
# Entropy analysis
# ---------------------------------------------------------------------------

def compute_step_entropy(logits_q16: list[int]) -> float:
    """Compute Shannon entropy of softmax distribution from Q16 logits."""
    logits_f = [l / (1 << 16) for l in logits_q16]
    max_l = max(logits_f)
    exps = [math.exp(l - max_l) for l in logits_f]
    total = sum(exps)
    probs = [e / total for e in exps]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) as fractions in [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    lo = max(0.0, centre - spread)
    hi = min(1.0, centre + spread)
    return (lo, hi)


def mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mu = sum(values) / n
    if n == 1:
        return (mu, 0.0)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return (mu, math.sqrt(var))


def ci_of_mean(values: List[float], z: float = 1.96) -> Tuple[float, float]:
    """95% CI for the mean of a list of values (normal approx)."""
    mu, sd = mean_std(values)
    n = len(values)
    if n <= 1:
        return (mu, mu)
    se = sd / math.sqrt(n)
    return (mu - z * se, mu + z * se)


# ---------------------------------------------------------------------------
# Attack runners
# ---------------------------------------------------------------------------

ATTACK_TYPES = [
    "policy_mismatch",
    "randomness_replay",
    "candidate_manipulation_no_gt",
    "candidate_manipulation_gt",
    "transcript_drop",
]


def run_attack(
    attack_name: str,
    transcript,
    policy: PolicyParams,
    seed: bytes,
    candidates: list,
    rng: random.Random,
) -> dict:
    """Run a single attack and verify. Returns dict with results."""
    t0 = time.perf_counter()

    if attack_name == "policy_mismatch":
        tampered = attack_policy_mismatch(
            transcript, 1.0,
            new_T_q16=policy.T_q16 // 2,
            rng=rng,
        )
        results = verify_transcript(tampered, policy, seed)

    elif attack_name == "randomness_replay":
        tampered = attack_randomness_replay(
            transcript, 0.5,
            rng=rng,
        )
        results = verify_transcript(tampered, policy, seed)

    elif attack_name == "candidate_manipulation_no_gt":
        # Without ground truth: verifier has no GT candidates
        tampered = attack_candidate_manipulation(
            transcript, 0.5,
            rng=rng,
        )
        results = verify_transcript(tampered, policy, seed)

    elif attack_name == "candidate_manipulation_gt":
        # With ground truth: verifier has GT candidates
        tampered = attack_candidate_manipulation(
            transcript, 0.5,
            rng=rng,
        )
        gt = [(s["token_ids"], s["logits_q16"]) for s in candidates]
        results = verify_transcript(
            tampered, policy, seed,
            ground_truth_candidates=gt,
        )

    elif attack_name == "transcript_drop":
        tampered = attack_transcript_drop(
            transcript, 0.5,
            rng=rng,
        )
        results = verify_transcript(tampered, policy, seed)

    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    elapsed = time.perf_counter() - t0
    detected = any(r.code != VerifyCode.PASS for r in results)
    codes = [r.code.value for r in results if r.code != VerifyCode.PASS]

    return {
        "detected": detected,
        "verify_time_ms": elapsed * 1000,
        "failure_codes": codes,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> int:
    K = 32
    max_steps = 32
    n_prompts = len(ALL_PROMPTS)

    print("=" * 72)
    print("GPT-2 FORENSIC VERIFICATION EXPERIMENT")
    print("=" * 72)
    print(f"  K={K}, N={max_steps} steps/prompt, {n_prompts} prompts")
    print(f"  Categories: {', '.join(CATEGORY_NAMES)} (25 each)")
    print(f"  Attack types: {', '.join(ATTACK_TYPES)}")
    print()

    # ---------------------------------------------------------------
    # Phase 1: Extract real logits from GPT-2
    # ---------------------------------------------------------------
    print("Phase 1: Extracting GPT-2 logits...")
    t0 = time.perf_counter()
    extractions = extract_gpt2_topk(ALL_PROMPTS, K=K, max_steps=max_steps)
    extract_time = time.perf_counter() - t0
    print(f"  Extraction completed in {extract_time:.1f}s")
    print()

    # ---------------------------------------------------------------
    # Phase 2: Entropy analysis by category
    # ---------------------------------------------------------------
    print("Phase 2: Entropy analysis...")
    all_entropies: List[float] = []
    entropies_by_cat: Dict[str, List[float]] = {c: [] for c in CATEGORY_NAMES}

    for ext, cat in zip(extractions, PROMPT_CATEGORIES):
        step_entropies = [
            compute_step_entropy(s["logits_q16"]) for s in ext["steps"]
        ]
        all_entropies.extend(step_entropies)
        entropies_by_cat[cat].extend(step_entropies)

    overall_mu, overall_sd = mean_std(all_entropies)
    print(f"  Overall entropy: mean={overall_mu:.3f} std={overall_sd:.3f} "
          f"min={min(all_entropies):.3f} max={max(all_entropies):.3f}")

    entropy_stats: Dict[str, dict] = {}
    for cat in CATEGORY_NAMES:
        vals = entropies_by_cat[cat]
        mu, sd = mean_std(vals)
        ci_lo, ci_hi = ci_of_mean(vals)
        entropy_stats[cat] = {
            "mean": round(mu, 4),
            "std": round(sd, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
            "n_steps": len(vals),
        }
        print(f"  {cat:25s}: mean={mu:.3f} std={sd:.3f} "
              f"[{ci_lo:.3f}, {ci_hi:.3f}] "
              f"min={min(vals):.3f} max={max(vals):.3f}")
    print()

    # ---------------------------------------------------------------
    # Phase 3: Forensic verification (honest + all attacks)
    # ---------------------------------------------------------------
    print("Phase 3: Forensic verification (honest + 5 attack variants)...")
    rng = random.Random(2025)

    # Per-prompt results
    per_prompt_results: List[dict] = []

    # Aggregate counters
    honest_pass_count = 0
    honest_fail_count = 0
    attack_results_agg: Dict[str, dict] = {
        a: {"detected": 0, "missed": 0, "times_ms": []}
        for a in ATTACK_TYPES
    }
    all_honest_latencies: List[float] = []
    all_evidence_sizes: List[int] = []

    for pi, (ext, cat) in enumerate(zip(extractions, PROMPT_CATEGORIES)):
        if (pi + 1) % 10 == 0 or pi == 0:
            print(f"  [{pi+1:3d}/{n_prompts}] Processing...", flush=True)

        n_steps = ext["n_steps"]
        candidates = ext["steps"]
        cand_tuples = [
            (s["token_ids"], s["logits_q16"]) for s in candidates
        ]

        policy = PolicyParams(
            K=K,
            top_k=max(1, K // 4),
            top_p_q16=int(0.9 * (1 << 16)),
            T_q16=1 << 16,
            max_tokens=n_steps,
        )
        seed = rng.randbytes(32)
        request_id = rng.randbytes(32)

        # Generate honest transcript
        transcript = generate_honest_transcript(policy, seed, request_id, cand_tuples)

        # Verify honest transcript
        t0 = time.perf_counter()
        honest_results = verify_transcript(transcript, policy, seed)
        honest_lat = (time.perf_counter() - t0) * 1000
        honest_pass = all(r.code == VerifyCode.PASS for r in honest_results)

        if honest_pass:
            honest_pass_count += 1
        else:
            honest_fail_count += 1

        # Measure honest verification latency (10 runs, skip first as warmup)
        honest_lats = []
        for run_i in range(11):
            t0 = time.perf_counter()
            verify_transcript(transcript, policy, seed)
            lat = (time.perf_counter() - t0) * 1000
            if run_i > 0:  # skip warmup
                honest_lats.append(lat)
        all_honest_latencies.extend(honest_lats)

        evidence_size = len(serialize_transcript(transcript))
        all_evidence_sizes.append(evidence_size)

        # Step entropies for this prompt
        step_entropies = [
            compute_step_entropy(s["logits_q16"]) for s in ext["steps"]
        ]

        # Run all attacks
        prompt_attack_results = {}
        for attack_name in ATTACK_TYPES:
            # Use a fresh RNG fork for reproducibility
            attack_rng = random.Random(rng.randint(0, 2**63))
            ar = run_attack(
                attack_name, transcript, policy, seed, candidates, attack_rng
            )
            prompt_attack_results[attack_name] = {
                "detected": ar["detected"],
                "verify_time_ms": round(ar["verify_time_ms"], 4),
                "failure_codes": ar["failure_codes"],
            }
            if ar["detected"]:
                attack_results_agg[attack_name]["detected"] += 1
            else:
                attack_results_agg[attack_name]["missed"] += 1
            attack_results_agg[attack_name]["times_ms"].append(ar["verify_time_ms"])

        per_prompt_results.append({
            "prompt_index": pi,
            "prompt": ext["prompt"][:80],
            "category": cat,
            "n_steps": n_steps,
            "honest_pass": honest_pass,
            "honest_latency_mean_ms": round(sum(honest_lats) / len(honest_lats), 4),
            "evidence_bytes": evidence_size,
            "entropy_mean": round(sum(step_entropies) / len(step_entropies), 4),
            "entropy_min": round(min(step_entropies), 4),
            "entropy_max": round(max(step_entropies), 4),
            "attacks": prompt_attack_results,
        })

    print()

    # ---------------------------------------------------------------
    # Phase 4: Statistical summary
    # ---------------------------------------------------------------
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    # Honest transcript FP
    fp_rate = honest_fail_count / n_prompts
    fp_ci = wilson_ci(honest_fail_count, n_prompts)
    tp_rate = honest_pass_count / n_prompts
    print(f"\n  Honest transcripts: {honest_pass_count}/{n_prompts} pass "
          f"(FP rate = {fp_rate*100:.1f}%, "
          f"95% Wilson CI [{fp_ci[0]*100:.2f}%, {fp_ci[1]*100:.2f}%])")

    # Per-attack detection rates with CIs
    print(f"\n  {'Attack Type':<35s} {'Det':>4} {'Miss':>5} {'Rate':>7} {'95% CI':>20}")
    print(f"  {'-'*75}")
    attack_summary = {}
    for attack_name in ATTACK_TYPES:
        agg = attack_results_agg[attack_name]
        det = agg["detected"]
        miss = agg["missed"]
        total = det + miss
        rate = det / total if total > 0 else 0
        ci = wilson_ci(det, total)
        attack_summary[attack_name] = {
            "detected": det,
            "missed": miss,
            "total": total,
            "detection_rate": round(rate, 4),
            "ci_95_lo": round(ci[0], 4),
            "ci_95_hi": round(ci[1], 4),
        }

        # Latency stats
        times = agg["times_ms"]
        if times:
            lat_mu, lat_sd = mean_std(times)
            attack_summary[attack_name]["verify_latency_mean_ms"] = round(lat_mu, 4)
            attack_summary[attack_name]["verify_latency_std_ms"] = round(lat_sd, 4)

        print(f"  {attack_name:<35s} {det:>4} {miss:>5} {rate*100:>6.1f}% "
              f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")

    # Verification latency stats
    lat_mu, lat_sd = mean_std(all_honest_latencies)
    lat_ci = ci_of_mean(all_honest_latencies)
    per_step_lat = lat_mu / max_steps if max_steps > 0 else 0
    print(f"\n  Honest verification latency (per transcript):")
    print(f"    mean={lat_mu:.3f} ms  std={lat_sd:.3f} ms  "
          f"95% CI [{lat_ci[0]:.3f}, {lat_ci[1]:.3f}] ms")
    print(f"    per-step: {per_step_lat:.4f} ms/step")
    print(f"    min={min(all_honest_latencies):.3f} ms  "
          f"max={max(all_honest_latencies):.3f} ms")

    # Evidence size stats
    size_mu, size_sd = mean_std([float(s) for s in all_evidence_sizes])
    print(f"\n  Evidence size: mean={size_mu:.0f} B  std={size_sd:.0f} B  "
          f"min={min(all_evidence_sizes)} B  max={max(all_evidence_sizes)} B")

    # Per-category entropy table
    print(f"\n  {'Category':<25s} {'H mean':>7} {'H std':>7} {'H min':>7} {'H max':>7} {'95% CI':>20}")
    print(f"  {'-'*78}")
    for cat in CATEGORY_NAMES:
        es = entropy_stats[cat]
        print(f"  {cat:<25s} {es['mean']:>7.3f} {es['std']:>7.3f} "
              f"{es['min']:>7.3f} {es['max']:>7.3f} "
              f"[{es['ci_95_lo']:.3f}, {es['ci_95_hi']:.3f}]")

    # ---------------------------------------------------------------
    # Save full results
    # ---------------------------------------------------------------
    output = {
        "config": {
            "K": K,
            "max_steps": max_steps,
            "n_prompts": n_prompts,
            "categories": CATEGORY_NAMES,
            "prompts_per_category": 25,
            "attack_types": ATTACK_TYPES,
            "extraction_time_s": round(extract_time, 1),
        },
        "honest_verification": {
            "pass_count": honest_pass_count,
            "fail_count": honest_fail_count,
            "total": n_prompts,
            "fp_rate": round(fp_rate, 6),
            "fp_rate_wilson_ci_95": [round(fp_ci[0], 6), round(fp_ci[1], 6)],
        },
        "attack_detection": attack_summary,
        "entropy_overall": {
            "mean": round(overall_mu, 4),
            "std": round(overall_sd, 4),
            "min": round(min(all_entropies), 4),
            "max": round(max(all_entropies), 4),
            "ci_95": list(map(lambda x: round(x, 4), ci_of_mean(all_entropies))),
            "n_steps_total": len(all_entropies),
        },
        "entropy_by_category": entropy_stats,
        "verification_latency": {
            "mean_ms": round(lat_mu, 4),
            "std_ms": round(lat_sd, 4),
            "ci_95_ms": [round(lat_ci[0], 4), round(lat_ci[1], 4)],
            "per_step_ms": round(per_step_lat, 4),
            "min_ms": round(min(all_honest_latencies), 4),
            "max_ms": round(max(all_honest_latencies), 4),
            "n_runs": len(all_honest_latencies),
        },
        "evidence_size": {
            "mean_bytes": round(size_mu, 1),
            "std_bytes": round(size_sd, 1),
            "min_bytes": min(all_evidence_sizes),
            "max_bytes": max(all_evidence_sizes),
        },
        "per_prompt_results": per_prompt_results,
    }

    out_path = os.path.join(SCRIPT_DIR, "gpt2_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
