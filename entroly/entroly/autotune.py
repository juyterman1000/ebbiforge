"""
Entroly Autotune -- Autonomous Self-Tuning Loop
==============================================

Keep/discard experimentation loop to autonomously improve Entroly's
hyperparameters. Each iteration mutates one parameter in tuning_config.json,
evaluates the result on a fixed benchmark suite, and keeps improvements.

For Entroly, the loop maps to:
  - The mutable surface  = bench/tuning_config.json (the ONLY file we mutate)
  - The evaluation step  = running optimize_context() on benchmark cases
  - The objective metric = context_efficiency (information_retained / tokens_used)
  - The keep/discard     = compare efficiency, keep improvements, revert regressions

Each iteration takes seconds on CPU, so ~1000 experiments run overnight.

Usage:
    python -m entroly.autotune                    # Run 100 iterations
    python -m entroly.autotune --iterations 500   # Run 500 iterations
    python -m entroly.autotune --bench-only       # Just evaluate current config
    python -m entroly.autotune --time-budget 60   # Max seconds per iteration

Single-file mutation discipline:
  - Only bench/tuning_config.json is modified
  - bench/cases.json is read-only (the fixed validation set)
  - This file (autotune.py) is read-only (the evaluation harness)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -- Constants ---------------------------------------------------------------

BENCH_DIR = Path(__file__).parent.parent / "bench"
CASES_PATH = BENCH_DIR / "cases.json"
CONFIG_PATH = BENCH_DIR / "tuning_config.json"
RESULTS_PATH = BENCH_DIR / "results.tsv"

# Fixed time budget per benchmark evaluation.
# Iterations exceeding this are auto-discarded (poor configs stall the loop).
DEFAULT_TIME_BUDGET_SECS = 5.0

# Parameters and their mutation ranges
TUNABLE_PARAMS = {
    "weight_recency":       (0.05, 0.80),
    "weight_frequency":     (0.05, 0.80),
    "weight_semantic_sim":  (0.05, 0.80),
    "weight_entropy":       (0.05, 0.80),
    "decay_half_life_turns": (5, 50),
    "min_relevance_threshold": (0.01, 0.20),
    "exploration_rate":     (0.0, 0.3),
}


@dataclass
class BenchResult:
    """Result of running the benchmark suite with a given config."""
    context_efficiency: float
    recall_accuracy: float
    avg_wall_time_ms: float
    total_tokens_used: int
    total_information: float
    per_case: List[Dict[str, Any]] = field(default_factory=list)


def load_cases() -> List[Dict[str, Any]]:
    """Load the fixed benchmark cases (read-only val set)."""
    with open(CASES_PATH) as f:
        return json.load(f)


def load_config() -> Dict[str, Any]:
    """Load the current tuning config (the file we mutate)."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: Dict[str, Any]) -> None:
    """Save tuning config (single-file mutation)."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def evaluate(config: Dict[str, Any], cases: List[Dict[str, Any]],
             time_budget: float = DEFAULT_TIME_BUDGET_SECS) -> BenchResult:
    """
    Run the benchmark suite with a given config.

    Fixed time budget evaluation: if any single case exceeds time_budget seconds,
    the entire run is marked as failed.
    """
    try:
        from entroly_core import EntrolyEngine
    except ImportError:
        print("ERROR: entroly_core not available. Run `maturin develop` first.",
              file=sys.stderr)
        sys.exit(1)

    total_information = 0.0
    total_tokens_used = 0
    correct_selections = 0
    total_expected = 0
    wall_times: List[float] = []
    per_case: List[Dict[str, Any]] = []

    for case in cases:
        engine = EntrolyEngine(
            w_recency=config.get("weight_recency", 0.30),
            w_frequency=config.get("weight_frequency", 0.25),
            w_semantic=config.get("weight_semantic_sim", 0.25),
            w_entropy=config.get("weight_entropy", 0.20),
            decay_half_life=config.get("decay_half_life_turns", 15),
            min_relevance=config.get("min_relevance_threshold", 0.05),
            exploration_rate=config.get("exploration_rate", 0.1),
        )

        frag_id_map: Dict[str, str] = {}
        for frag_data in case["fragments"]:
            result = engine.ingest(
                frag_data["content"],
                frag_data["source"],
                frag_data["token_count"],
                False,
            )
            if hasattr(result, '__getitem__'):
                fid = result.get("fragment_id", result.get("duplicate_of", ""))
            else:
                fid = str(result)
            frag_id_map[frag_data["source"]] = fid

        t0 = time.perf_counter()
        opt_result = engine.optimize(case["token_budget"], case["query"])
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000

        if wall_ms > time_budget * 1000:
            return BenchResult(
                context_efficiency=0.0, recall_accuracy=0.0,
                avg_wall_time_ms=wall_ms, total_tokens_used=0,
                total_information=0.0,
                per_case=[{"case_id": case["id"], "status": "timeout",
                           "wall_ms": wall_ms}],
            )

        wall_times.append(wall_ms)

        selected_sources: set = set()
        if hasattr(opt_result, '__getitem__'):
            selected_list = opt_result.get("selected", [])
            for item in selected_list:
                if hasattr(item, '__getitem__'):
                    src = item.get("source", "")
                    if src:
                        selected_sources.add(src)

        expected_sources = {
            f["source"] for f in case["fragments"] if f.get("expected_selected")
        }
        hits = len(selected_sources & expected_sources)
        total_expected += len(expected_sources)
        correct_selections += hits

        case_tokens = sum(
            f["token_count"] for f in case["fragments"]
            if f["source"] in selected_sources
        )
        case_info = sum(
            1.0 for f in case["fragments"]
            if f["source"] in selected_sources and f.get("expected_selected")
        )
        total_tokens_used += case_tokens
        total_information += case_info

        per_case.append({
            "case_id": case["id"],
            "recall": hits / len(expected_sources) if expected_sources else 1.0,
            "tokens_used": case_tokens,
            "wall_ms": round(wall_ms, 2),
            "selected": list(selected_sources),
        })

    ctx_eff = total_information / max(total_tokens_used, 1)
    recall_acc = correct_selections / max(total_expected, 1)
    avg_wall = sum(wall_times) / max(len(wall_times), 1)

    return BenchResult(
        context_efficiency=ctx_eff,
        recall_accuracy=recall_acc,
        avg_wall_time_ms=avg_wall,
        total_tokens_used=total_tokens_used,
        total_information=total_information,
        per_case=per_case,
    )


def mutate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate one parameter at a time (single-change experiments for interpretability)."""
    new_config = dict(config)
    param = random.choice(list(TUNABLE_PARAMS.keys()))
    lo, hi = TUNABLE_PARAMS[param]
    current = new_config.get(param, (lo + hi) / 2)

    if isinstance(lo, int):
        delta = random.randint(-3, 3)
        new_val = max(lo, min(hi, int(current) + delta))
    else:
        sigma = (hi - lo) * 0.10
        new_val = current + random.gauss(0, sigma)
        new_val = max(lo, min(hi, round(new_val, 4)))

    new_config[param] = new_val

    weight_keys = ["weight_recency", "weight_frequency",
                   "weight_semantic_sim", "weight_entropy"]
    weight_sum = sum(new_config.get(k, 0.25) for k in weight_keys)
    if weight_sum > 0:
        for k in weight_keys:
            new_config[k] = round(new_config.get(k, 0.25) / weight_sum, 4)

    new_config["_mutated_param"] = param
    return new_config


def composite_score(result: BenchResult, config: Optional[Dict[str, Any]] = None,
                    defaults: Optional[Dict[str, Any]] = None,
                    drift_weight: float = 0.1) -> float:
    """Single metric: efficiency + recall - config drift penalty.

    Adds a drift penalty that penalises configs straying from defaults
    (prevents adversarial parameter regions).

    composite = 0.6·recall + 0.4·efficiency×100 - drift_weight·drift²×100
    """
    base = 0.6 * result.recall_accuracy + 0.4 * result.context_efficiency * 100

    if config is None or defaults is None or drift_weight <= 0:
        return base

    drift_sq = 0.0
    count = 0
    for key, (lo, hi) in TUNABLE_PARAMS.items():
        if key in config and key in defaults:
            span = max(float(hi) - float(lo), 1e-9)
            delta = (float(config[key]) - float(defaults.get(key, config[key]))) / span
            drift_sq += delta * delta
            count += 1

    if count > 0:
        drift = (drift_sq / count) ** 0.5
        return base - drift_weight * drift * drift * 100
    return base


# ══════════════════════════════════════════════════════════════════════
# Cautious Parameter Updates — Momentum-Dampened Autotune
# ══════════════════════════════════════════════════════════════════════
#
# Instead of binary keep/discard, three mechanisms:
#
# 1. EMA Blending: Instead of full replacement, blend the winning
#    config with the current best:
#      p_new = (1 - α) · p_old + α · p_candidate
#    where α scales with improvement magnitude (big improvement = faster adoption).
#
# 2. Polyak Averaging: Maintain a running average of all kept configs.
#    After tuning completes, the Polyak average is often more robust
#    than the single best (Polyak & Juditsky, 1992).
#
# 3. Config Drift Penalty: The composite score penalises configs that
#    stray too far from defaults, preventing the autotuner from finding
#    adversarial parameter regions that overfit the benchmark.
# ══════════════════════════════════════════════════════════════════════

def _ema_blend(best: Dict[str, Any], candidate: Dict[str, Any],
               alpha: float) -> Dict[str, Any]:
    """EMA blend: p = (1-α)·best + α·candidate for numeric params."""
    blended = dict(best)
    for key in TUNABLE_PARAMS:
        if key in candidate and key in best:
            old_val = float(best[key])
            new_val = float(candidate[key])
            val = (1 - alpha) * old_val + alpha * new_val
            lo, hi = TUNABLE_PARAMS[key]
            val = max(float(lo), min(float(hi), val))
            if isinstance(lo, int):
                blended[key] = int(round(val))
            else:
                blended[key] = round(val, 4)
    return blended


def _polyak_update(avg: Dict[str, Any], config: Dict[str, Any],
                   count: int) -> Dict[str, Any]:
    """Polyak running average: avg = ((n-1)·avg + config) / n."""
    updated = dict(avg)
    for key in TUNABLE_PARAMS:
        if key in config and key in avg:
            old_avg = float(avg[key])
            new_val = float(config[key])
            val = (old_avg * (count - 1) + new_val) / count
            lo, hi = TUNABLE_PARAMS[key]
            val = max(float(lo), min(float(hi), val))
            if isinstance(lo, int):
                updated[key] = int(round(val))
            else:
                updated[key] = round(val, 4)
    return updated


def log_result(iteration: int, config: Dict[str, Any], result: BenchResult,
               status: str, description: str) -> None:
    """Append to results.tsv (structured experiment log)."""
    header = "iteration\tscore\trecall\tefficiency\tavg_ms\tstatus\tdescription\n"
    if not RESULTS_PATH.exists():
        with open(RESULTS_PATH, "w") as f:
            f.write(header)

    score = composite_score(result)
    with open(RESULTS_PATH, "a") as f:
        f.write(f"{iteration}\t{score:.4f}\t{result.recall_accuracy:.4f}\t"
                f"{result.context_efficiency:.6f}\t{result.avg_wall_time_ms:.1f}\t"
                f"{status}\t{description}\n")


def run_autotune(iterations: int = 100,
                 time_budget: float = DEFAULT_TIME_BUDGET_SECS,
                 bench_only: bool = False) -> None:
    """
    The experiment loop: mutate → evaluate → keep/discard → repeat.

    LOOP:
      1. Load current config
      2. Mutate one parameter
      3. Evaluate on benchmark suite
      4. If score improved → keep (advance)
      5. If score equal or worse → discard (revert)
      6. Log results
      7. Repeat
    """
    cases = load_cases()
    config = load_config()

    print(f"Entroly Autotune -- {len(cases)} benchmark cases loaded")
    print(f"Time budget per case: {time_budget}s")

    print("\n--- Baseline evaluation ---")
    baseline = evaluate(config, cases, time_budget)
    baseline_score = composite_score(baseline)
    print(f"Baseline score: {baseline_score:.4f} "
          f"(recall={baseline.recall_accuracy:.3f}, "
          f"efficiency={baseline.context_efficiency:.6f}, "
          f"avg_ms={baseline.avg_wall_time_ms:.1f})")
    log_result(0, config, baseline, "keep", "baseline")

    if bench_only:
        print("\nPer-case breakdown:")
        for pc in baseline.per_case:
            print(f"  {pc['case_id']}: recall={pc.get('recall', 0):.2f}, "
                  f"tokens={pc.get('tokens_used', 0)}, "
                  f"wall_ms={pc.get('wall_ms', 0):.1f}")
        return

    best_score = baseline_score
    best_config = dict(config)
    defaults = dict(config)  # Original config for drift penalty
    improvements = 0

    # Polyak averaging state
    polyak_avg = dict(config)
    polyak_count = 1

    # EMA blending rate: scales with improvement magnitude
    # Base alpha = 0.3 (cautious). Doubles when improvement > 5% of score.
    ema_base_alpha = 0.3

    print(f"\n--- Starting {iterations} experiments (cautious updates) ---")
    print("(EMA blending + Polyak averaging + drift penalty)\n")

    for i in range(1, iterations + 1):
        candidate = mutate_config(best_config)
        mutated_param = candidate.pop("_mutated_param", "unknown")
        old_val = best_config.get(mutated_param)
        new_val = candidate.get(mutated_param)

        result = evaluate(candidate, cases, time_budget)
        score = composite_score(result, candidate, defaults)

        if score > best_score:
            # ── Cautious update: EMA blend instead of full replacement ──
            # Alpha scales with improvement magnitude: big jumps get faster
            # adoption, small improvements get cautious blending.
            delta_pct = (score - best_score) / max(best_score, 0.001)
            alpha = min(1.0, ema_base_alpha * (1.0 + delta_pct * 10.0))

            status = "keep"
            improvements += 1
            best_score = score
            best_config = _ema_blend(best_config, candidate, alpha)
            save_config(best_config)

            # Update Polyak average
            polyak_count += 1
            polyak_avg = _polyak_update(polyak_avg, best_config, polyak_count)

            marker = f">>> α={alpha:.2f}"
        elif (score == best_score and
              result.avg_wall_time_ms < baseline.avg_wall_time_ms):
            status = "keep"
            best_config = _ema_blend(best_config, candidate, ema_base_alpha * 0.5)
            save_config(best_config)
            marker = "  ="
        else:
            status = "discard"
            marker = "   "

        description = f"{mutated_param}: {old_val} -> {new_val}"
        log_result(i, candidate, result, status, description)

        print(f"{marker} [{i:04d}] score={score:.4f} "
              f"(recall={result.recall_accuracy:.3f}, "
              f"eff={result.context_efficiency:.6f}) "
              f"| {status:7s} | {description}")

    # ── Final: evaluate Polyak average ──
    if polyak_count > 2:
        print(f"\n--- Evaluating Polyak average ({polyak_count} samples) ---")
        polyak_result = evaluate(polyak_avg, cases, time_budget)
        polyak_score = composite_score(polyak_result, polyak_avg, defaults)
        print(f"Polyak score: {polyak_score:.4f} vs best: {best_score:.4f}")

        if polyak_score >= best_score:
            print("  → Polyak average is at least as good — using it")
            best_config = polyak_avg
            best_score = polyak_score
        else:
            print("  → Best single config is better — keeping it")

    print(f"\n--- Summary ---")
    print(f"Total experiments: {iterations}")
    print(f"Improvements found: {improvements}")
    print(f"Baseline score: {baseline_score:.4f}")
    print(f"Best score: {best_score:.4f}")
    delta_final = ((best_score - baseline_score) / max(baseline_score, 0.001)) * 100
    print(f"Improvement: {delta_final:.1f}%")
    print(f"Polyak samples: {polyak_count}")
    print(f"\nBest config saved to {CONFIG_PATH}")
    print(f"Full results log: {RESULTS_PATH}")

    save_config(best_config)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Entroly autonomous self-tuning loop")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--time-budget", type=float,
                        default=DEFAULT_TIME_BUDGET_SECS)
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    run_autotune(
        iterations=args.iterations,
        time_budget=args.time_budget,
        bench_only=args.bench_only,
    )


if __name__ == "__main__":
    main()
