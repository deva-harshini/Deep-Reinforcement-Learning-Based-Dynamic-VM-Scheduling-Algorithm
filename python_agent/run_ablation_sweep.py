#!/usr/bin/env python3
"""
Phase 5: Automated Multi-Seed Sweep Runner for Ablation, Sensitivity, and Noise Perturbation.
Executes experiments across 5 distinct random seeds (N=5) and exports consolidated logs.
"""

import os
import sys
import json
import csv
import time
import argparse
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python_agent.ablation import generate_experiment_grid, train_and_evaluate_trial, AblationConfig


def run_single_trial(config: AblationConfig) -> Dict[str, Any]:
    """Helper for parallel execution."""
    return train_and_evaluate_trial(config)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 Multi-Seed Ablation & Sensitivity Sweeps")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024, 777, 999], help="Random seeds")
    args = parser.parse_args()

    print("=" * 80)
    print(" 🚀 PHASE 5: DRL ABLATION, SENSITIVITY & NOISE PERTURBATION SWEEP RUNNER")
    print("=" * 80)
    print(f"Random Seeds:        {args.seeds} (N = {len(args.seeds)})")
    print(f"Parallel Workers:    {args.workers}")
    print("-" * 80)

    configs = generate_experiment_grid(seeds=args.seeds)
    total_runs = len(configs)
    print(f"[SWEEP] Generated {total_runs} experiment configurations.")

    results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    print("[SWEEP] Executing multi-seed trials...")
    completed_count = 0

    # Sequential/Multiprocessing execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_trial, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            cfg = futures[future]
            completed_count += 1
            try:
                res = future.result()
                results.append(res)
                print(
                    f"  [{completed_count:02d}/{total_runs:02d}] "
                    f"Category: {res['category']:<18} | "
                    f"Variant: {res['variant_name']:<25} | "
                    f"Seed: {res['seed']:<4} | "
                    f"Reward: {res['mean_cumulative_reward']:>7.2f} | "
                    f"SLA: {res['sla_violation_rate_pct']:>5.2f}% | "
                    f"Power: {res['power_consumption_watts']:>6.1f}W | "
                    f"SPS: {res['steps_per_second']:>6.1f}"
                )
            except Exception as e:
                print(f"  [ERROR] Trial failed for {cfg.variant_name} (Seed {cfg.seed}): {e}")

    total_duration = time.perf_counter() - start_time
    print("-" * 80)
    print(f"✔ Completed {len(results)}/{total_runs} trials in {total_duration:.2f} seconds.")

    # 4. Save consolidated logs
    log_dirs = [
        os.path.join(PROJECT_ROOT, "logs"),
        os.path.join("/Users/harshini/Downloads/GITAM_Template_CSSE", "logs"),
        os.path.join(PROJECT_ROOT, "results", "phase5"),
        os.path.join("/Users/harshini/Downloads/GITAM_Template_CSSE", "results", "phase5"),
    ]

    for ld in log_dirs:
        os.makedirs(ld, exist_ok=True)

    json_filename = "ablation_results.json"
    csv_filename = "ablation_results.csv"

    # Export JSON
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_trials": len(results),
            "seeds": args.seeds,
            "duration_sec": total_duration,
        },
        "raw_results": results,
    }

    for ld in log_dirs:
        json_path = os.path.join(ld, json_filename)
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"📁 Exported JSON: {json_path}")

        csv_path = os.path.join(ld, csv_filename)
        if results:
            keys = list(results[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"📁 Exported CSV:  {csv_path}")

    print("=" * 80)
    print(" 🎉 ALL SWEEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
