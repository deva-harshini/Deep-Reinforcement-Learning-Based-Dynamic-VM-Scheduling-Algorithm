#!/usr/bin/env python3
"""
Phase 5 Master Orchestrator: Hyperparameter Ablation and Sensitivity Analysis
Title: Deep Reinforcement Learning-Based Dynamic VM Scheduling Algorithm for Energy-Efficient Cloud Datacenters

Executes:
1. Algorithmic Component Ablation Matrix (5 Seeds per configuration)
2. Hyperparameter Sensitivity Sweeps (5 Seeds per parameter value)
3. Rigorous Statistical Analysis (95% CI, Welch's t-test, Mann-Whitney U test, p-values)
4. Manuscript-Ready LaTeX Tables, CSVs, and Structured JSON Artifacts
"""

import os
import sys
import json
import time
import argparse
import subprocess
import datetime
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python_agent.phase5.ablation_engine import AblationEngine
from python_agent.phase5.sensitivity_sweep import SensitivitySweepEngine


def get_git_commit_hash():
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_latex_ablation_table(ablation_summary: dict, output_path: str):
    """Generates a publication-grade LaTeX table comparing ablation configurations."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation Study: Performance Impact of Core Algorithmic Components across 5 Random Seeds ($N=5$). $p$-values denote two-tailed Welch's $t$-test comparisons against the proposed \textbf{FULL\_MODEL}.}",
        r"\label{tab:ablation_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Configuration / Variant} & \textbf{Energy (kWh) $\downarrow$} & \textbf{SLAV ($\times 10^{-4}$) $\downarrow$} & \textbf{Migrations $\downarrow$} & \textbf{Cost Metric $J$ $\downarrow$} & \textbf{Conv. Step} & \textbf{$p$-value ($J$ vs Full)} \\",
        r"\midrule",
    ]

    for cfg_key, data in ablation_summary.items():
        name = cfg_key.replace("_", r"\_")
        e_mean, e_std = data["energy_kwh_mean"], data["energy_kwh_std"]
        slav_mean, slav_std = data["slav_mean"] * 1e4, data["slav_std"] * 1e4
        mig_mean = data["migrations_mean"]
        cost_mean, cost_std = data["composite_cost_mean"], data["composite_cost_std"]
        conv_mean = data["convergence_steps_mean"]
        p_val = data["p_value_cost"]

        p_str = "---" if cfg_key == "FULL_MODEL" else (r"\textbf{<0.001}" if p_val < 0.001 else f"{p_val:.4f}")

        if cfg_key == "FULL_MODEL":
            row = f"\\textbf{{{name}}} & \\textbf{{{e_mean:.3f} $\\pm$ {e_std:.3f}}} & \\textbf{{{slav_mean:.3f} $\\pm$ {slav_std:.3f}}} & \\textbf{{{mig_mean:.1f}}} & \\textbf{{{cost_mean:.3f} $\\pm$ {cost_std:.3f}}} & \\textbf{{{conv_mean:.0f}}} & {p_str} \\\\"
        else:
            row = f"{name} & {e_mean:.3f} $\\pm$ {e_std:.3f} & {slav_mean:.3f} $\\pm$ {slav_std:.3f} & {mig_mean:.1f} & {cost_mean:.3f} $\\pm$ {cost_std:.3f} & {conv_mean:.0f} & {p_str} \\\\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Generated LaTeX Ablation Table: {output_path}")


def generate_latex_sensitivity_table(sensitivity_summary: dict, output_path: str):
    """Generates a publication-grade LaTeX table of hyperparameter sensitivity sweeps."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Hyperparameter Sensitivity Analysis across 5 Random Seeds ($N=5$). Summary of energy consumption, SLA violations, and convergence rate across parameter search ranges.}",
        r"\label{tab:sensitivity_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Hyperparameter} & \textbf{Evaluated Range / Value} & \textbf{Energy (kWh) $\downarrow$} & \textbf{SLAV ($\times 10^{-4}$) $\downarrow$} & \textbf{Composite Cost $J$ $\downarrow$} & \textbf{Convergence Steps $\downarrow$} \\",
        r"\midrule",
    ]

    param_labels = {
        "learning_rate": "Learning Rate ($\\alpha$)",
        "discount_factor_gamma": "Discount Factor ($\\gamma$)",
        "entropy_coef": "Entropy Regularization ($c_{\\text{ent}}$)",
        "batch_size": "Mini-batch Size ($B$)",
        "reward_alpha_beta_ratio": "Reward Ratio ($\\alpha_{\\text{energy}} / \\beta_{\\text{SLA}}$)",
    }

    for param_key, values_dict in sensitivity_summary.items():
        param_label = param_labels.get(param_key, param_key)
        first_row = True

        # Find best performing value for bolding
        best_val = min(values_dict.keys(), key=lambda v: values_dict[v]["composite_cost_mean"])

        for val_str, stats in values_dict.items():
            prefix = param_label if first_row else ""
            first_row = False

            e_mean, e_std = stats["energy_kwh_mean"], stats["energy_kwh_std"]
            slav_mean, slav_std = stats["slav_mean"] * 1e4, stats["slav_std"] * 1e4
            cost_mean, cost_std = stats["composite_cost_mean"], stats["composite_cost_std"]
            conv_mean = stats["convergence_steps_mean"]

            if val_str == best_val:
                row = f"\\textbf{{{prefix}}} & \\textbf{{{val_str}}} & \\textbf{{{e_mean:.3f} $\\pm$ {e_std:.3f}}} & \\textbf{{{slav_mean:.3f} $\\pm$ {slav_std:.3f}}} & \\textbf{{{cost_mean:.3f} $\\pm$ {cost_std:.3f}}} & \\textbf{{{conv_mean:.0f}}} \\\\"
            else:
                row = f"{prefix} & {val_str} & {e_mean:.3f} $\\pm$ {e_std:.3f} & {slav_mean:.3f} $\\pm$ {slav_std:.3f} & {cost_mean:.3f} $\\pm$ {cost_std:.3f} & {conv_mean:.0f} \\\\"
            lines.append(row)
        lines.append(r"\midrule")

    # Remove the last midrule and add bottomrule
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Generated LaTeX Sensitivity Table: {output_path}")


def export_csv_data(ablation_raw: dict, sensitivity_raw: dict, output_dir: str):
    """Exports raw trials to cleanly structured CSV files."""
    # 1. Ablation CSV
    ablation_csv_path = os.path.join(output_dir, "ablation_results.csv")
    with open(ablation_csv_path, "w") as f:
        f.write("config,seed,energy_kwh,avg_power_w,slata,pdm,slav,migrations,composite_cost_j,mean_reward,convergence_steps\n")
        for cfg, runs in ablation_raw.items():
            for r in runs:
                f.write(f"{cfg},{r['seed']},{r['energy_kwh']:.6f},{r['avg_power_w']:.4f},{r['slata']:.6f},{r['pdm']:.6f},{r['slav']:.8f},{r['migrations']},{r['composite_cost_j']:.6f},{r['mean_reward']:.4f},{r['convergence_steps']}\n")
    print(f"✅ Exported Ablation Raw CSV: {ablation_csv_path}")

    # 2. Sensitivity CSV
    sensitivity_csv_path = os.path.join(output_dir, "sensitivity_results.csv")
    with open(sensitivity_csv_path, "w") as f:
        f.write("param_name,param_value,seed,energy_kwh,avg_power_w,slata,pdm,slav,migrations,composite_cost_j,convergence_steps\n")
        for param, val_dict in sensitivity_raw.items():
            for val, runs in val_dict.items():
                for r in runs:
                    f.write(f"{param},{val},{r['seed']},{r['energy_kwh']:.6f},{r['avg_power_w']:.4f},{r['slata']:.6f},{r['pdm']:.6f},{r['slav']:.8f},{r['migrations']},{r['composite_cost_j']:.6f},{r['convergence_steps']}\n")
    print(f"✅ Exported Sensitivity Raw CSV: {sensitivity_csv_path}")


def print_markdown_ablation_summary(ablation_summary: dict):
    print("\n" + "=" * 115)
    print(" 📊 PHASE 5: ABLATION MATRIX STATISTICAL SUMMARY (N = 5 SEEDS)")
    print("=" * 115)
    print(f"{'Configuration':<24} | {'Energy (kWh)':<15} | {'SLAV (x10^-4)':<15} | {'Migrations':<10} | {'Cost J':<15} | {'p-value (Cost)'}")
    print("-" * 115)
    for cfg, stats in ablation_summary.items():
        e_str = f"{stats['energy_kwh_mean']:.3f} ± {stats['energy_kwh_std']:.3f}"
        slav_str = f"{stats['slav_mean']*1e4:.3f} ± {stats['slav_std']*1e4:.3f}"
        mig_str = f"{stats['migrations_mean']:.1f}"
        cost_str = f"{stats['composite_cost_mean']:.3f} ± {stats['composite_cost_std']:.3f}"
        p_val = stats["p_value_cost"]
        p_str = "Baseline (1.0)" if cfg == "FULL_MODEL" else (f"<0.001 (***)" if p_val < 0.001 else f"{p_val:.4f}")
        print(f"{cfg:<24} | {e_str:<15} | {slav_str:<15} | {mig_str:<10} | {cost_str:<15} | {p_str}")
    print("=" * 115)


def print_markdown_sensitivity_summary(sensitivity_summary: dict):
    print("\n" + "=" * 115)
    print(" 🔍 PHASE 5: HYPERPARAMETER SENSITIVITY SWEEP SUMMARY (N = 5 SEEDS)")
    print("=" * 115)
    print(f"{'Hyperparameter':<25} | {'Param Value':<12} | {'Energy (kWh)':<15} | {'SLAV (x10^-4)':<15} | {'Cost J':<15} | {'Conv. Steps'}")
    print("-" * 115)
    for param, val_dict in sensitivity_summary.items():
        for val, stats in val_dict.items():
            e_str = f"{stats['energy_kwh_mean']:.3f} ± {stats['energy_kwh_std']:.3f}"
            slav_str = f"{stats['slav_mean']*1e4:.3f} ± {stats['slav_std']*1e4:.3f}"
            cost_str = f"{stats['composite_cost_mean']:.3f} ± {stats['composite_cost_std']:.3f}"
            conv_str = f"{stats['convergence_steps_mean']:.0f}"
            print(f"{param:<25} | {val:<12} | {e_str:<15} | {slav_str:<15} | {cost_str:<15} | {conv_str}")
        print("-" * 115)
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Hyperparameter Ablation & Sensitivity Analysis")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per evaluation")
    parser.add_argument("--steps", type=int, default=500, help="Timesteps per evaluation episode")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "results", "phase5")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(logs_dir, f"phase5_run_{timestamp}.log")

    seeds_list = [42, 101, 2024, 777, 999][:args.seeds]

    print("=" * 80)
    print(" 🚀 Starting Phase 5: Hyperparameter Ablation & Sensitivity Analysis Pipeline")
    print(f" Timestamp:      {datetime.datetime.now().isoformat()}")
    print(f" Git Commit:     {get_git_commit_hash()}")
    print(f" Random Seeds:   {seeds_list} (Count: {len(seeds_list)})")
    print(f" Steps/Episode:  {args.steps}")
    print(f" Results Output: {output_dir}")
    print("=" * 80)

    # 1. Execute Ablation Matrix
    print("\n▶ [STEP 1/3] Executing Ablation Matrix across 5 Configurations...")
    ablation_engine = AblationEngine(seeds=seeds_list, eval_steps=args.steps)
    start_ablation = time.perf_counter()

    ablation_res = ablation_engine.run_all(
        progress_callback=lambda cfg, s, n_s, curr, tot: print(f"  [Ablation {curr:02d}/{tot:02d}] {cfg:<22} | Seed {s}/{n_s} ({seeds_list[s-1]})")
    )
    ablation_time = time.perf_counter() - start_ablation
    print(f"✔ Completed Ablation Matrix in {ablation_time:.2f}s")

    # 2. Execute Sensitivity Sweeps
    print("\n▶ [STEP 2/3] Executing Hyperparameter Sensitivity Sweeps across 5 Search Spaces...")
    sensitivity_engine = SensitivitySweepEngine(seeds=seeds_list, eval_steps=args.steps)
    start_sens = time.perf_counter()

    sensitivity_res = sensitivity_engine.run_all(
        progress_callback=lambda p, v, s, n_s, curr, tot: print(f"  [Sweep {curr:03d}/{tot:03d}] {p:<22} = {v:<8} | Seed {s}/{n_s}")
    )
    sens_time = time.perf_counter() - start_sens
    print(f"✔ Completed Sensitivity Sweeps in {sens_time:.2f}s")

    # 3. Print Summaries & Export Artifacts
    print("\n▶ [STEP 3/3] Generating Statistical Reports & Publication Artifacts...")
    print_markdown_ablation_summary(ablation_res["summary"])
    print_markdown_sensitivity_summary(sensitivity_res["summary"])

    # Export LaTeX tables
    latex_ablation_path = os.path.join(output_dir, "table_ablation_results.tex")
    generate_latex_ablation_table(ablation_res["summary"], latex_ablation_path)

    latex_sensitivity_path = os.path.join(output_dir, "table_hyperparameter_sensitivity.tex")
    generate_latex_sensitivity_table(sensitivity_res["summary"], latex_sensitivity_path)

    # Export CSVs
    export_csv_data(ablation_res["raw"], sensitivity_res["raw"], output_dir)

    # Export Full JSON Master Summary
    master_summary = {
        "metadata": {
            "pipeline": "Phase 5 Hyperparameter Ablation and Sensitivity Analysis",
            "paper_title": "Deep Reinforcement Learning-Based Dynamic VM Scheduling Algorithm for Energy-Efficient Cloud Datacenters",
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": get_git_commit_hash(),
            "random_seeds": seeds_list,
            "steps_per_episode": args.steps,
            "ablation_runtime_sec": round(ablation_time, 2),
            "sensitivity_runtime_sec": round(sens_time, 2),
        },
        "ablation_analysis": ablation_res["summary"],
        "sensitivity_analysis": sensitivity_res["summary"],
    }

    json_path = os.path.join(output_dir, "ablation_sensitivity_summary.json")
    with open(json_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    print(f"✅ Saved Master JSON Artifact: {json_path}")

    # Write log file
    with open(log_file_path, "w") as f:
        f.write(f"Phase 5 Execution Log - {datetime.datetime.now().isoformat()}\n")
        f.write(f"Git Commit: {get_git_commit_hash()}\n")
        f.write(f"Seeds: {seeds_list}\n")
        f.write(f"Ablation Runtime: {ablation_time:.2f}s\n")
        f.write(f"Sensitivity Runtime: {sens_time:.2f}s\n")
        f.write("\nAblation Summary:\n")
        f.write(json.dumps(ablation_res["summary"], indent=2))
        f.write("\n\nSensitivity Summary:\n")
        f.write(json.dumps(sensitivity_res["summary"], indent=2))
    print(f"✅ Saved Execution Log: {log_file_path}")

    print("\n" + "=" * 80)
    print(" 🎉 PHASE 5 EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" Total Runtime: {ablation_time + sens_time:.2f} seconds")
    print(f" Artifacts Location: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
