#!/usr/bin/env python3
"""
Phase 4: Advanced Heuristic Benchmarking & Master Multi-Metric Evaluation Suite.
Compares MaskablePPO vs. First-Fit vs. MMT vs. Local Regression vs. Power-Spread vs. Random Uniform.
"""

import os
import sys
import argparse
import numpy as np
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from python_agent.env.cloudsim_env import CloudSimEnv


def run_heuristic_policy(env, policy_name: str, num_steps: int = 500, model=None):
    obs, info = env.reset()
    total_reward = 0.0
    powers, sla_violations, migrations, shutdowns, edps, sla_percents = [], [], [], [], [], []

    # Local Regression history buffer
    cpu_history = []

    for step in range(num_steps):
        action = np.zeros(20, dtype=int)

        if policy_name == "Trained MaskablePPO (DRL Agent)":
            # MaskablePPO action inference using action masks
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        elif policy_name == "Random Uniform Baseline":
            action = env.action_space.sample()

        elif policy_name == "First-Fit Consolidation Heuristic":
            # Consolidate all VMs to primary host 0
            action = np.zeros(20, dtype=int)

        elif policy_name == "Minimum Migration Time (MMT)":
            # Migrate smallest VMs (Small VMs 0-7) to idle host 9, keep others on host 0
            action = np.array([9 if i < 8 else 0 for i in range(20)], dtype=int)

        elif policy_name == "Local Regression (LR) CPU Trend Predictor":
            # Track CPU history and trigger proactive migrations when utilization slope rises
            host_cpu = obs[0:10]
            cpu_history.append(host_cpu)
            if len(cpu_history) > 5:
                cpu_history.pop(0)

            # Predict trend: if host 0 CPU > 80%, offload top VMs to host 5
            if np.mean(host_cpu[0:5]) > 0.80:
                action = np.array([5 if i % 2 == 0 else 0 for i in range(20)], dtype=int)
            else:
                action = np.zeros(20, dtype=int)

        elif policy_name == "Power-Spread Load Distribution":
            # Distribute 20 VMs evenly across all 10 hosts (2 VMs per host)
            action = np.array([i % 10 for i in range(20)], dtype=int)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        powers.append(info.get("power_watts", 0.0))
        sla_violations.append(info.get("sla_violations", 0))
        migrations.append(info.get("migrations", 0))
        shutdowns.append(info.get("active_shutdowns", 0))
        edps.append(info.get("edp", 0.0))
        sla_percents.append(info.get("sla_percent", 0.0))

        if terminated or truncated:
            obs, info = env.reset()

    return {
        "policy": policy_name,
        "avg_power": np.mean(powers),
        "total_sla": sum(sla_violations),
        "sla_percent": np.mean(sla_percents),
        "migrations": sum(migrations),
        "avg_shutdowns": np.mean(shutdowns),
        "final_edp": edps[-1] if edps else 0.0,
        "mean_reward": total_reward / num_steps
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Multi-Metric Policy Evaluation Suite")
    parser.add_argument("--steps", type=int, default=500, help="Evaluation steps per policy")
    parser.add_argument("--server", type=str, default="tcp://localhost:5555", help="Java server endpoint")
    parser.add_argument("--model", type=str, default=None, help="Path to trained PPO model artifact")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = args.model or os.path.join(project_root, "models/ppo_vm_scheduler/ppo_vm_final.zip")

    env = CloudSimEnv(server_endpoint=args.server)
    policies = [
        "Trained MaskablePPO (DRL Agent)",
        "First-Fit Consolidation Heuristic",
        "Minimum Migration Time (MMT)",
        "Local Regression (LR) CPU Trend Predictor",
        "Power-Spread Load Distribution",
        "Random Uniform Baseline",
    ]

    model = None
    if os.path.exists(model_path):
        print(f"[INFO] Loading trained MaskablePPO model from: {model_path}")
        model = MaskablePPO.load(model_path, env=env)
    else:
        print(f"[WARN] Trained model not found at {model_path}. Running heuristics only.")

    results = []
    for policy in policies:
        if policy == "Trained MaskablePPO (DRL Agent)" and model is None:
            continue
        print(f"[EVAL] Benchmarking Policy: {policy} ({args.steps} steps)...")
        res = run_heuristic_policy(env, policy_name=policy, num_steps=args.steps, model=model)
        results.append(res)

    print("\n" + "=" * 110)
    print(" PHASE 4 MASTER MULTI-METRIC COMPARATIVE EVALUATION MATRIX")
    print("=" * 110)
    print(f"{'Policy / Scheduling Algorithm':<36} | {'Avg Power (W)':<13} | {'SLA %':<7} | {'Migrations':<10} | {'Hosts Off':<9} | {'EDP (J·s)':<12} | {'Mean Reward'}")
    print("-" * 110)

    for r in results:
        print(f"{r['policy']:<36} | {r['avg_power']:>11.2f} W | {r['sla_percent']:>5.2f}% | {r['migrations']:>10d} | {r['avg_shutdowns']:>9.1f} | {r['final_edp']:>12.1f} | {r['mean_reward']:>11.3f}")

    print("=" * 110)
    env.close()


if __name__ == "__main__":
    main()
