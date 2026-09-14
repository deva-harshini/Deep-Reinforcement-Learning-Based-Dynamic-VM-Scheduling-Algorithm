#!/usr/bin/env python3
"""
Phase 2 Evaluation & Benchmarking:
Compares Trained PPO Agent vs Random Policy vs First-Fit Heuristic Baseline
over 200 interaction steps on CloudSimPlus datacenter simulation.
Logs power consumption (W), SLA violations, total VM migrations, and reward per step.
"""

import os
import sys
import argparse
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python_agent.env.cloudsim_env import CloudSimEnv
from sb3_contrib import MaskablePPO


class FirstFitPolicy:
    """First-Fit Consolidation Heuristic: packs VMs onto lowest-indexed available hosts."""

    def __init__(self, num_hosts: int = 10, num_vms: int = 20):
        self.num_hosts = num_hosts
        self.num_vms = num_vms

    def predict(self, observation: np.ndarray, action_masks: np.ndarray = None) -> np.ndarray:
        # Action is array of target host IDs for each VM
        # In First-Fit, assign VM i to first host that is not full
        action = np.zeros(self.num_vms, dtype=np.int64)
        ram_util = observation[self.num_hosts : 2 * self.num_hosts]

        # Target packing onto first few hosts (e.g., hosts 0, 1, 2, 3...)
        for vm in range(self.num_vms):
            assigned = False
            for h in range(self.num_hosts):
                if ram_util[h] < 0.85:
                    action[vm] = h
                    assigned = True
                    break
            if not assigned:
                action[vm] = 0
        return action


class RandomPolicy:
    """Random Uniform Action Selection Baseline."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation: np.ndarray, action_masks: np.ndarray = None) -> np.ndarray:
        return self.action_space.sample()


def run_evaluation_episode(env: CloudSimEnv, policy, policy_name: str, num_steps: int = 200) -> dict:
    print(f"\n[EVAL] Running {policy_name} over {num_steps} steps...")
    obs, info = env.reset()

    powers = []
    sla_violations = []
    migrations = []
    rewards = []
    step_latencies = []

    start_time = time.perf_counter()

    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()

        if hasattr(policy, "predict"):
            if isinstance(policy, MaskablePPO):
                action_masks = env.action_masks()
                action, _ = policy.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action = policy.predict(obs)
        else:
            action = env.action_space.sample()

        t1 = time.perf_counter()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        t2 = time.perf_counter()

        step_latencies.append((t2 - t0) * 1000.0)
        rewards.append(reward)
        powers.append(step_info.get("power_watts", 0.0))
        sla_violations.append(step_info.get("sla_violations", 0))
        migrations.append(step_info.get("migrations", 0))

        obs = next_obs

    total_time = time.perf_counter() - start_time
    total_energy_kwh = (np.sum(powers) * 10.0) / (3600.0 * 1000.0)  # 10s per step

    results = {
        "policy": policy_name,
        "steps": num_steps,
        "total_time_s": total_time,
        "fps": num_steps / total_time if total_time > 0 else 0,
        "avg_latency_ms": np.mean(step_latencies),
        "avg_power_w": np.mean(powers),
        "total_energy_kwh": total_energy_kwh,
        "total_sla": sum(sla_violations),
        "total_migrations": sum(migrations),
        "mean_reward": np.mean(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
    }

    return results


def evaluate(
    model_path: str = "models/ppo_vm_scheduler/ppo_vm_final.zip",
    server_endpoint: str = "tcp://localhost:5555",
    num_steps: int = 200,
):
    print("=" * 80)
    print(" Phase 2: Dynamic VM Scheduling Policy Evaluation & Benchmarking")
    print("=" * 80)
    print(f"Target Server:   {server_endpoint}")
    print(f"Model Path:      {model_path}")
    print(f"Steps per Trial: {num_steps}")
    print("-" * 80)

    env = CloudSimEnv(server_endpoint=server_endpoint)

    # 1. Load Trained PPO Agent
    if os.path.exists(model_path):
        print(f"[EVAL] Loading trained MaskablePPO model from {model_path}...")
        ppo_model = MaskablePPO.load(model_path, env=env)
    else:
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")

    # 2. Instantiate Baselines
    random_policy = RandomPolicy(env.action_space)
    first_fit_policy = FirstFitPolicy(num_hosts=env.num_hosts, num_vms=env.num_vms)

    # 3. Execute Benchmarking Runs
    res_ppo = run_evaluation_episode(env, ppo_model, "Trained PPO (DRL)", num_steps=num_steps)
    res_first_fit = run_evaluation_episode(env, first_fit_policy, "First-Fit Heuristic", num_steps=num_steps)
    res_random = run_evaluation_episode(env, random_policy, "Random Baseline", num_steps=num_steps)

    # 4. Display Formatted Results Table
    print("\n" + "=" * 80)
    print(" COMPARATIVE PERFORMANCE EVALUATION MATRIX")
    print("=" * 80)
    header = f"{'Policy / Algorithm':<22} | {'Avg Power (W)':<14} | {'Total SLA':<10} | {'Migrations':<11} | {'Mean Reward':<12}"
    print(header)
    print("-" * 80)

    all_res = [res_ppo, res_first_fit, res_random]
    for r in all_res:
        row = (
            f"{r['policy']:<22} | "
            f"{r['avg_power_w']:>10.2f} W  | "
            f"{r['total_sla']:>8d}   | "
            f"{r['total_migrations']:>9d}   | "
            f"{r['mean_reward']:>10.3f}"
        )
        print(row)

    print("-" * 80)

    # Calculate Improvements
    power_vs_rand = ((res_random['avg_power_w'] - res_ppo['avg_power_w']) / res_random['avg_power_w']) * 100
    power_vs_ff = ((res_first_fit['avg_power_w'] - res_ppo['avg_power_w']) / res_first_fit['avg_power_w']) * 100
    reward_vs_rand = res_ppo['mean_reward'] - res_random['mean_reward']
    reward_vs_ff = res_ppo['mean_reward'] - res_first_fit['mean_reward']

    print(f"📊 DRL Improvement Highlights:")
    print(f"   • Energy Reduction vs Random:    {power_vs_rand:+.1f}% ({res_ppo['avg_power_w']:.1f}W vs {res_random['avg_power_w']:.1f}W)")
    print(f"   • Energy Reduction vs First-Fit: {power_vs_ff:+.1f}% ({res_ppo['avg_power_w']:.1f}W vs {res_first_fit['avg_power_w']:.1f}W)")
    print(f"   • Reward Advantage vs Random:    {reward_vs_rand:+.3f} points per step")
    print(f"   • Reward Advantage vs First-Fit: {reward_vs_ff:+.3f} points per step")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DRL Agent vs Baselines")
    parser.add_argument("--model", type=str, default="models/ppo_vm_scheduler/ppo_vm_final.zip", help="Model path")
    parser.add_argument("--steps", type=int, default=200, help="Evaluation steps")
    parser.add_argument("--server", type=str, default="tcp://localhost:5555", help="Java server endpoint")
    args = parser.parse_args()

    evaluate(model_path=args.model, server_endpoint=args.server, num_steps=args.steps)
