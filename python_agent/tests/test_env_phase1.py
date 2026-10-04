#!/usr/bin/env python3
"""
Phase 1 Integration Test: CloudSimPlus Gymnasium Environment Verification
Validates MDP observation space, action space, step/reset lifecycle, reward calculation,
and dynamic power modeling over 100 interaction steps.
"""

import sys
import os
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from python_agent.env.cloudsim_env import CloudSimEnv


def test_cloudsim_gymnasium_env(num_steps: int = 100):
    print("=" * 70)
    print(" PHASE 1 INTEGRATION TEST: CloudSimPlus Gymnasium Environment")
    print("=" * 70)

    env = CloudSimEnv(server_endpoint="tcp://localhost:5555")
    print(f"[TEST] Environment initialized successfully.")
    print(f"[TEST] Observation Space: {env.observation_space}")
    print(f"[TEST] Action Space:      {env.action_space}")
    print("-" * 70)

    # 1. Test Reset
    print("[TEST] Executing env.reset()...")
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray), f"Expected np.ndarray, got {type(obs)}"
    assert obs.dtype == np.float32, f"Expected float32 dtype, got {obs.dtype}"
    assert obs.shape == (40,), f"Expected shape (40,), got {obs.shape}"
    assert np.all((obs >= 0.0) & (obs <= 1.0)), f"Observation values out of bounds: {obs}"
    assert isinstance(info, dict), f"Expected dict info, got {type(info)}"

    print(f"✅ Reset passed! Initial Observation shape: {obs.shape}, values bounded [0.0, 1.0]")
    print(f"   Initial Info: {info}")
    print("-" * 70)

    # 2. Run Step Loop
    print(f"[TEST] Executing {num_steps}-step loop with random action sampling...")
    rewards = []
    powers = []
    sla_violations = []
    migrations_list = []
    step_latencies = []

    start_time = time.perf_counter()

    for step in range(1, num_steps + 1):
        action = env.action_space.sample()
        assert action.shape == (20,), f"Sampled action shape mismatch: {action.shape}"

        t0 = time.perf_counter()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        t1 = time.perf_counter()

        step_latencies.append((t1 - t0) * 1000.0)

        # Assertions
        assert isinstance(next_obs, np.ndarray), f"Step {step}: next_obs is not np.ndarray"
        assert next_obs.shape == (40,), f"Step {step}: next_obs shape is {next_obs.shape}"
        assert next_obs.dtype == np.float32, f"Step {step}: next_obs dtype is {next_obs.dtype}"
        assert np.all((next_obs >= 0.0) & (next_obs <= 1.0)), f"Step {step}: Out-of-bounds observation: {next_obs}"
        assert isinstance(reward, float), f"Step {step}: Reward is not float ({type(reward)})"
        assert isinstance(terminated, bool), f"Step {step}: Terminated is not bool"
        assert isinstance(truncated, bool), f"Step {step}: Truncated is not bool"
        assert "power_watts" in step_info, f"Step {step}: Missing 'power_watts' in info"
        assert "sla_violations" in step_info, f"Step {step}: Missing 'sla_violations' in info"
        assert "migrations" in step_info, f"Step {step}: Missing 'migrations' in info"

        rewards.append(reward)
        powers.append(step_info["power_watts"])
        sla_violations.append(step_info["sla_violations"])
        migrations_list.append(step_info["migrations"])

        if step == 1 or step % 20 == 0 or step == num_steps:
            print(
                f"Step {step:3d} | Power: {step_info['power_watts']:7.2f} W | "
                f"SLA Violations: {step_info['sla_violations']:2d} | "
                f"Migrations: {step_info['migrations']:2d} | "
                f"Reward: {reward:6.2f} | "
                f"Latency: {step_latencies[-1]:.2f} ms"
            )

    total_time = time.perf_counter() - start_time
    throughput = num_steps / total_time

    # 3. Test Reset after Steps
    print("-" * 70)
    print("[TEST] Testing post-run reset...")
    reset_obs, reset_info = env.reset()
    assert reset_obs.shape == (40,)
    print("✅ Post-run reset successful!")

    # 4. Summary & Verification
    print("=" * 70)
    print(" PHASE 1 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"Total Steps Executed:   {num_steps}")
    print(f"Total Wall-Clock Time: {total_time:.3f} s")
    print(f"Step Throughput Rate:   {throughput:.1f} steps/sec")
    print(f"Avg Step Latency:       {np.mean(step_latencies):.2f} ms")
    print(f"Avg Power Draw:         {np.mean(powers):.2f} W (Min: {np.min(powers):.2f} W, Max: {np.max(powers):.2f} W)")
    print(f"Total SLA Violations:   {sum(sla_violations)}")
    print(f"Total VM Migrations:    {sum(migrations_list)}")
    print(f"Avg Reward per Step:    {np.mean(rewards):.3f} (Min: {np.min(rewards):.3f}, Max: {np.max(rewards):.3f})")
    print("-" * 70)

    # Dynamic behavior check
    power_dynamic = np.std(powers) > 0.0 or (np.max(powers) != np.min(powers))
    reward_dynamic = np.std(rewards) > 0.0

    assert power_dynamic, "Power draw did not vary dynamically across steps!"
    assert reward_dynamic, "Rewards did not vary dynamically across steps!"

    print("✅ ALL PHASE 1 CHECKS PASSED: MDP spaces, dynamic power model, and Gymnasium API verified!")
    print("=" * 70)

    env.close()
    return True


if __name__ == "__main__":
    steps = 100
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
    success = test_cloudsim_gymnasium_env(num_steps=steps)
    sys.exit(0 if success else 1)
