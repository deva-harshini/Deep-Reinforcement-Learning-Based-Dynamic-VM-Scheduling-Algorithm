#!/usr/bin/env python3
"""
Phase 2: DRL Agent Implementation & Training (PPO / MaskablePPO)
Trains a Deep Reinforcement Learning agent for energy-efficient VM scheduling in CloudSimPlus.
Logs to TensorBoard and saves checkpoints to models/ppo_vm_scheduler/.
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
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class DatacenterMetricsCallback(BaseCallback):
    """Custom callback for logging CloudSimPlus datacenter telemetry to TensorBoard and console."""

    def __init__(self, check_freq: int = 500, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_powers = []
        self.episode_slas = []
        self.episode_migrations = []
        self.episode_rewards = []
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _on_step(self) -> bool:
        # Extract info from step
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        if rewards is not None and len(rewards) > 0:
            self.episode_rewards.append(float(rewards[0]))

        if infos and len(infos) > 0:
            info = infos[0]
            if "power_watts" in info:
                self.episode_powers.append(info["power_watts"])
            if "sla_violations" in info:
                self.episode_slas.append(info["sla_violations"])
            if "migrations" in info:
                self.episode_migrations.append(info["migrations"])

        # Periodic logging
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-self.check_freq:]) if self.episode_rewards else 0.0
            mean_power = np.mean(self.episode_powers[-self.check_freq:]) if self.episode_powers else 0.0
            total_sla = sum(self.episode_slas[-self.check_freq:]) if self.episode_slas else 0
            total_migrations = sum(self.episode_migrations[-self.check_freq:]) if self.episode_migrations else 0

            # Log to TensorBoard
            self.logger.record("datacenter/mean_reward", mean_reward)
            self.logger.record("datacenter/mean_power_watts", mean_power)
            self.logger.record("datacenter/total_sla_violations", total_sla)
            self.logger.record("datacenter/total_vm_migrations", total_migrations)

            elapsed = time.perf_counter() - self.start_time
            fps = self.n_calls / elapsed if elapsed > 0 else 0

            if self.verbose > 0:
                print(
                    f"Step {self.n_calls:5d} / {self.locals.get('total_timesteps', 10000)} | "
                    f"Mean Reward: {mean_reward:6.2f} | "
                    f"Avg Power: {mean_power:7.2f} W | "
                    f"SLA Violations: {total_sla:2d} | "
                    f"Migrations: {total_migrations:3d} | "
                    f"FPS: {fps:5.1f}"
                )

        return True


def train(
    total_timesteps: int = 10000,
    server_endpoint: str = "tcp://localhost:5555",
    log_dir: str = "logs/tensorboard",
    model_dir: str = "models/ppo_vm_scheduler",
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
):
    print("=" * 70)
    print(" Phase 2: DRL Agent Implementation & Training (MaskablePPO)")
    print("=" * 70)
    print(f"Target Server:    {server_endpoint}")
    print(f"Total Timesteps:  {total_timesteps:,}")
    print(f"Model Directory:  {model_dir}")
    print(f"TensorBoard Logs: {log_dir}")
    print("-" * 70)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize Environment
    env = CloudSimEnv(server_endpoint=server_endpoint)

    # Configure Checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=2500,
        save_path=model_dir,
        name_prefix="ppo_vm_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    metrics_callback = DatacenterMetricsCallback(check_freq=500, verbose=1)

    # Instantiate MaskablePPO Agent
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=0,
    )

    print(f"[TRAIN] Starting MaskablePPO training for {total_timesteps:,} steps...")
    start_time = time.perf_counter()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=False,
    )

    total_time = time.perf_counter() - start_time
    print("-" * 70)
    print(f"✅ Training completed in {total_time:.2f}s ({total_timesteps / total_time:.1f} steps/sec)")

    # Save Final Model Artifact
    final_model_path = os.path.join(model_dir, "ppo_vm_final.zip")
    model.save(final_model_path)
    print(f"✅ Saved final model to: {final_model_path}")
    print("=" * 70)

    env.close()
    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL Agent for Dynamic VM Scheduling")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--server", type=str, default="tcp://localhost:5555", help="Java server endpoint")
    args = parser.parse_args()

    train(total_timesteps=args.timesteps, server_endpoint=args.server)
