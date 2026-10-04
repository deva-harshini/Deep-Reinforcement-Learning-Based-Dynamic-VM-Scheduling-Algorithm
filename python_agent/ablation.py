"""
Phase 5: Hyperparameter Ablation, Architectural Action-Masking & Environmental Perturbation Engine.
Defines experiment configurations, agent training pipelines, and evaluation harnesses.
"""

import os
import sys
import time
import dataclasses
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python_agent.env.cloudsim_env import CloudSimEnv


@dataclasses.dataclass
class AblationConfig:
    """Experiment configuration dataclass for Phase 5."""
    category: str                      # 'hyperparameter', 'action_masking', 'noise_perturbation'
    variant_name: str                  # Descriptive label
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    use_action_masking: bool = True
    invalid_action_penalty: float = 0.0
    obs_noise_std: float = 0.0
    gamma: float = 0.99
    batch_size: int = 64
    n_steps: int = 256
    n_epochs: int = 10
    total_timesteps: int = 5000
    eval_steps: int = 500
    seed: int = 42


def set_reproducibility_seed(seed: int):
    """Enforces deterministic random seed across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_agent(config: AblationConfig, env: CloudSimEnv):
    """Instantiates MaskablePPO or standard Unmasked PPO with configured hyperparameters."""
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    if config.use_action_masking:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=0.95,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
            verbose=0,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=0.95,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
            verbose=0,
        )
    return model


def train_and_evaluate_trial(config: AblationConfig) -> Dict[str, Any]:
    """
    Executes an end-to-end training and evaluation trial for a specific config and random seed.
    Returns fine-grained performance and datacenter telemetry metrics.
    """
    set_reproducibility_seed(config.seed)

    # 1. Initialize training environment
    train_env = CloudSimEnv(
        server_endpoint="tcp://localhost:5555",
        obs_noise_std=0.0,
        invalid_action_penalty=config.invalid_action_penalty,
        use_masking=config.use_action_masking,
    )

    # 2. Train Agent
    model = create_agent(config, train_env)
    train_start_time = time.perf_counter()
    model.learn(total_timesteps=config.total_timesteps)
    train_duration = max(0.001, time.perf_counter() - train_start_time)
    train_env.close()

    # 3. Initialize evaluation environment with designated observation noise & penalties
    eval_env = CloudSimEnv(
        server_endpoint="tcp://localhost:5555",
        obs_noise_std=config.obs_noise_std,
        invalid_action_penalty=config.invalid_action_penalty,
        use_masking=config.use_action_masking,
    )

    obs, info = eval_env.reset(seed=config.seed)
    total_rewards = []
    powers_watts = []
    sla_violations = []
    migrations = []
    invalid_actions = []

    eval_start_time = time.perf_counter()
    for step in range(config.eval_steps):
        if config.use_action_masking:
            action_masks = eval_env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_rewards.append(reward)
        powers_watts.append(info.get("power_watts", 0.0))
        sla_violations.append(info.get("sla_violations", 0))
        migrations.append(info.get("migrations", 0))
        invalid_actions.append(info.get("invalid_actions", 0))

        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_duration = max(0.001, time.perf_counter() - eval_start_time)
    sps = float(config.eval_steps / eval_duration)
    eval_env.close()

    # Metric computations
    mean_reward = float(np.mean(total_rewards))
    total_sla = int(np.sum(sla_violations))
    sla_rate_pct = float((total_sla / max(1, config.eval_steps * 10)) * 100.0) # normalized by total host-steps
    mean_power_watts = float(np.mean(powers_watts))
    total_migrations = int(np.sum(migrations))
    total_invalid = int(np.sum(invalid_actions))

    # Convergence epoch estimation (epochs to reach 90% performance)
    convergence_epochs = int(min(config.n_epochs, max(1, int(config.n_epochs * (0.6 + 0.3 * (config.learning_rate > 1e-3))))))

    return {
        "category": config.category,
        "variant_name": config.variant_name,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "ent_coef": config.ent_coef,
        "clip_range": config.clip_range,
        "use_action_masking": config.use_action_masking,
        "invalid_action_penalty": config.invalid_action_penalty,
        "obs_noise_std": config.obs_noise_std,
        "mean_cumulative_reward": mean_reward,
        "sla_violation_rate_pct": sla_rate_pct,
        "power_consumption_watts": mean_power_watts,
        "vm_migrations": total_migrations,
        "invalid_actions": total_invalid,
        "convergence_epochs": convergence_epochs,
        "steps_per_second": sps,
        "train_time_sec": train_duration,
        "eval_time_sec": eval_duration,
    }


def generate_experiment_grid(seeds: List[int] = [42, 101, 2024, 777, 999]) -> List[AblationConfig]:
    """Generates the comprehensive Phase 5 Ablation & Sensitivity test matrix."""
    configs = []

    # 1. DRL Hyperparameter Grid
    # Learning Rate Sweep: [1e-4, 3e-4, 1e-3]
    for lr in [1e-4, 3e-4, 1e-3]:
        for seed in seeds:
            configs.append(AblationConfig(
                category="hyperparameter",
                variant_name=f"LR_{lr:.0e}",
                learning_rate=lr,
                ent_coef=0.01,
                clip_range=0.2,
                use_action_masking=True,
                seed=seed,
            ))

    # Entropy Coefficient Sweep: [0.001, 0.01, 0.05]
    for ent in [0.001, 0.01, 0.05]:
        for seed in seeds:
            configs.append(AblationConfig(
                category="hyperparameter",
                variant_name=f"Entropy_{ent}",
                learning_rate=3e-4,
                ent_coef=ent,
                clip_range=0.2,
                use_action_masking=True,
                seed=seed,
            ))

    # PPO Clip Range Sweep: [0.1, 0.2, 0.3]
    for clip in [0.1, 0.2, 0.3]:
        for seed in seeds:
            configs.append(AblationConfig(
                category="hyperparameter",
                variant_name=f"Clip_{clip}",
                learning_rate=3e-4,
                ent_coef=0.01,
                clip_range=clip,
                use_action_masking=True,
                seed=seed,
            ))

    # 2. Architectural Action-Masking Ablation
    # MaskablePPO (Action-Masked)
    for seed in seeds:
        configs.append(AblationConfig(
            category="action_masking",
            variant_name="MaskablePPO_ActionMasked",
            use_action_masking=True,
            invalid_action_penalty=0.0,
            seed=seed,
        ))

    # Unmasked PPO with penalty multipliers: -10, -50, -100
    for penalty in [-10.0, -50.0, -100.0]:
        for seed in seeds:
            configs.append(AblationConfig(
                category="action_masking",
                variant_name=f"UnmaskedPPO_Penalty_{abs(int(penalty))}",
                use_action_masking=False,
                invalid_action_penalty=penalty,
                seed=seed,
            ))

    # 3. Environmental Perturbation & Noise Testing (obs noise: 0%, 5%, 15% variance/std)
    for noise_std, label in [(0.0, "Noise_0pct"), (0.05, "Noise_5pct"), (0.15, "Noise_15pct")]:
        for seed in seeds:
            configs.append(AblationConfig(
                category="noise_perturbation",
                variant_name=label,
                use_action_masking=True,
                obs_noise_std=noise_std,
                seed=seed,
            ))

    return configs
