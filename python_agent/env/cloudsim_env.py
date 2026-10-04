"""
CloudSimPlus Gymnasium Environment
Phase 1 & Phase 5: Environment, Action Masking, Observation Noise Perturbation, and Invalid Action Penalty Shaping.
"""

from typing import Any, Dict, Optional, Tuple
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import zmq


class CloudSimEnv(gym.Env):
    """
    Gymnasium environment interface for CloudSimPlus dynamic VM scheduling.
    
    Observation Space:
        Box(40,), float32 in [0.0, 1.0]
        - [0..9]   : Normalized Host CPU Utilization (10 Hosts)
        - [10..19] : Normalized Host RAM Utilization (10 Hosts)
        - [20..29] : Normalized Host Network Bandwidth Utilization (10 Hosts)
        - [30..39] : Normalized Host Task Queue Load Ratio (10 Hosts)
        
    Action Space:
        MultiDiscrete([10] * 20)
        - For each of the 20 VMs, an integer in [0..9] representing the target host.
        
    Reward Function:
        R_t = - ( alpha * TotalPowerWatts + beta * SLAViolations + gamma * MigrationCost ) + invalid_action_penalty
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        server_endpoint: str = "tcp://localhost:5555",
        timeout_ms: int = 500,
        num_hosts: int = 10,
        num_vms: int = 20,
        obs_noise_std: float = 0.0,
        invalid_action_penalty: float = 0.0,
        use_masking: bool = True,
        use_fallback_sim: bool = True,
    ):
        super().__init__()

        self.server_endpoint = server_endpoint
        self.timeout_ms = timeout_ms
        self.num_hosts = num_hosts
        self.num_vms = num_vms
        self.obs_noise_std = float(obs_noise_std)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.use_masking = use_masking
        self.use_fallback_sim = use_fallback_sim

        # 40 observation features
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4 * num_hosts,),
            dtype=np.float32,
        )

        self.action_space = spaces.MultiDiscrete(
            [num_hosts] * num_vms,
            dtype=np.int64,
        )

        self.max_steps = 500
        self.current_step = 0
        self._current_obs = None
        self._connected = False
        self._context = None
        self._socket = None

        # Simulation state for fallback / standalone execution
        self._init_simulation_state()

        # Attempt connection to Java server if desired
        if server_endpoint:
            try:
                self._context = zmq.Context()
                self._socket = self._context.socket(zmq.REQ)
                self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.connect(self.server_endpoint)
                # Test ping
                self._send_command({"command": "RESET"})
                self._connected = True
            except Exception:
                self._connected = False
                if self._socket:
                    self._socket.close()
                    self._socket = None
                if self._context:
                    self._context.term()
                    self._context = None

    def _init_simulation_state(self):
        """Initializes internal high-fidelity datacenter state for standalone execution."""
        # Host configs: Hosts 0-4 (Dual-core 3.0GHz, 16GB RAM, 250W max, 90W idle)
        # Hosts 5-9 (Dual-core 3.0GHz, 8GB RAM, 180W max, 70W idle)
        self.host_ram_capacity = np.array([16384] * 5 + [8192] * 5, dtype=np.float32)
        self.host_max_power = np.array([250.0] * 5 + [180.0] * 5, dtype=np.float32)
        self.host_idle_power = np.array([90.0] * 5 + [70.0] * 5, dtype=np.float32)
        self.host_mips_capacity = np.array([6000.0] * 10, dtype=np.float32)

        # VM configs: 0-7 Small (2GB RAM, 1000 MIPS), 8-15 Med (4GB RAM, 1200 MIPS), 16-19 Large (4GB RAM, 2000 MIPS)
        self.vm_ram = np.array([2048] * 8 + [4096] * 12, dtype=np.float32)
        self.vm_mips = np.array([1000.0] * 8 + [1200.0] * 8 + [2000.0] * 4, dtype=np.float32)
        self.vm_allocations = np.zeros(self.num_vms, dtype=np.int64) # Initially on host 0-3

        self.cumulative_energy_joules = 0.0
        self.total_sla_throttling_seconds = 0.0
        self.total_execution_seconds = 0.0
        self.current_step = 0
        self.sim_clock = 0.0
        self.step_duration_sec = 10.0

    def _send_command(self, payload: dict) -> dict:
        """Sends a JSON command to the Java CloudSimPlus server if connected."""
        if not self._connected or self._socket is None:
            raise ConnectionError("ZeroMQ socket not connected")
        req_str = json.dumps(payload)
        self._socket.send_string(req_str)
        reply_str = self._socket.recv_string()
        return json.loads(reply_str)

    def _apply_noise(self, obs: np.ndarray) -> np.ndarray:
        """Injects Gaussian observation noise if obs_noise_std > 0."""
        if self.obs_noise_std > 0.0:
            noise = np.random.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the datacenter simulation."""
        super().reset(seed=seed)
        self.current_step = 0
        self.cumulative_energy_joules = 0.0
        self.total_sla_throttling_seconds = 0.0
        self.total_execution_seconds = 0.0
        self.sim_clock = 0.0

        if seed is not None:
            np.random.seed(seed)

        # VM initial placement: First-Fit across suitable hosts
        self.vm_allocations = np.array([0]*6 + [1]*6 + [2]*4 + [3]*4, dtype=np.int64)

        if self._connected:
            try:
                response = self._send_command({"command": "RESET"})
                raw_obs = response.get("observation", [0.0] * (4 * self.num_hosts))
                observation = np.array(raw_obs, dtype=np.float32)
                observation = np.clip(observation, 0.0, 1.0)
                self._current_obs = observation
                info = response.get("info", {})
                return self._apply_noise(observation), info
            except Exception:
                self._connected = False

        # Fallback simulation reset
        observation = self._get_fallback_observation()
        self._current_obs = observation
        info = {
            "power_watts": self._calculate_fallback_power(),
            "sla_violations": 0,
            "migrations": 0,
            "active_shutdowns": self._calculate_active_shutdowns(),
            "edp": 0.0,
            "sla_percent": 0.0,
            "total_energy_joules": 0.0,
            "step": 0,
            "sim_time": 0.0
        }
        return self._apply_noise(observation), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances the simulation by applying target VM allocations."""
        self.current_step += 1
        self.sim_clock += self.step_duration_sec

        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.int64)
        elif not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.int64)

        if self._connected:
            try:
                payload = {"command": "STEP", "action": action.tolist()}
                response = self._send_command(payload)
                raw_obs = response.get("observation", [0.0] * (4 * self.num_hosts))
                observation = np.array(raw_obs, dtype=np.float32)
                observation = np.clip(observation, 0.0, 1.0)
                self._current_obs = observation
                reward = float(response.get("reward", 0.0))
                done = bool(response.get("done", False))
                info = response.get("info", {})
                return self._apply_noise(observation), reward, done, False, info
            except Exception:
                self._connected = False

        # Fallback in-process execution matching CloudSimPlus dynamics
        migrations_count = 0
        invalid_actions_count = 0

        # Calculate pre-allocation capacity
        host_allocated_ram = np.zeros(self.num_hosts, dtype=np.float32)
        for vm_id in range(self.num_vms):
            curr_h = self.vm_allocations[vm_id]
            host_allocated_ram[curr_h] += self.vm_ram[vm_id]

        new_allocations = self.vm_allocations.copy()

        for vm_id in range(self.num_vms):
            target_h = int(action[vm_id]) if vm_id < len(action) else int(self.vm_allocations[vm_id])
            target_h = max(0, min(self.num_hosts - 1, target_h))
            curr_h = self.vm_allocations[vm_id]

            if target_h != curr_h:
                req_ram = self.vm_ram[vm_id]
                avail_ram = self.host_ram_capacity[target_h] - host_allocated_ram[target_h]
                if avail_ram >= req_ram:
                    # Valid migration
                    host_allocated_ram[curr_h] -= req_ram
                    host_allocated_ram[target_h] += req_ram
                    new_allocations[vm_id] = target_h
                    migrations_count += 1
                else:
                    # Invalid action (RAM constraint violated)
                    invalid_actions_count += 1

        self.vm_allocations = new_allocations
        observation = self._get_fallback_observation()
        self._current_obs = observation

        power_watts = self._calculate_fallback_power()
        sla_violations = self._calculate_sla_violations()
        active_shutdowns = self._calculate_active_shutdowns()

        self.cumulative_energy_joules += power_watts * self.step_duration_sec
        self.total_execution_seconds += self.step_duration_sec * self.num_hosts
        if sla_violations > 0:
            self.total_sla_throttling_seconds += sla_violations * self.step_duration_sec

        sla_percent = (self.total_sla_throttling_seconds / max(1.0, self.total_execution_seconds)) * 100.0
        edp = self.cumulative_energy_joules * (self.sim_clock / max(1, self.current_step))

        # Reward formulation: R = -(alpha * P + beta * SLA + gamma * Mig) + invalid_penalty
        alpha_power = 0.005
        beta_sla = 2.0
        gamma_migration = 0.5

        base_reward = -(alpha_power * power_watts + beta_sla * sla_violations + gamma_migration * migrations_count)
        penalty = self.invalid_action_penalty * invalid_actions_count
        reward = float(base_reward + penalty)

        terminated = bool(self.current_step >= self.max_steps)
        truncated = False

        info = {
            "power_watts": float(power_watts),
            "sla_violations": int(sla_violations),
            "invalid_actions": int(invalid_actions_count),
            "migrations": int(migrations_count),
            "active_shutdowns": int(active_shutdowns),
            "edp": float(edp),
            "sla_percent": float(sla_percent),
            "total_energy_joules": float(self.cumulative_energy_joules),
            "step": int(self.current_step),
            "sim_time": float(self.sim_clock),
        }

        return self._apply_noise(observation), reward, terminated, truncated, info

    def _get_fallback_observation(self) -> np.ndarray:
        """Calculates 40-dimensional observation state."""
        obs = np.zeros(40, dtype=np.float32)
        host_vms = [[] for _ in range(self.num_hosts)]
        for vm_id in range(self.num_vms):
            h = self.vm_allocations[vm_id]
            host_vms[h].append(vm_id)

        # Time-varying dynamic workload wave with periodic surges
        time = self.sim_clock
        for h in range(self.num_hosts):
            vms_on_h = host_vms[h]
            if len(vms_on_h) == 0:
                obs[h] = 0.0          # CPU
                obs[10 + h] = 0.0     # RAM
                obs[20 + h] = 0.0     # BW
                obs[30 + h] = 0.0     # Queue load
            else:
                total_vm_ram = sum(self.vm_ram[v] for v in vms_on_h)
                total_vm_mips = 0.0
                for v in vms_on_h:
                    phase = (v * np.pi / 10.0)
                    period = 100.0 + (v * 15.0)
                    base_wave = 0.45 + 0.35 * np.sin(2.0 * np.pi * time / period + phase)
                    noise = np.random.uniform(-0.05, 0.05)
                    util = np.clip(base_wave + noise, 0.10, 0.98)
                    total_vm_mips += self.vm_mips[v] * util

                cpu_util = np.clip(total_vm_mips / self.host_mips_capacity[h], 0.0, 1.0)
                ram_util = np.clip(total_vm_ram / self.host_ram_capacity[h], 0.0, 1.0)
                bw_util = np.clip(len(vms_on_h) * 0.12, 0.0, 1.0)
                queue_load = np.clip(len(vms_on_h) / self.num_vms, 0.0, 1.0)

                obs[h] = cpu_util
                obs[10 + h] = ram_util
                obs[20 + h] = bw_util
                obs[30 + h] = queue_load

        return obs

    def _calculate_fallback_power(self) -> float:
        """Calculates power consumption based on active hosts and CPU utilization."""
        total_p = 0.0
        for h in range(self.num_hosts):
            # Check if host has VMs
            has_vms = np.any(self.vm_allocations == h)
            if has_vms and self._current_obs is not None:
                cpu_u = self._current_obs[h]
                # Linear power model: P(u) = P_idle + (P_max - P_idle) * u
                p = self.host_idle_power[h] + (self.host_max_power[h] - self.host_idle_power[h]) * cpu_u
                total_p += p
        return float(total_p)

    def _calculate_sla_violations(self) -> int:
        """Counts active hosts exceeding SLA CPU threshold (0.90)."""
        violations = 0
        if self._current_obs is not None:
            for h in range(self.num_hosts):
                has_vms = np.any(self.vm_allocations == h)
                if has_vms and self._current_obs[h] >= 0.90:
                    violations += 1
        return violations

    def _calculate_active_shutdowns(self) -> int:
        """Counts idle physical hosts with zero allocated VMs."""
        active_shutdowns = 0
        for h in range(self.num_hosts):
            if not np.any(self.vm_allocations == h):
                active_shutdowns += 1
        return active_shutdowns

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean action mask for MaskablePPO.
        Shape: (num_vms * num_hosts,) = (200,)
        Masks out destination hosts whose RAM utilization would exceed 95%.
        """
        mask = np.ones((self.num_vms, self.num_hosts), dtype=bool)
        if self._current_obs is not None:
            ram_util = self._current_obs[self.num_hosts : 2 * self.num_hosts]
            for h in range(self.num_hosts):
                if ram_util[h] >= 0.95:
                    mask[:, h] = False

            # Ensure every VM has at least one valid action (staying on current host)
            for vm in range(self.num_vms):
                if not np.any(mask[vm]):
                    curr_h = self.vm_allocations[vm]
                    mask[vm, curr_h] = True

        return mask.reshape(-1)

    def close(self):
        """Closes environment and releases ZeroMQ resources."""
        if self._connected and self._socket:
            try:
                self._send_command({"command": "CLOSE"})
            except Exception:
                pass
            finally:
                if self._socket and not self._socket.closed:
                    self._socket.close()
                if self._context and not self._context.closed:
                    self._context.term()
                self._connected = False

    def __del__(self):
        self.close()
