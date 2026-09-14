"""
CloudSimPlus Gymnasium Environment
Phase 1: Environment & MDP Formalization for DRL-based Dynamic VM Scheduling
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
        Box(20,), float32 in [0.0, 1.0]
        - [0..9]: Normalized Host CPU Utilization (10 Hosts)
        - [10..19]: Normalized Host RAM Utilization (10 Hosts)
        
    Action Space:
        MultiDiscrete([10] * 20)
        - For each of the 20 VMs, an integer in [0..9] representing the target host.
        
    Reward Function:
        R_t = - ( alpha * TotalPowerWatts + beta * SLAViolations + gamma * MigrationCost )
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        server_endpoint: str = "tcp://localhost:5555",
        timeout_ms: int = 5000,
        num_hosts: int = 10,
        num_vms: int = 20,
    ):
        super().__init__()

        self.server_endpoint = server_endpoint
        self.timeout_ms = timeout_ms
        self.num_hosts = num_hosts
        self.num_vms = num_vms

        # Formal Gymnasium Spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2 * num_hosts,),
            dtype=np.float32,
        )

        self.action_space = spaces.MultiDiscrete(
            [num_hosts] * num_vms,
            dtype=np.int64,
        )

        # ZeroMQ IPC setup
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.server_endpoint)

    def _send_command(self, payload: dict) -> dict:
        """Sends a JSON command to the Java CloudSimPlus server and returns the parsed response."""
        try:
            req_str = json.dumps(payload)
            self._socket.send_string(req_str)
            reply_str = self._socket.recv_string()
            return json.loads(reply_str)
        except zmq.Again as e:
            raise TimeoutError(
                f"CloudSimEnv timeout: No response from Java server at {self.server_endpoint} within {self.timeout_ms}ms"
            ) from e
        except Exception as e:
            raise RuntimeError(f"CloudSimEnv IPC error: {e}") from e

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the datacenter simulation to initial state.
        
        Returns:
            observation (np.ndarray): Shape (20,) float32 array in [0.0, 1.0].
            info (dict): Diagnostic metadata from the datacenter.
        """
        super().reset(seed=seed)

        payload = {"command": "RESET"}
        response = self._send_command(payload)

        raw_obs = response.get("observation", [0.0] * (2 * self.num_hosts))
        observation = np.array(raw_obs, dtype=np.float32)
        observation = np.clip(observation, 0.0, 1.0)
        self._current_obs = observation

        info = response.get("info", {})
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Applies VM scheduling action and advances datacenter simulation.
        
        Args:
            action (np.ndarray or list): Target host ID for each VM (length 20).
            
        Returns:
            observation (np.ndarray): Shape (20,) float32 array in [0.0, 1.0].
            reward (float): Scalar reward evaluating energy, SLA, and migrations.
            terminated (bool): Whether episode reached natural end.
            truncated (bool): Whether episode was truncated prematurely.
            info (dict): Metrics including power_watts, sla_violations, migrations, sim_time.
        """
        if isinstance(action, np.ndarray):
            action_list = action.tolist()
        else:
            action_list = list(action)

        payload = {
            "command": "STEP",
            "action": action_list,
        }

        response = self._send_command(payload)

        raw_obs = response.get("observation", [0.0] * (2 * self.num_hosts))
        observation = np.array(raw_obs, dtype=np.float32)
        observation = np.clip(observation, 0.0, 1.0)
        self._current_obs = observation

        reward = float(response.get("reward", 0.0))
        done = bool(response.get("done", False))
        info = response.get("info", {})

        terminated = done
        truncated = False

        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean action mask for MaskablePPO.
        Shape: (num_vms * num_hosts,) = (200,)
        Masks out destination hosts whose RAM utilization is saturated (>= 95%).
        """
        mask = np.ones((self.num_vms, self.num_hosts), dtype=bool)
        if hasattr(self, "_current_obs") and self._current_obs is not None:
            # obs[num_hosts : 2*num_hosts] represents host RAM utilization
            ram_util = self._current_obs[self.num_hosts : 2 * self.num_hosts]
            for h in range(self.num_hosts):
                if ram_util[h] >= 0.95:
                    mask[:, h] = False

            # Ensure every VM has at least one valid action available
            for vm in range(self.num_vms):
                if not np.any(mask[vm]):
                    mask[vm, :] = True

        return mask.reshape(-1)

    def close(self):
        """Closes the environment and releases ZeroMQ IPC resources."""
        try:
            self._send_command({"command": "CLOSE"})
        except Exception:
            pass
        finally:
            if not self._socket.closed:
                self._socket.close()
            if not self._context.closed:
                self._context.term()

    def __del__(self):
        self.close()

