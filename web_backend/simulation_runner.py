"""
Live Datacenter Simulation Engine & Real-Time Telemetry Streamer.
Interfaces with CloudSimEnv and broadcasts step-by-step telemetry over WebSockets.
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket

from python_agent.env.cloudsim_env import CloudSimEnv
from web_backend.model_service import model_service


class SimulationRunner:
    """Manages active simulation instances and broadcasts telemetry."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.is_running = False
        self.current_step = 0
        self.max_steps = 200
        self.history_metrics: List[Dict[str, Any]] = []

    async def connect_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect_websocket(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast_telemetry(self, data: Dict[str, Any]):
        """Broadcasts live step telemetry to all connected WebSocket clients."""
        if not self.active_connections:
            return

        message = json.dumps(data)
        dead_connections = set()
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                dead_connections.add(conn)

        for dead in dead_connections:
            self.active_connections.discard(dead)

    def generate_workload_state(self, step: int, pattern: str = "sinusoidal", noise_std: float = 0.0) -> np.ndarray:
        """Generates dynamic 40-feature observation state based on pattern."""
        obs = np.zeros(40, dtype=np.float32)
        t = float(step)

        # Host dynamic profiles
        for h in range(10):
            if pattern == "stochastic":
                base_cpu = 0.50 + 0.30 * np.sin(2.0 * np.pi * t / 40.0 + h) + np.random.normal(0, 0.12)
            elif pattern == "surge":
                # periodic sudden traffic surges
                surge = 0.40 if (step % 25 < 8 and h < 5) else 0.0
                base_cpu = 0.35 + surge + 0.15 * np.sin(2.0 * np.pi * t / 30.0 + h)
            elif pattern == "linear":
                base_cpu = 0.20 + (0.60 * (step / max(1, self.max_steps)))
            else:  # sinusoidal
                base_cpu = 0.45 + 0.35 * np.sin(2.0 * np.pi * t / 50.0 + (h * np.pi / 5.0))

            if noise_std > 0:
                base_cpu += np.random.normal(0, noise_std)

            cpu_val = float(np.clip(base_cpu, 0.05, 0.98))
            ram_val = float(np.clip(0.40 + 0.30 * np.cos(2.0 * np.pi * t / 45.0 + h), 0.10, 0.92))
            bw_val = float(np.clip(cpu_val * 0.85, 0.05, 0.95))
            queue_val = float(np.clip(cpu_val * 0.70, 0.05, 0.90))

            obs[h] = cpu_val
            obs[10 + h] = ram_val
            obs[20 + h] = bw_val
            obs[30 + h] = queue_val

        return obs

    async def run_simulation_episode(
        self,
        policy: str = "MaskablePPO (Trained DRL Agent)",
        workload_pattern: str = "sinusoidal",
        noise_std: float = 0.0,
        total_steps: int = 100,
        delay_ms: int = 40,
    ) -> Dict[str, Any]:
        """
        Executes a real-time or fast simulation run, streaming live metrics to clients.
        """
        self.is_running = True
        self.max_steps = total_steps
        self.history_metrics.clear()

        # Initialize Environment
        env = CloudSimEnv(
            server_endpoint="tcp://localhost:5555",
            obs_noise_std=noise_std,
            use_masking=True,
        )

        obs, info = env.reset()
        cumulative_energy_kwh = 0.0
        total_migrations = 0
        total_sla_violations = 0
        powers = []
        sla_percents = []
        edps = []
        host_telemetry_history = []

        start_time = time.perf_counter()

        for step in range(1, total_steps + 1):
            if not self.is_running:
                break

            self.current_step = step

            # Step 1: Compute Action Masks
            action_masks = env.action_masks()

            # Step 2: Policy Action Decision
            action, meta = model_service.predict(
                policy=policy,
                obs=obs,
                action_masks=action_masks,
                deterministic=True,
            )

            # Step 3: Advance Datacenter State
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract Telemetry
            power_w = float(info.get("power_watts", 950.0))
            sla_v = int(info.get("sla_violations", 0))
            migs = int(info.get("migrations", 0))
            shutdowns = int(info.get("active_shutdowns", 0))
            sla_pct = float(info.get("sla_percent", 0.0))
            edp = float(info.get("edp", 0.0))

            step_energy_kwh = (power_w * 10.0) / (3600.0 * 1000.0)  # 10s step duration
            cumulative_energy_kwh += step_energy_kwh
            total_migrations += migs
            total_sla_violations += sla_v

            powers.append(power_w)
            sla_percents.append(sla_pct)
            edps.append(edp)

            # Build detailed 10-Host Physical Rack Data
            hosts_data = []
            for h in range(10):
                # Rack A (0..4) = Type A (Quad-Core, 16GB, 250W)
                # Rack B (5..9) = Type B (Dual-Core, 8GB, 180W)
                is_type_a = h < 5
                host_type = "Type A (Quad-Core 3.0GHz, 16GB RAM)" if is_type_a else "Type B (Dual-Core 3.0GHz, 8GB RAM)"
                max_p = 250.0 if is_type_a else 180.0
                idle_p = 90.0 if is_type_a else 70.0

                cpu_util = float(obs[h] * 100.0)
                ram_util = float(obs[10 + h] * 100.0)
                bw_util = float(obs[20 + h] * 100.0)
                queue_load = float(obs[30 + h] * 100.0)

                # Count VMs on this host
                vms_on_host = [i for i, target in enumerate(action) if target == h]
                is_active = len(vms_on_host) > 0
                host_power = (idle_p + (max_p - idle_p) * (cpu_util / 100.0)) if is_active else 0.0

                hosts_data.append({
                    "id": h,
                    "rack": "Rack A (High Performance)" if is_type_a else "Rack B (Energy Saver)",
                    "type": host_type,
                    "status": "ACTIVE" if is_active else "SLEEPING",
                    "cpu_percent": round(cpu_util, 1),
                    "ram_percent": round(ram_util, 1),
                    "bw_percent": round(bw_util, 1),
                    "queue_percent": round(queue_load, 1),
                    "power_watts": round(host_power, 1),
                    "vm_count": len(vms_on_host),
                    "vms": vms_on_host,
                    "is_sla_breach": cpu_util >= 90.0 and is_active,
                })

            # VM Allocation State (20 VMs)
            vm_data = []
            for vm_id in range(20):
                vm_size = "Small (2GB RAM, 1000 MIPS)" if vm_id < 8 else ("Medium (4GB RAM, 1200 MIPS)" if vm_id < 16 else "Large (4GB RAM, 2000 MIPS)")
                target_h = int(action[vm_id])
                vm_data.append({
                    "id": vm_id,
                    "size": vm_size,
                    "allocated_host": target_h,
                    "host_type": "Rack A" if target_h < 5 else "Rack B",
                })

            # Real-time frame payload
            telemetry_frame = {
                "step": step,
                "total_steps": total_steps,
                "policy": policy,
                "workload_pattern": workload_pattern,
                "power_watts": round(power_w, 2),
                "cumulative_energy_kwh": round(cumulative_energy_kwh, 4),
                "sla_violations": sla_v,
                "sla_percent": round(sla_pct, 2),
                "step_migrations": migs,
                "total_migrations": total_migrations,
                "active_shutdowns": shutdowns,
                "active_hosts": 10 - shutdowns,
                "edp": round(edp, 2),
                "reward": round(reward, 3),
                "hosts": hosts_data,
                "vms": vm_data,
                "timestamp": time.strftime("%H:%M:%S"),
            }

            self.history_metrics.append(telemetry_frame)

            # Broadcast via WebSocket
            await self.broadcast_telemetry(telemetry_frame)

            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)

            if terminated or truncated:
                obs, info = env.reset()

        total_time = time.perf_counter() - start_time
        env.close()
        self.is_running = False

        # Build Summary Report
        summary = {
            "policy": policy,
            "total_steps": total_steps,
            "execution_duration_sec": round(total_time, 2),
            "steps_per_second": round(total_steps / max(0.001, total_time), 1),
            "avg_power_watts": round(float(np.mean(powers)), 2) if powers else 0.0,
            "cumulative_energy_kwh": round(cumulative_energy_kwh, 4),
            "total_sla_violations": total_sla_violations,
            "avg_sla_percent": round(float(np.mean(sla_percents)), 2) if sla_percents else 0.0,
            "total_vm_migrations": total_migrations,
            "avg_sleeping_hosts": round(float(np.mean([f["active_shutdowns"] for f in self.history_metrics])), 1) if self.history_metrics else 0.0,
            "final_edp": round(edps[-1], 2) if edps else 0.0,
            "status": "COMPLETED",
        }
        return summary

    def stop(self):
        self.is_running = False


# Global simulation runner
simulation_runner = SimulationRunner()
