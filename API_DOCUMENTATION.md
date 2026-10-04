# CloudSim-DRL Dynamic VM Scheduler: REST & WebSocket API Specification

**Version:** 2.0.0 Enterprise  
**Base URL:** `http://localhost:8000/api/v1`  
**Interactive Swagger UI:** `http://localhost:8000/docs`  
**ReDoc Specification:** `http://localhost:8000/redoc`  
**WebSocket Telemetry Stream:** `ws://localhost:8000/ws/telemetry`

---

## 1. System Health & Diagnostics

### `GET /api/v1/health`
Retrieves the operational status of the model inference gateway, loaded policy checkpoints, and Java ZeroMQ IPC socket connectivity.

#### Response (`200 OK`):
```json
{
  "status": "HEALTHY",
  "service": "CloudSim-DRL-Scheduler-Backend",
  "version": "2.0.0",
  "timestamp": "2026-10-04T17:30:00Z",
  "model_loaded": true,
  "model_path": "/app/models/ppo_vm_scheduler/ppo_vm_final.zip",
  "available_policies": [
    "MaskablePPO (Trained DRL Agent)",
    "First-Fit Consolidation Heuristic",
    "Minimum Migration Time (MMT)",
    "Local Regression (LR) CPU Trend Predictor",
    "Power-Spread Load Distribution",
    "Random Uniform Baseline"
  ],
  "active_simulation": false,
  "connected_clients": 1
}
```

---

## 2. Simulation & Telemetry Streaming

### `POST /api/v1/simulate`
Triggers an asynchronous simulation run on the datacenter environment. Telemetry frames are continuously streamed to active WebSocket subscribers at `/ws/telemetry`.

#### Request Body:
```json
{
  "policy": "MaskablePPO (Trained DRL Agent)",
  "workload_pattern": "sinusoidal",
  "noise_std": 0.05,
  "total_steps": 100,
  "delay_ms": 35
}
```

#### Field Descriptions:
- `policy` *(string, default: "MaskablePPO (Trained DRL Agent)")*: Scheduling algorithm to execute.
- `workload_pattern` *(string, options: `sinusoidal`, `surge`, `stochastic`, `linear`)*: Dynamic traffic profile.
- `noise_std` *(float, 0.0 to 0.15)*: Gaussian telemetric sensor noise standard deviation.
- `total_steps` *(integer, 10 to 500)*: Simulation duration in discrete timesteps (10 seconds per step).
- `delay_ms` *(integer, 0 to 500)*: Real-time visual throttle interval in milliseconds.

#### Response (`200 OK`):
```json
{
  "status": "STARTED",
  "policy": "MaskablePPO (Trained DRL Agent)",
  "workload_pattern": "sinusoidal",
  "total_steps": 100,
  "message": "Simulation started. Stream telemetry at /ws/telemetry"
}
```

---

### `POST /api/v1/simulate/stop`
Cancels the active simulation run immediately.

#### Response (`200 OK`):
```json
{
  "status": "STOPPED",
  "message": "Active simulation terminated."
}
```

---

## 3. Real-Time WebSocket Telemetry Protocol

### `WebSocket /ws/telemetry`
Full-duplex WebSocket streaming real-time physical rack gauges, host states, and VM migration events.

#### Inbound Control Commands (Client -> Server):
```json
{
  "action": "START",
  "policy": "MaskablePPO (Trained DRL Agent)",
  "workload_pattern": "surge",
  "noise_std": 0.0,
  "total_steps": 100,
  "delay_ms": 35
}
```

#### Outbound Telemetry Frame (Server -> Client):
```json
{
  "step": 42,
  "total_steps": 100,
  "policy": "MaskablePPO (Trained DRL Agent)",
  "power_watts": 948.4,
  "cumulative_energy_kwh": 0.695,
  "sla_violations": 0,
  "sla_percent": 0.0,
  "step_migrations": 2,
  "total_migrations": 18,
  "active_shutdowns": 4,
  "active_hosts": 6,
  "edp": 3840000.0,
  "reward": -8.76,
  "hosts": [
    {
      "id": 0,
      "rack": "Rack A (High Performance)",
      "type": "Type A (Quad-Core 3.0GHz, 16GB RAM)",
      "status": "ACTIVE",
      "cpu_percent": 68.4,
      "ram_percent": 75.0,
      "bw_percent": 48.0,
      "power_watts": 199.4,
      "vm_count": 4,
      "vms": [0, 1, 2, 3],
      "is_sla_breach": false
    }
  ],
  "vms": [
    {
      "id": 0,
      "size": "Small (2GB RAM, 1000 MIPS)",
      "allocated_host": 0,
      "host_type": "Rack A"
    }
  ],
  "timestamp": "17:35:10"
}
```

---

## 4. Single-Step Inference Engine

### `POST /api/v1/predict`
Executes single-step greedy action prediction for external clients supplying a 40-feature datacenter telemetry vector.

#### Request Body:
```json
{
  "observation": [
    0.45, 0.52, 0.61, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.50, 0.55, 0.60, 0.45, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.35, 0.40, 0.45, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.30, 0.30, 0.20, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
  ],
  "policy": "MaskablePPO (Trained DRL Agent)",
  "deterministic": true
}
```

#### Response (`200 OK`):
```json
{
  "policy": "MaskablePPO (Trained DRL Agent)",
  "actions": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 1, 2, 3],
  "action_masks_valid_ratio": 1.0,
  "inference_latency_ms": 0.842,
  "metadata": {
    "policy": "MaskablePPO",
    "type": "DRL_Inference",
    "masked": true
  },
  "timestamp": 1728045310.124
}
```

---

## 5. Multi-Policy Comparative Benchmark

### `POST /api/v1/benchmark`
Runs side-by-side evaluation across all 6 scheduling algorithms on identical workload sequences and returns structured comparison matrices.

#### Request Body:
```json
{
  "steps_per_policy": 50,
  "workload_pattern": "sinusoidal",
  "noise_std": 0.0
}
```

#### Response (`200 OK`):
```json
{
  "benchmark_type": "Multi-Heuristic Comparative Suite",
  "steps_per_policy": 50,
  "workload_pattern": "sinusoidal",
  "results": [
    {
      "policy": "MaskablePPO (Trained DRL Agent)",
      "total_steps": 50,
      "execution_duration_sec": 0.45,
      "steps_per_second": 111.1,
      "avg_power_watts": 984.4,
      "cumulative_energy_kwh": 0.1367,
      "total_sla_violations": 0,
      "avg_sla_percent": 0.0,
      "total_vm_migrations": 18,
      "avg_sleeping_hosts": 4.0,
      "final_edp": 4820000.0,
      "status": "COMPLETED"
    }
  ]
}
```
