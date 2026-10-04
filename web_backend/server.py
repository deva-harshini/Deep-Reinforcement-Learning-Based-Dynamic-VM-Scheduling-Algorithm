"""
Enterprise FastAPI Backend Gateway for Dynamic DRL VM Scheduling.
Provides REST APIs, Model Inference, WebSocket Telemetry, and Serves Web Dashboard.
"""

import os
import sys
import time
import asyncio
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_backend.config import settings
from web_backend.model_service import model_service
from web_backend.simulation_runner import simulation_runner

app = FastAPI(
    title="CloudSim DRL Dynamic VM Scheduler API",
    description="Production-grade REST & WebSocket API for Deep Reinforcement Learning (Action-Masked PPO) Dynamic VM Scheduling in Cloud Datacenters.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Vercel, Netlify, and custom enterprise domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if "*" not in settings.ALLOWED_ORIGINS else ["*"],
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.netlify\.app" if "*" not in settings.ALLOWED_ORIGINS else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    policy: str = Field("MaskablePPO (Trained DRL Agent)", description="Scheduling algorithm")
    workload_pattern: str = Field("sinusoidal", description="Workload profile: sinusoidal, stochastic, surge, linear")
    noise_std: float = Field(0.0, ge=0.0, le=0.5, description="Gaussian telemetric noise standard deviation")
    total_steps: int = Field(100, ge=10, le=500, description="Simulation duration in timesteps (10s per step)")
    delay_ms: int = Field(35, ge=0, le=500, description="Inter-step streaming delay in milliseconds")


class PredictRequest(BaseModel):
    observation: List[float] = Field(..., description="40-dimensional datacenter state vector [10 CPU, 10 RAM, 10 BW, 10 Queue]")
    policy: str = Field("MaskablePPO (Trained DRL Agent)", description="Target policy to evaluate")
    deterministic: bool = Field(True, description="Deterministic greedy action selection")


class BenchmarkRequest(BaseModel):
    steps_per_policy: int = Field(50, ge=10, le=200, description="Evaluation steps per benchmark policy")
    workload_pattern: str = Field("sinusoidal", description="Workload profile")
    noise_std: float = Field(0.0, description="Observation noise std")


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", tags=["System Health"])
async def health_check():
    """Returns system status, active model weights, and IPC socket bridge condition."""
    return {
        "status": "HEALTHY",
        "service": "CloudSim-DRL-Scheduler-Backend",
        "version": "2.0.0",
        "environment": "production" if os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT") else "development",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_loaded": model_service.is_model_loaded,
        "model_path": model_service.model_path,
        "available_policies": model_service.AVAILABLE_POLICIES,
        "active_simulation": simulation_runner.is_running,
        "connected_clients": len(simulation_runner.active_connections),
    }


@app.post("/api/v1/simulate", tags=["Simulation Control"])
async def trigger_simulation(req: SimulateRequest, background_tasks: BackgroundTasks):
    """Triggers an asynchronous simulation run streaming real-time telemetry over WebSockets."""
    if simulation_runner.is_running:
        return JSONResponse(
            status_code=409,
            content={"error": "A simulation run is already in progress. Call /api/v1/simulate/stop first."},
        )

    # Launch simulation task
    background_tasks.add_task(
        simulation_runner.run_simulation_episode,
        policy=req.policy,
        workload_pattern=req.workload_pattern,
        noise_std=req.noise_std,
        total_steps=req.total_steps,
        delay_ms=req.delay_ms,
    )

    return {
        "status": "STARTED",
        "policy": req.policy,
        "workload_pattern": req.workload_pattern,
        "total_steps": req.total_steps,
        "message": "Simulation started. Stream telemetry at /ws/telemetry",
    }


@app.post("/api/v1/simulate/stop", tags=["Simulation Control"])
async def stop_simulation():
    """Cancels active simulation run."""
    simulation_runner.stop()
    return {"status": "STOPPED", "message": "Active simulation terminated."}


@app.get("/api/v1/metrics", tags=["Telemetry & Metrics"])
async def get_metrics_history():
    """Retrieves the telemetry frame history of the latest simulation episode."""
    return {
        "total_frames": len(simulation_runner.history_metrics),
        "history": simulation_runner.history_metrics[-50:],
    }


@app.post("/api/v1/predict", tags=["Inference Engine"])
async def predict_single_step(req: PredictRequest):
    """
    Executes single-step inference taking custom 40-feature telemetry vectors
    and returning 20-VM target host decisions alongside valid action masks and latency.
    """
    if len(req.observation) != 40:
        raise HTTPException(status_code=400, detail=f"Observation must contain exactly 40 floats, got {len(req.observation)}")

    t0 = time.perf_counter()
    action_masks = model_service.compute_action_masks(req.observation)

    actions, meta = model_service.predict(
        policy=req.policy,
        obs=req.observation,
        action_masks=action_masks,
        deterministic=req.deterministic,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "policy": req.policy,
        "actions": actions,
        "action_masks_valid_ratio": float(np.mean(action_masks)),
        "inference_latency_ms": round(latency_ms, 3),
        "metadata": meta,
        "timestamp": time.time(),
    }


@app.post("/api/v1/benchmark", tags=["Benchmarking Suite"])
async def run_policy_benchmark(req: BenchmarkRequest):
    """
    Executes a comprehensive multi-heuristic comparative benchmark across all 6 scheduling algorithms.
    """
    policies = model_service.AVAILABLE_POLICIES
    results = []

    for pol in policies:
        res = await simulation_runner.run_simulation_episode(
            policy=pol,
            workload_pattern=req.workload_pattern,
            noise_std=req.noise_std,
            total_steps=req.steps_per_policy,
            delay_ms=0,
        )
        results.append(res)

    return {
        "benchmark_type": "Multi-Heuristic Comparative Suite",
        "steps_per_policy": req.steps_per_policy,
        "workload_pattern": req.workload_pattern,
        "results": results,
    }


# ---------------------------------------------------------------------------
# WebSocket Streaming Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    Full-duplex WebSocket streaming real-time datacenter rack gauges,
    host power draws, VM migrations, and SLA alarms.
    """
    await simulation_runner.connect_websocket(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                action = cmd.get("action")
                if action == "START":
                    asyncio.create_task(simulation_runner.run_simulation_episode(
                        policy=cmd.get("policy", "MaskablePPO (Trained DRL Agent)"),
                        workload_pattern=cmd.get("workload_pattern", "sinusoidal"),
                        noise_std=float(cmd.get("noise_std", 0.0)),
                        total_steps=int(cmd.get("total_steps", 100)),
                        delay_ms=int(cmd.get("delay_ms", 35)),
                    ))
                elif action == "STOP":
                    simulation_runner.stop()
            except Exception:
                pass
    except WebSocketDisconnect:
        simulation_runner.disconnect_websocket(websocket)


# ---------------------------------------------------------------------------
# Static Web Dashboard Frontend Mounting
# ---------------------------------------------------------------------------

DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "web_dashboard")
if not os.path.exists(DASHBOARD_DIR):
    DASHBOARD_DIR = "/Users/harshini/Downloads/GITAM_Template_CSSE/web_dashboard"

if os.path.exists(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

    @app.api_route("/", methods=["GET", "HEAD"], tags=["Frontend"])
    async def serve_dashboard():
        index_file = os.path.join(DASHBOARD_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Web dashboard frontend file index.html is being prepared."}


if __name__ == "__main__":
    import uvicorn
    print(f"Starting CloudSim DRL Scheduler API on http://{settings.HOST}:{settings.PORT}")
    uvicorn.run("web_backend.server:app", host=settings.HOST, port=settings.PORT, workers=1)
