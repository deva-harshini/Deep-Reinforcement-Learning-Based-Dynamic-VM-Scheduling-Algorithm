#!/bin/bash
set -e

echo "=================================================================="
echo " 🚀 Starting CloudSim-DRL Production Container"
echo "=================================================================="

# Determine port (Render and Railway set $PORT dynamically)
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Start Java CloudSimPlus Simulation Server in the background if jar exists
if [ -f "/app/app.jar" ]; then
    echo "[PROD] Starting Java CloudSimPlus Simulation Engine on tcp://127.0.0.1:5555..."
    java -cp "/app/app.jar:/app/lib/*" com.capstone.sim.CloudSimEnvServer > /tmp/cloudsim_java.log 2>&1 &
    JAVA_PID=$!
    echo "[PROD] Java Simulation Engine started (PID: $JAVA_PID)."
else
    echo "[PROD] Note: Java JAR not found, using high-fidelity in-process CloudSim emulator."
fi

echo "[PROD] Starting FastAPI Model Gateway on http://${HOST}:${PORT}..."
exec uvicorn web_backend.server:app --host "${HOST}" --port "${PORT}" --workers 1
