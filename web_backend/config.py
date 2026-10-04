"""
Production Configuration for CloudSim-DRL Backend Gateway.
Loads environment variables for Render, Railway, AWS, GCP, and local environments.
"""

import os
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    # Host & Port settings (Render/Railway dynamically inject $PORT)
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Model artifact path
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ppo_vm_scheduler", "ppo_vm_final.zip")
    )

    # Java Simulation Engine IPC Socket
    SERVER_ENDPOINT: str = os.getenv("SERVER_ENDPOINT", "tcp://127.0.0.1:5555")

    # CORS Allowed Origins (Comma-separated or wildcard for Vercel/Netlify)
    RAW_ALLOWED_ORIGINS: str = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000,http://localhost:5173,https://*.vercel.app,https://*.netlify.app,*"
    )

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        if self.RAW_ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.RAW_ALLOWED_ORIGINS.split(",") if origin.strip()]


settings = Settings()
