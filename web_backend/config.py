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

    # CORS Allowed Origins (Explicit domains, previews, and wildcard fallback)
    RAW_ALLOWED_ORIGINS: str = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8000,http://127.0.0.1:5173,https://deep-reinforcement-learning-based-d.vercel.app,https://*.vercel.app,https://*.netlify.app,*"
    )

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        if self.RAW_ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        origins = [origin.strip() for origin in self.RAW_ALLOWED_ORIGINS.split(",") if origin.strip()]
        # Ensure production Vercel frontend is always explicitly present
        vercel_origin = "https://deep-reinforcement-learning-based-d.vercel.app"
        if vercel_origin not in origins and "*" not in origins:
            origins.append(vercel_origin)
        return origins


settings = Settings()
