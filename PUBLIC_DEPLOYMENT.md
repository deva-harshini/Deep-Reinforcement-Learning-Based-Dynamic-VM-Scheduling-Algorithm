# Public Cloud Deployment & Production Hosting Guide

This guide provides step-by-step instructions to deploy the **CloudSim-DRL Dynamic VM Scheduling Platform** to public cloud infrastructure (**Render**, **Railway**, **Vercel**, and **Netlify**), providing a publicly accessible, commercial-grade web demonstration.

---

## 1. Deployment Architecture Options

```
OPTION 1: UNIFIED SINGLE-CONTAINER DEPLOYMENT (Recommended - Simplest & Fastest)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Render.com / Railway.app Web Service (Port 8000 / $PORT)                               │
│                                                                                        │
│   ┌─────────────────────────────────────┐      ZeroMQ IPC Socket      ┌──────────────┐ │
│   │  FastAPI Model Gateway & REST API   │ ◄─────────────────────────► │ Java CloudSim│ │
│   │  - SB3 MaskablePPO Inference        │        (127.0.0.1:5555)     │ Simulation   │ │
│   │  - WebSocket Telemetry Streamer     │                             │ Engine       │ │
│   │  - Serves Web Dashboard Frontend    │                             └──────────────┘ │
│   └─────────────────────────────────────┘                                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
                    ▲
                    │ HTTPS / WSS
                    │
            Client Browser (https://cloudsim-drl.onrender.com)
```

```
OPTION 2: DECOUPLED DEPLOYMENT (Vercel Frontend + Render Backend)
┌───────────────────────────────────────┐            ┌───────────────────────────────────┐
│ Vercel / Netlify Frontend Dashboard   │   HTTPS/WS │ Render / Railway Backend API      │
│ (https://cloudsim-drl.vercel.app)     ├───────────►│ (https://cloudsim-api.onrender.com) │
└───────────────────────────────────────┘            └───────────────────────────────────┘
```

---

## 2. Step 1: Push Repository to GitHub

Ensure all project files, model weights, and Docker configurations are committed:

```bash
cd "Capstone Project"

# Check git status
git status

# Add all production files
git add Dockerfile.production docker-compose.prod.yml entrypoint.sh web_backend/ web_dashboard/ models/ requirements.txt PUBLIC_DEPLOYMENT.md

# Commit and push to your remote repository
git commit -m "feat(deploy): production multi-stage Dockerfile, Vercel config, and public cloud deployment harness"
git push origin main
```

---

## 3. Step 2: Deploy Backend to Render (1-Click Docker)

1. **Sign in to Render:** Go to [https://render.com](https://render.com) and log in with GitHub.
2. **Create New Web Service:**
   - Click **New +** -> **Web Service**.
   - Connect your GitHub repository (`Capstone Project`).
3. **Configure Service Settings:**
   - **Name:** `cloudsim-drl-backend` (or your preferred name)
   - **Region:** `Oregon (US West)` or `Frankfurt (EU Central)`
   - **Language / Environment:** `Docker`
   - **Branch:** `main`
   - **Dockerfile Path:** `Dockerfile.production`
   - **Instance Type:** `Starter` (0.5 CPU, 512MB RAM) or `Standard`
4. **Configure Environment Variables:**
   Click **Advanced** -> **Add Environment Variable**:
   - `ALLOWED_ORIGINS` = `*`
   - `PYTHONUNBUFFERED` = `1`
5. **Deploy:** Click **Create Web Service**.
   - Render will build the multi-stage Docker container (JDK 17 + Python 3.10) and assign a public URL:
   - `https://cloudsim-drl-backend.onrender.com`

---

## 4. Step 3: Deploy Backend to Railway (Alternative)

1. **Sign in to Railway:** Go to [https://railway.app](https://railway.app).
2. **New Project:** Click **New Project** -> **Deploy from GitHub repo**.
3. **Select Repository:** Choose your repository.
4. **Configure Dockerfile:**
   - Go to **Settings** -> **Build**.
   - Set **Dockerfile Path** to `Dockerfile.production`.
5. **Generate Public Domain:**
   - Go to **Settings** -> **Networking** -> Click **Generate Domain**.
   - Copy your public domain (e.g. `https://cloudsim-drl-production.up.railway.app`).

---

## 5. Step 4: Deploy Frontend to Vercel (Optional Decoupled Hosting)

If you wish to host the static Web Dashboard on Vercel with lightning-fast Global CDN edge routing:

1. **Sign in to Vercel:** Go to [https://vercel.com](https://vercel.com).
2. **Import Project:** Click **Add New...** -> **Project** -> Import your GitHub repository.
3. **Configure Root Directory:**
   - Click **Edit** next to **Root Directory** and select `web_dashboard`.
   - **Framework Preset:** `Other`.
4. **Environment Variables:**
   - Add `NEXT_PUBLIC_API_URL` = `https://your-backend.onrender.com`
   - Add `VITE_API_URL` = `https://your-backend.onrender.com`
5. **Deploy:** Click **Deploy**.
   - Your frontend dashboard will be available at `https://cloudsim-drl-dashboard.vercel.app`.
   - The dashboard includes a built-in API config modal to connect to your Render backend with one click.

---

## 6. Step 5: Deploy Frontend to Netlify (Alternative)

1. **Sign in to Netlify:** Go to [https://netlify.com](https://netlify.com).
2. **Import from Git:** Select your GitHub repo.
3. **Build Settings:**
   - **Base directory:** `web_dashboard`
   - **Publish directory:** `.`
4. **Deploy Site:** Click **Deploy Site**.

---

## 7. Verification & Client Demo Checklist

Once deployed, verify your public URLs:

1. **API Health Check:**
   ```bash
   curl https://your-backend.onrender.com/api/v1/health
   ```
   *Expected Response:* `{"status": "HEALTHY", "model_loaded": true, ...}`

2. **Interactive Swagger Documentation:**
   Open `https://your-backend.onrender.com/docs` in your browser.

3. **Live Web Dashboard:**
   Open `https://your-backend.onrender.com/` (or your Vercel URL).
   - Test **Start Run** with `MaskablePPO` policy.
   - Verify live WebSocket physical rack gauges, host CPU/RAM bars, and real-time Chart.js power graphs.
   - Run **Benchmark Suite** to evaluate side-by-side performance across all 6 scheduling algorithms.
