# DEPTHWIZARD — Single-View Height Estimation & 3D Flythrough

> **Smart India Hackathon (SIH) Prototype Submission**

DEPTHWIZARD is a computer vision and monocular 3D geometry prototype that estimates target object heights, reconstructs 3D point cloud scenes, and generates virtual camera flythrough videos from a **single outdoor RGB image**.

---

## 🏗️ System Architecture (Decoupled Frontend / AI Backend)

To satisfy serverless deployment limits ($< 500\text{ MB}$) while preserving 100% real AI/ML inference (Depth Anything V2, Open3D point clouds, MP4 flythrough rendering), DepthWizard is split into two lightweight architectural boundaries:

```
                    PUBLIC URL
                        │
                        ▼
               DEPTHWIZARD FRONTEND (Streamlit / Serverless)
                        │
                        │ Upload RGB Image & Keypoints
                        ▼
               AI BACKEND REST API (FastAPI / Docker / GPU)
                        │
            ┌───────────┴───────────┐
            │                       │
     Depth Anything V2         Open3D 3D Engine
            │                       │
            └───────────┬───────────┘
                        │
                        ▼
               RESULTS (JSON / Assets)
                        │
                        ▼
               FRONTEND DASHBOARD
```

- **Lightweight Frontend (`app.py`)**: Lightweight UI handling image uploads, interactive visualizers, keypoint inputs, and asset rendering. Contains **zero PyTorch/CUDA dependencies** ($< 15\text{ MB}$ bundle).
- **AI Computation Backend (`backend/`)**: FastAPI REST service running Depth Anything V2, scene scale calibration, Open3D 3D back-projection, and MP4 flythrough video rendering.

---

## 🚀 Local Development & Execution

### 1. Launch AI Computation Backend API
```bash
# Terminal 1: Start Backend API (runs on http://localhost:8000)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Backend Endpoints:
- `GET /api/health`: System status, loaded model name, GPU/CPU device.
- `POST /api/analyze`: Accepts RGB image upload + parameters, returns metric depth, scale calibration, height estimate, point cloud PLY, and flythrough MP4 video.

### 2. Launch Streamlit Public Frontend
```bash
# Terminal 2: Start Frontend GUI
streamlit run app.py
```
Configured via `.env` environment variable:
```env
BACKEND_API_URL=http://localhost:8000
```

---

## 🐳 Docker Deployment

The AI Computation Backend can be containerized for cloud deployment:

```bash
# Build Docker container for AI Backend
docker build -t depthwizard-backend -f backend/Dockerfile .

# Run Backend Container
docker run -p 8000:8000 depthwizard-backend
```

---

## 🌟 Core Pipeline Steps

1. **Monocular Metric Depth Estimation**: Uses Depth Anything V2 (`Depth-Anything-V2-Metric-Outdoor-Small-hf`) to predict relative/metric depth.
2. **Reference-Assisted Scale Calibration**: Solves for true scale factor $S_{calib}$ using known reference object height (e.g. $1.70\text{ m}$ person).
3. **Pinhole 3D Geometry Height Solver**: Calculates 3D camera back-projection target height.
4. **3D Point Cloud Scene Reconstruction**: Back-projects RGB depth map into metric 3D point cloud (`.ply`) rendered dynamically in interactive WebGL Three.js.
5. **Virtual Camera Flythrough**: Renders 3D camera orbital flythrough trajectory into playable MP4 video.

---

*Developed for Smart India Hackathon (SIH) Prototype Demonstration.*
