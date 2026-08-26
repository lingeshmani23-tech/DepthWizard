# DEPTHWIZARD — Single-View Height Estimation & 3D Flythrough

> **Smart India Hackathon (SIH) Prototype Submission**

[![GitHub Pages Showcase](https://img.shields.io/badge/GitHub%20Pages-Live%20Showcase-38BDF8?style=for-the-badge&logo=github)](https://lingeshmani23-tech.github.io/DepthWizard/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Repository](https://img.shields.io/badge/GitHub-DepthWizard-indigo?style=for-the-badge&logo=github)](https://github.com/lingeshmani23-tech/DepthWizard)

DEPTHWIZARD is a computer vision and monocular 3D geometry prototype that estimates target object heights, reconstructs 3D point cloud scenes, and generates virtual camera flythrough videos from a **single outdoor RGB image**.

### 🚀 SIH Demonstration Commands

```bash
# 🏆 PRIMARY AUTOMATIC SIH DEMO (Zero manual input required)
python demo.py

# 🎮 OPTIONAL INTERACTIVE GUI (Streamlit interface)
streamlit run app.py
```

---

### 🌐 Live Deployment Links
- **Vercel Production Showcase**: [https://depth-wizard.vercel.app](https://depth-wizard.vercel.app)
- **GitHub Pages Interactive Showcase**: [https://lingeshmani23-tech.github.io/DepthWizard/](https://lingeshmani23-tech.github.io/DepthWizard/)
- **Streamlit Community Cloud Deployment**: Main file path `app/app.py`

---

## 🌟 Key Features & Research Novelty

1. **Pretrained Metric Depth Estimation**: Uses Hugging Face `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` to predict monocular depth in meters on an NVIDIA RTX 4050 GPU.
2. **Reference-Assisted Scale Calibration**: Monocular depth models suffer from inherent scale ambiguity. DEPTHWIZARD resolves this by using a known reference object (e.g., person of height $1.70\text{ m}$ or traffic pole) to calibrate the true metric scale.
3. **Pinhole 3D Geometry Back-Projection**: Target heights are measured using 3D perspective camera geometry:
   $$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = Z$$
4. **3D Scene Reconstruction**: Generates colored `.ply` metric point clouds with vertex colors and spatial bounds.
5. **Virtual Camera Flythrough**: Renders smooth 3D camera flythrough trajectories into playable MP4 videos (`flythrough.mp4`).
6. **Interactive Streamlit Web Dashboard**: User-friendly GUI for uploading photos, visual calibration, 3D point cloud inspection, flythrough video playback, and quantitative evaluation.

---

## 🏗️ System Architecture

```
Single Outdoor RGB Image
           │
           ▼
Depth Anything V2 Metric Outdoor (CUDA / PyTorch)
           │
           ▼
    Metric Depth Map (depth.npy / depth.png)
           │
           ▼
Reference-Assisted Scale Calibration (height_estimation.py)
           │
           ▼
   Calibrated Target Height Estimate (height_result.json)
           │
           ▼
 3D Point Cloud Reconstruction (create_pointcloud.py → pointcloud.ply)
           │
           ▼
 Virtual Camera Flythrough Video (flythrough.py → flythrough.mp4)
           │
           ▼
   Streamlit Web Application (app/app.py)
```

---

## 💻 Hardware & Environment

- **OS**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **CPU**: Intel Core i5-13420H
- **Python**: 3.13 / 3.12 64-bit
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4 enabled)

---

## 📁 Repository Structure

```
DepthWizard/
│
├── app/
│   └── app.py                 # Interactive Streamlit Web Application
├── scripts/
│   ├── depth_estimation.py    # Metric Depth Anything V2 inference engine
│   ├── height_estimation.py   # Reference scale calibration & 3D back-projection
│   ├── create_pointcloud.py   # 3D Point cloud generator (.ply)
│   ├── flythrough.py          # Virtual camera flythrough renderer (.mp4)
│   └── evaluation.py          # Quantitative error benchmarking
├── input/
│   └── test.jpg               # Input outdoor RGB image
├── output/
│   ├── depth.npy              # Metric depth NumPy array (meters)
│   ├── depth.png              # Colorized depth map visualization
│   ├── height_result.json     # Numerical height results & coordinates
│   ├── pointcloud.ply         # Reconstructed 3D point cloud
│   └── flythrough.mp4         # Virtual camera flythrough video
├── evaluation/
│   └── results.csv            # Error metrics and ground truth benchmark
├── config.py                  # Global configuration constants
├── requirements.txt            # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-username/DepthWizard.git
cd DepthWizard

py -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch with CUDA Support
```bash
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Dependencies
```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run Individual Pipeline Scripts

```bash
# Check GPU detection
venv\Scripts\python.exe -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Step 1: Metric Depth Estimation
venv\Scripts\python.exe scripts/depth_estimation.py

# Step 2: Reference-Assisted Height Estimation
venv\Scripts\python.exe scripts/height_estimation.py

# Step 3: 3D Point Cloud Reconstruction
venv\Scripts\python.exe scripts/create_pointcloud.py

# Step 4: Virtual Camera Flythrough Video
venv\Scripts\python.exe scripts/flythrough.py

# Step 5: Quantitative Error Evaluation
venv\Scripts\python.exe scripts/evaluation.py
```

### 5. Launch Interactive Streamlit App
```bash
streamlit run app/app.py
```

---

## 📊 Quantitative Evaluation

Benchmarked across outdoor samples with known reference and ground truth heights:

| Image | Reference Object | Known Height (m) | Actual Target Height (m) | Estimated Height (m) | Absolute Error (m) | Error % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `test.jpg` | Person | 1.70 m | 2.45 m | 2.39 m | 0.058 m | 2.37 % |
| `sample_street_scene.jpg` | Traffic Pole | 3.00 m | 2.60 m | 2.48 m | 0.125 m | 4.79 % |

**Overall Mean Absolute Percentage Error (MAPE)**: ~3.58 %

---

## ⚠️ Limitations & Future Work

- **Camera Intrinsics**: Uses field-of-view (FOV) approximations when exact camera focal length matrix is unknown.
- **Occlusions**: Monocular reconstruction back-projects single-view surfaces; back faces of objects are unobserved.
- **Future Work**: Integrate multi-view pose estimation, automatic target detection bounding boxes, and real-time WebGL point cloud rendering.

---

*Developed for Smart India Hackathon (SIH) Prototype Demonstration.*
