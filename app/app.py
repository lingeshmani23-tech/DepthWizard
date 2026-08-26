"""
DEPTHWIZARD — Single-View Height Estimation & 3D Flythrough
Smart India Hackathon (SIH) Prototype Application
Built with Streamlit, PyTorch, Depth Anything V2, OpenCV & Open3D/Trimesh
"""

import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import streamlit as st
import torch

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import config
from scripts.depth_estimation import run_depth_estimation
from scripts.height_estimation import estimate_height
from scripts.create_pointcloud import generate_pointcloud
from scripts.flythrough import render_flythrough
from scripts.evaluation import run_evaluation

# Page Configuration
st.set_page_config(
    page_title="DEPTHWIZARD — SIH Prototype",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark Mode Aesthetic
st.markdown("""
<style>
    /* Dark theme colors */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #F9FAFB !important;
        font-weight: 700 !important;
    }
    .header-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Card containers */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Main Header Section
st.markdown('<div class="header-title">🪄 DEPTHWIZARD</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Single-View Height Estimation & 3D Virtual Flythrough | SIH Prototype</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎮 Control Panel")
st.sidebar.markdown("---")

# GPU & Hardware Status Card
cuda_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Mode"
st.sidebar.success(f"⚡ Device: `{gpu_name}`" if cuda_available else "⚠️ CPU Execution Mode")
st.sidebar.info(f"🧠 Model: `Depth-Anything-V2-Metric-Outdoor-Small-hf`")

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input Image")
uploaded_file = st.sidebar.file_uploader("Upload Outdoor RGB Image", type=["jpg", "jpeg", "png"])

input_image_path = config.INPUT_DIR / "test.jpg"

if uploaded_file is not None:
    input_img = Image.open(uploaded_file).convert("RGB")
    input_img.save(input_image_path)
    st.sidebar.success("Custom image uploaded!")
else:
    if not input_image_path.exists():
        sample_src = BASE_DIR / "sample_images" / "sample_person_building.jpg"
        if sample_src.exists():
            input_img = Image.open(sample_src).convert("RGB")
            input_img.save(input_image_path)

if input_image_path.exists():
    input_img = Image.open(input_image_path).convert("RGB")
    st.sidebar.image(input_img, caption="Active Input Image", use_container_width=True)

# Main Navigation Tabs
tab_depth, tab_calib, tab_3d, tab_flythrough, tab_eval = st.tabs([
    "📷 Depth Estimation",
    "📐 Calibration & Height",
    "🧊 3D Point Cloud",
    "🎬 Virtual Flythrough",
    "📊 Benchmark Evaluation"
])

# ----------------------------------------------------
# TAB 1: DEPTH ESTIMATION
# ----------------------------------------------------
with tab_depth:
    st.header("Monocular Metric Depth Estimation")
    st.caption("Powered by Depth Anything V2 Metric Outdoor Model (Predicts metric depth in meters)")
    
    col_btn, col_blank = st.columns([1, 3])
    with col_btn:
        run_depth_btn = st.button("🚀 Run Depth Inference", key="depth_btn")
        
    depth_npy_path = config.OUTPUT_DIR / "depth.npy"
    depth_png_path = config.OUTPUT_DIR / "depth.png"
    
    if run_depth_btn or not depth_npy_path.exists():
        with st.spinner("Processing depth estimation with CUDA..."):
            try:
                res = run_depth_estimation(input_image_path)
                st.success("Depth map generated successfully!")
            except Exception as e:
                st.error(f"Depth Estimation Error: {e}")
                
    if depth_npy_path.exists() and depth_png_path.exists():
        depth_np = np.load(depth_npy_path)
        min_d = np.min(depth_np)
        max_d = np.max(depth_np)
        mean_d = np.mean(depth_np)
        
        # Display Statistics
        st.markdown("### Depth Map Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{min_d:.2f} m</div><div class="metric-label">Min Distance</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{max_d:.2f} m</div><div class="metric-label">Max Distance</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{mean_d:.2f} m</div><div class="metric-label">Mean Distance</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-value">{input_img.width}x{input_img.height}</div><div class="metric-label">Resolution</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        col_img, col_depth = st.columns(2)
        with col_img:
            st.image(input_img, caption="Original Outdoor RGB Image", use_container_width=True)
        with col_depth:
            st.image(Image.open(depth_png_path), caption="Metric Depth Map (Inferno Colormap)", use_container_width=True)

# ----------------------------------------------------
# TAB 2: REFERENCE-ASSISTED CALIBRATION & HEIGHT
# ----------------------------------------------------
with tab_calib:
    st.header("Reference-Assisted Scale Calibration & Height Estimation")
    st.caption("Select pixel coordinates for a reference object with known height to calibrate metric scale.")
    
    img_w, img_h = input_img.size
    
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.subheader("1️⃣ Reference Object Calibration")
        ref_real_h = st.number_input("Known Reference Real Height (meters)", min_value=0.1, max_value=20.0, value=1.70, step=0.05)
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            ref_x = st.slider("Ref X Coordinate", 0, img_w - 1, int(img_w * 0.35))
            ref_y_top = st.slider("Ref Top Y", 0, img_h - 1, int(img_h * 0.30))
        with col_r2:
            ref_y_bot = st.slider("Ref Bottom Y", 0, img_h - 1, int(img_h * 0.75))
            
        st.subheader("2️⃣ Target Object Measurement")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tgt_x = st.slider("Target X Coordinate", 0, img_w - 1, int(img_w * 0.70))
            tgt_y_top = st.slider("Target Top Y", 0, img_h - 1, int(img_h * 0.20))
        with col_t2:
            tgt_y_bot = st.slider("Target Bottom Y", 0, img_h - 1, int(img_h * 0.90))
            
        calc_height_btn = st.button("📏 Estimate Target Height", key="calc_h_btn")
        
    with c_right:
        st.subheader("Target & Reference Coordinates Preview")
        
        ref_top_pt = (ref_x, ref_y_top)
        ref_bot_pt = (ref_x, ref_y_bot)
        tgt_top_pt = (tgt_x, tgt_y_top)
        tgt_bot_pt = (tgt_x, tgt_y_bot)
        
        # Calculate Height
        if calc_height_btn or not (config.OUTPUT_DIR / "height_result.json").exists():
            try:
                res_h = estimate_height(
                    image_path=input_image_path,
                    depth_npy_path=depth_npy_path,
                    reference_top=ref_top_pt,
                    reference_bottom=ref_bot_pt,
                    reference_real_height_m=ref_real_h,
                    target_top=tgt_top_pt,
                    target_bottom=tgt_bot_pt
                )
            except Exception as e:
                st.error(f"Height Estimation Error: {e}")
                
        overlay_png = config.OUTPUT_DIR / "height_overlay.png"
        if overlay_png.exists():
            st.image(Image.open(overlay_png), caption="Interactive Reference (Green) & Target (Cyan) Overlay", use_container_width=True)
            
        json_res_path = config.OUTPUT_DIR / "height_result.json"
        if json_res_path.exists():
            with open(json_res_path) as f:
                h_data = json.load(f)
                
            st.markdown("### Estimation Results")
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><div class="metric-value">{h_data["calibration_scale"]:.4f}</div><div class="metric-label">Calibration Scale</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-value">{h_data["target_measurement"]:.2f} m</div><div class="metric-label">Raw Measurement</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#10B981;">{h_data["estimated_target_height_m"]:.2f} m</div><div class="metric-label">Calibrated Height</div></div>', unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 3: 3D POINT CLOUD RECONSTRUCTION
# ----------------------------------------------------
with tab_3d:
    st.header("3D Point Cloud Scene Reconstruction")
    st.caption("Back-projects metric depth and RGB pixels into an uncalibrated/calibrated 3D point cloud.")
    
    col_pc_btn, _ = st.columns([1, 3])
    with col_pc_btn:
        gen_pc_btn = st.button("🧊 Generate 3D Point Cloud", key="pc_btn")
        
    ply_path = config.OUTPUT_DIR / "pointcloud.ply"
    
    if gen_pc_btn or not ply_path.exists():
        with st.spinner("Generating 3D metric point cloud..."):
            try:
                generate_pointcloud(input_image_path, depth_npy_path, ply_path)
                st.success("3D Point Cloud created successfully!")
            except Exception as e:
                st.error(f"Point Cloud Error: {e}")
                
    if ply_path.exists():
        st.success(f"Point Cloud File Ready: `pointcloud.ply` ({ply_path.stat().st_size / 1024:.1f} KB)")
        
        with open(ply_path, "rb") as f:
            st.download_button(
                label="💾 Download 3D Point Cloud (.PLY)",
                data=f.read(),
                file_name="depthwizard_scene.ply",
                mime="application/octet-stream"
            )
            
        st.markdown("### 3D Scene Geometry Details")
        st.info("The exported `.ply` point cloud contains 3D camera coordinates (X, Y, Z in meters) with per-vertex RGB colors. It can be opened directly in MeshLab, CloudCompare, Blender, or Open3D.")

# ----------------------------------------------------
# TAB 4: VIRTUAL FLYTHROUGH
# ----------------------------------------------------
with tab_flythrough:
    st.header("Virtual Camera Flythrough Animation")
    st.caption("Generates a smooth 3D camera flythrough trajectory through the reconstructed 3D point cloud scene.")
    
    col_fly_btn, _ = st.columns([1, 3])
    with col_fly_btn:
        render_fly_btn = st.button("🎬 Render Flythrough Video", key="fly_btn")
        
    mp4_path = config.OUTPUT_DIR / "flythrough.mp4"
    
    if render_fly_btn or not mp4_path.exists():
        with st.spinner("Rendering smooth 3D camera flythrough video..."):
            try:
                render_flythrough(ply_path, mp4_path)
                st.success("Flythrough video rendered!")
            except Exception as e:
                st.error(f"Flythrough Rendering Error: {e}")
                
    if mp4_path.exists():
        st.video(str(mp4_path))
        st.caption("3D Virtual Camera Orbit & Flythrough (HTML5 MP4 Video)")

# ----------------------------------------------------
# TAB 5: EVALUATION & NOVELTY
# ----------------------------------------------------
with tab_eval:
    st.header("Quantitative Benchmark & Research Novelty")
    
    eval_csv_path = config.EVALUATION_DIR / "results.csv"
    
    if not eval_csv_path.exists():
        run_evaluation()
        
    if eval_csv_path.exists():
        df_eval = pd.read_csv(eval_csv_path)
        st.subheader("Height Estimation Error Analysis")
        st.dataframe(df_eval, use_container_width=True)
        
        if "Absolute Error (m)" in df_eval.columns:
            mae = df_eval["Absolute Error (m)"].mean()
            mape = df_eval["Error %"].mean()
            
            c_e1, c_e2 = st.columns(2)
            c_e1.markdown(f'<div class="metric-card"><div class="metric-value">{mae:.3f} m</div><div class="metric-label">Mean Absolute Error (MAE)</div></div>', unsafe_allow_html=True)
            c_e2.markdown(f'<div class="metric-card"><div class="metric-value">{mape:.2f} %</div><div class="metric-label">Mean Absolute Percentage Error (MAPE)</div></div>', unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("💡 Research Novelty & Scientific Grounding")
    st.markdown("""
    - **Monocular Scale Ambiguity Solution**: Standard monocular depth estimation outputs depth up to an unknown scale factor. DepthWizard resolves this ambiguity by combining pretrained monocular metric depth with **reference-assisted scale calibration**.
    - **Pinhole Camera Back-Projection**: Instead of naive 2D pixel height subtraction, DepthWizard projects pixel coordinates $(u, v)$ and depth $Z$ into metric 3D camera space $(X, Y, Z)$ using perspective camera geometry.
    - **3D Flythrough from Single Image**: Generates a smooth virtual camera trajectory and playable video flythrough directly from a single RGB image.
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6B7280; font-size: 0.85rem;'>DEPTHWIZARD Prototype | Built for Smart India Hackathon (SIH)</div>", unsafe_allow_html=True)
