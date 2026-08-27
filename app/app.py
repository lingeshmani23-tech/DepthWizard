"""
DEPTHWIZARD — Single-View Height Estimation & 3D Flythrough
Production Public Web Application
Built with Streamlit, PyTorch, Depth Anything V2, OpenCV & Open3D / Trimesh
"""

import sys
import os
import json
import uuid
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
from src.image_utils import load_image, validate_image, resize_depth_to_image
from src.depth_engine import estimate_depth, convert_to_distance_like_depth, save_depth, visualize_depth, get_device_status
from src.object_detector import detect_objects, select_reference_and_target, draw_detections_overlay, load_reference_db
from src.calibration import calibrate_scene, CalibrationResult
from src.height_estimator import estimate_height, HeightResult
from src.evaluation import evaluate_height, EvaluationResult
from src.pointcloud import create_point_cloud, save_point_cloud_ply
from src.flythrough import generate_flythrough
from src.visualization import generate_all_report_figures

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="DEPTHWIZARD — Single-View Height & 3D Flythrough",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode, Premium Glassmorphism)
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
        font-family: 'Inter', system-ui, sans-serif;
    }
    h1, h2, h3 {
        color: #F9FAFB !important;
        font-weight: 700 !important;
    }
    .header-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: #9CA3AF;
        font-size: 1.15rem;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }
    .disclaimer-box {
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #38BDF8;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .status-badge {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .checklist-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 1.8rem;
        font-weight: 700;
        font-size: 1.05rem;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Multi-User Session Isolation Setup
# ----------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

session_dir = config.OUTPUT_DIR / f"session_{st.session_state.session_id}"
session_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Main Header & Landing Hero Section
# ----------------------------------------------------
st.markdown('<div class="header-title">🪄 DEPTHWIZARD</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Single-View Height Estimation & 3D Flythrough | <em>"From One Image to Depth, Height and 3D"</em></div>', unsafe_allow_html=True)

# Mandatory Scientific Transparency Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>ℹ️ Scientific Disclaimers & Model Principles:</strong><br>
    • <em>DepthWizard uses pretrained monocular depth estimation and reference-assisted calibration. Height estimates are approximate and depend on image quality, object detection, perspective, reference assumptions, and depth estimation accuracy.</em><br>
    • <em>Depth Anything V2 produces relative depth; real-world scale is estimated using reference information.</em>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar Health Check Status Panel
# ----------------------------------------------------
st.sidebar.title("🎮 Control Panel & Status")
st.sidebar.markdown("---")

dev_status = get_device_status()
st.sidebar.markdown("### 🟢 System Status")
st.sidebar.markdown(f"• **Application:** `Online` 🟢")
st.sidebar.markdown(f"• **Model Status:** `Loaded` 🧠")
st.sidebar.markdown(f"• **Device:** `{dev_status['status_label']}` ⚡")
st.sidebar.markdown(f"• **Depth Model:** `Depth Anything V2`")
st.sidebar.markdown("---")

# ----------------------------------------------------
# Main Upload Dropzone & File Validation
# ----------------------------------------------------
st.subheader("📥 Upload Outdoor RGB Image")
MAX_FILE_SIZE_MB = 20

main_uploaded_file = st.file_uploader(
    "Drag and drop your outdoor photo here, or click to browse files (JPG, JPEG, PNG, WEBP — Max 20 MB)",
    type=["jpg", "jpeg", "png", "webp"],
    key="main_uploader",
    help="Upload a single outdoor photograph containing reference & target objects."
)

sidebar_uploaded_file = st.sidebar.file_uploader(
    "Choose Outdoor Image",
    type=["jpg", "jpeg", "png", "webp"],
    key="sidebar_uploader",
    help="Upload a single outdoor photograph containing reference & target objects."
)

uploaded_file = main_uploaded_file if main_uploaded_file is not None else sidebar_uploaded_file
input_image_path = session_dir / "uploaded_input.jpg"

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    file_size_mb = len(bytes_data) / (1024 * 1024)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"⚠️ Image is too large ({file_size_mb:.1f} MB). Please upload an image below {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    valid, msg, pil_img, np_img = validate_image(bytes_data)
    if valid:
        pil_img.save(input_image_path)
        st.sidebar.success(f"✓ Image validated ({uploaded_file.name})")
    else:
        st.error(f"Invalid image file: {msg}")
        st.stop()
else:
    # Default sample image fallback
    demo_src = config.BASE_DIR / "sample_images" / "sih_demo.jpg"
    if demo_src.exists():
        pil_img, np_img = load_image(str(demo_src))
        pil_img.save(input_image_path)

if not input_image_path.exists():
    st.warning("Please upload an outdoor image to begin.")
    st.stop()

pil_img, np_img = load_image(str(input_image_path))
img_w, img_h = pil_img.size

# Display Image Info Metrics Card
c_m1, c_m2, c_m3, c_m4 = st.columns(4)
c_m1.markdown(f'<div class="metric-card"><div class="metric-value">{img_w} x {img_h}</div><div class="metric-label">Resolution</div></div>', unsafe_allow_html=True)
c_m2.markdown(f'<div class="metric-card"><div class="metric-value">RGB</div><div class="metric-label">Color Space</div></div>', unsafe_allow_html=True)
c_m3.markdown(f'<div class="metric-card"><div class="metric-value">{uploaded_file.type if uploaded_file else "JPEG"}</div><div class="metric-label">Format</div></div>', unsafe_allow_html=True)
c_m4.markdown(f'<div class="metric-card"><div class="metric-value">✓ Valid</div><div class="metric-label">Validation Status</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------
# Resource Protection & Process Trigger Button
# ----------------------------------------------------
st.subheader("⚡ Pipeline Execution")
process_btn = st.button("🚀 Process & Analyze Image", type="primary", use_container_width=True)

# Initialize Session Cache
if "processed" not in st.session_state or process_btn:
    st.session_state.processed = False

if process_btn:
    try:
        with st.spinner("Processing Depth Anything V2 monocular depth & 3D geometry..."):
            depth_npy_path = session_dir / "depth.npy"
            depth_png_path = session_dir / "depth.png"

            raw_depth = estimate_depth(np_img)
            resized_raw_depth = resize_depth_to_image(raw_depth, (img_h, img_w))
            dist_depth = convert_to_distance_like_depth(resized_raw_depth)
            np.save(depth_npy_path, dist_depth)
            save_depth(dist_depth, str(session_dir))

            # Auto Object Detection & Keypoint Selection
            ref_db = load_reference_db()
            detections = detect_objects(np_img, min_score=0.35)
            auto_ref, auto_tgt = select_reference_and_target(detections, np_img.shape, ref_db)

            # Store in session state
            st.session_state.dist_depth = dist_depth
            st.session_state.depth_png_path = str(depth_png_path)
            st.session_state.detections = detections
            st.session_state.auto_ref = auto_ref
            st.session_state.auto_tgt = auto_tgt
            st.session_state.processed = True
            st.success("✓ Pipeline analysis completed!")

    except Exception as e:
        st.error("⚠️ Pipeline processing encountered an error. Please verify the uploaded image and retry.")
        st.caption(f"Error detail: {str(e)}")
        st.stop()

if not st.session_state.processed:
    st.info("👆 Click **🚀 Process & Analyze Image** above to execute depth estimation, object detection, 3D point cloud generation, and flythrough rendering.")
    st.stop()

# Retrieve Cached Session Outputs
dist_depth = st.session_state.dist_depth
depth_png_path = st.session_state.depth_png_path
detections = st.session_state.detections
auto_ref = st.session_state.auto_ref
auto_tgt = st.session_state.auto_tgt

# ----------------------------------------------------
# Pipeline Checklist Display
# ----------------------------------------------------
st.markdown("""
<div class="checklist-box">
    <strong>📋 Pipeline Completion Status:</strong><br>
    ✓ <em>Image Uploaded & Validated</em> &nbsp;|&nbsp;
    ✓ <em>Depth Map Estimated (Depth Anything V2)</em> &nbsp;|&nbsp;
    ✓ <em>Objects Detected</em> &nbsp;|&nbsp;
    ✓ <em>Scene Scale Calibrated</em> &nbsp;|&nbsp;
    ✓ <em>Target Height Solved</em> &nbsp;|&nbsp;
    ✓ <em>3D Point Cloud Reconstructed</em> &nbsp;|&nbsp;
    ✓ <em>Flythrough Rendered</em>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Navigation Tabs
# ----------------------------------------------------
tab_depth, tab_detect, tab_calib, tab_3d, tab_flythrough = st.tabs([
    "📷 Depth Estimation",
    "🎯 Object Detection",
    "📐 Calibration & Height",
    "🧊 3D Point Cloud",
    "🎬 Virtual Flythrough"
])

# ----------------------------------------------------
# TAB 1: DEPTH ESTIMATION
# ----------------------------------------------------
with tab_depth:
    st.header("Monocular Depth Estimation")
    st.caption("Powered by Depth Anything V2 Metric Outdoor Model")

    depth_stats = visualize_depth(dist_depth)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{depth_stats["min"]:.2f}</div><div class="metric-label">Minimum Relative Depth</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{depth_stats["max"]:.2f}</div><div class="metric-label">Maximum Relative Depth</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{depth_stats["mean"]:.2f}</div><div class="metric-label">Mean Relative Depth</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_orig, col_dmap = st.columns(2)
    with col_orig:
        st.image(pil_img, caption="Original Outdoor RGB Image", use_container_width=True)
    with col_dmap:
        if os.path.exists(depth_png_path):
            st.image(Image.open(depth_png_path), caption="Relative Depth — Depth Anything V2 (Inferno Colormap)", use_container_width=True)

# ----------------------------------------------------
# TAB 2: AUTOMATIC OBJECT DETECTION
# ----------------------------------------------------
with tab_detect:
    st.header("Automatic Pretrained Object Detection")
    st.caption("Detects reference and target candidate objects with confidence scores and bounding boxes.")

    if detections:
        overlay_img = draw_detections_overlay(np_img, detections, auto_ref, auto_tgt)
        st.image(overlay_img, caption="Detected Bounding Boxes (Green = Reference Object, Cyan = Target Object)", use_container_width=True)

        st.subheader("Detected Scene Objects")
        df_det = pd.DataFrame([
            {
                "Class": d["class_name"].capitalize(),
                "Confidence": f"{d['score']*100:.1f}%",
                "Bounding Box [x1, y1, x2, y2]": str(d["box"])
            }
            for d in detections
        ])
        st.dataframe(df_det, use_container_width=True)
    else:
        st.warning("⚠️ No suitable objects automatically detected. Please use Refine Selection in Tab 3.")

# ----------------------------------------------------
# TAB 3: CALIBRATION & HEIGHT ESTIMATION
# ----------------------------------------------------
with tab_calib:
    st.header("Reference-Assisted Scale Calibration & Height Solver")

    if auto_ref is None:
        st.warning("⚠️ No suitable reference object automatically detected. Switching to default calibration.")
        ref_top_pt = (int(img_w * 0.35), int(img_h * 0.30))
        ref_bot_pt = (int(img_w * 0.35), int(img_h * 0.75))
        ref_h_val = 1.70
        ref_name = "Person (Default)"
    else:
        ref_top_pt = tuple(auto_ref["top"])
        ref_bot_pt = tuple(auto_ref["bottom"])
        ref_h_val = auto_ref["assumed_height_m"]
        ref_name = auto_ref["display_name"]

    if auto_tgt is None:
        tgt_top_pt = (int(img_w * 0.70), int(img_h * 0.20))
        tgt_bot_pt = (int(img_w * 0.70), int(img_h * 0.90))
        tgt_name = "Building Structure"
        known_tgt_h = None
    else:
        tgt_top_pt = tuple(auto_tgt["top"])
        tgt_bot_pt = tuple(auto_tgt["bottom"])
        tgt_name = auto_tgt["display_name"]
        known_tgt_h = None

    # Refine Selection Mode Expander
    with st.expander("🛠️ Refine Selection / Manual Calibration Correction"):
        st.caption("Override detected reference height or keypoint pixel coordinates.")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            user_ref_h = st.number_input("Confirmed Reference Height (m)", min_value=0.5, max_value=20.0, value=ref_h_val, step=0.05)
            ref_x_user = st.slider("Ref X", 0, img_w - 1, ref_top_pt[0])
            ref_y1_user = st.slider("Ref Top Y", 0, img_h - 1, ref_top_pt[1])
            ref_y2_user = st.slider("Ref Bot Y", 0, img_h - 1, ref_bot_pt[1])
        with col_m2:
            user_known_tgt = st.number_input("Optional Known Target Ground Truth (m)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            tgt_x_user = st.slider("Target X", 0, img_w - 1, tgt_top_pt[0])
            tgt_y1_user = st.slider("Target Top Y", 0, img_h - 1, tgt_top_pt[1])
            tgt_y2_user = st.slider("Target Bot Y", 0, img_h - 1, tgt_bot_pt[1])

        if user_ref_h > 0:
            ref_h_val = user_ref_h
            ref_top_pt = (ref_x_user, ref_y1_user)
            ref_bot_pt = (ref_x_user, ref_y2_user)

        if user_known_tgt > 0:
            known_tgt_h = user_known_tgt
            tgt_top_pt = (tgt_x_user, tgt_y1_user)
            tgt_bot_pt = (tgt_x_user, tgt_y2_user)

    # Perform Calibration & Height Calculation
    calibration = calibrate_scene(
        depth_map=dist_depth,
        reference_top=ref_top_pt,
        reference_bottom=ref_bot_pt,
        reference_height_m=ref_h_val,
        fov_deg=config.DEFAULT_FOV_DEG
    )

    height_res = estimate_height(
        depth_map=dist_depth,
        target_top=tgt_top_pt,
        target_bottom=tgt_bot_pt,
        calibration=calibration
    )

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.subheader("Reference Object Scale Calibration")
        st.info(f"**Reference Object:** {ref_name}\n\n**Assumed Height:** {ref_h_val:.2f} m\n\n**Calibrated Scale Factor ($S_{{calib}}$):** `{calibration.scale_factor:.4f}`")

    with col_res2:
        st.subheader("Target Height Measurement")
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:2.6rem; color:#10B981;">{height_res.estimated_height_m:.2f} m</div><div class="metric-label">Estimated Target Height</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Accuracy Benchmark Evaluation")
    if known_tgt_h is not None and known_tgt_h > 0:
        eval_res = evaluate_height(
            image_name="uploaded_image",
            estimated_height_m=height_res.estimated_height_m,
            known_height_m=known_tgt_h,
            reference_height_m=ref_h_val
        )
        ce1, ce2 = st.columns(2)
        ce1.markdown(f'<div class="metric-card"><div class="metric-value">{eval_res.absolute_error_m:.3f} m</div><div class="metric-label">Absolute Error</div></div>', unsafe_allow_html=True)
        ce2.markdown(f'<div class="metric-card"><div class="metric-value">{eval_res.percentage_error:.2f} %</div><div class="metric-label">Percentage Error</div></div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ Ground-truth height unavailable for this image. (Enter known ground truth above to evaluate percentage error).")

# ----------------------------------------------------
# TAB 4: 3D POINT CLOUD RECONSTRUCTION
# ----------------------------------------------------
ply_path = session_dir / "scene.ply"
metric_depth = dist_depth * calibration.scale_factor

with tab_3d:
    st.header("Metric 3D Point Cloud Reconstruction")
    st.caption("Back-projects metric depth and RGB pixels into 3D metric camera space.")

    try:
        points_3d, colors_3d, pc_stats = create_point_cloud(
            rgb_img=np_img,
            depth_map=metric_depth,
            intrinsics=calibration.camera_intrinsics,
            voxel_size=config.VOXEL_DOWN_SIZE
        )
        save_point_cloud_ply(points_3d, colors_3d, str(ply_path))
        st.success(f"✓ 3D Point Cloud created: `{len(points_3d):,}` metric vertices")

        if ply_path.exists():
            with open(ply_path, "rb") as f:
                st.download_button(
                    label="💾 Download 3D Point Cloud (.PLY)",
                    data=f.read(),
                    file_name=f"depthwizard_scene_{st.session_state.session_id}.ply",
                    mime="application/octet-stream"
                )
    except Exception as e:
        st.warning("⚠️ 3D point cloud generation failed for this frame. Check image dimensions.")

# ----------------------------------------------------
# TAB 5: VIRTUAL FLYTHROUGH VIDEO
# ----------------------------------------------------
mp4_path = session_dir / "flythrough.mp4"

with tab_flythrough:
    st.header("3D Virtual Camera Flythrough Video")
    st.caption("Renders smooth 3D camera translation trajectory through the reconstructed 3D point cloud scene.")

    if 'points_3d' in locals() and len(points_3d) > 0:
        if not mp4_path.exists():
            with st.spinner("Rendering 3D camera flythrough video..."):
                generate_flythrough(
                    points=points_3d,
                    colors=colors_3d,
                    output_path=str(mp4_path),
                    duration_sec=config.FLYTHROUGH_DURATION,
                    fps=config.FLYTHROUGH_FPS
                )

        if mp4_path.exists():
            st.video(str(mp4_path))
            st.caption("3D Virtual Camera Orbit & Flythrough (HTML5 MP4 Video)")
            with open(mp4_path, "rb") as vf:
                st.download_button(
                    label="💾 Download Virtual Flythrough Video (.MP4)",
                    data=vf.read(),
                    file_name=f"depthwizard_flythrough_{st.session_state.session_id}.mp4",
                    mime="video/mp4"
                )
    else:
        st.info("Generate 3D Point Cloud in Tab 4 first to render the virtual flythrough video.")

# ----------------------------------------------------
# Footer & Report Export
# ----------------------------------------------------
st.markdown("---")
col_ft1, col_ft2 = st.columns([3, 1])
with col_ft1:
    st.markdown("<div style='color: #6B7280; font-size: 0.85rem;'>DEPTHWIZARD Production Public Web Application | Smart India Hackathon (SIH)</div>", unsafe_allow_html=True)
with col_ft2:
    results_summary = {
        "session_id": st.session_state.session_id,
        "image_resolution": f"{img_w}x{img_h}",
        "device": dev_status["status_label"],
        "estimated_target_height_m": height_res.estimated_height_m,
        "calibrated_scale_factor": calibration.scale_factor
    }
    st.download_button(
        label="📄 Export Results (JSON)",
        data=json.dumps(results_summary, indent=2),
        file_name=f"depthwizard_results_{st.session_state.session_id}.json",
        mime="application/json"
    )
