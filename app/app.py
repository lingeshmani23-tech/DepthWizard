"""
DEPTHWIZARD — Single-View Height Estimation & 3D Flythrough
Lightweight Public Frontend (Streamlit)
Entry Point: streamlit run app.py
"""

import sys
import os
import json
import uuid
import math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import streamlit.components.v1 as components

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config
from src.image_utils import load_image, validate_image, resize_depth_to_image
from src.api.client import check_backend_health, analyze_image_remote, get_backend_url

# Fallback local computational modules if backend is running locally in single-process mode
try:
    from src.depth_engine import estimate_depth, convert_to_distance_like_depth, save_depth, visualize_depth, get_device_status
    from src.calibration import calibrate_scene
    from src.height_estimator import estimate_height
    from src.evaluation import evaluate_height
    from src.pointcloud import create_point_cloud, save_point_cloud_ply
    from src.flythrough import generate_flythrough
    LOCAL_PIPELINE_AVAILABLE = True
except Exception:
    LOCAL_PIPELINE_AVAILABLE = False

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="DEPTHWIZARD — Single-View Height & 3D Flythrough",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Engineering Styling (Clean Dark Mode)
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #F9FAFB !important;
        font-weight: 700 !important;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        margin-bottom: 2rem;
    }
    .header-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: #9CA3AF;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38BDF8 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-left: 4px solid #38BDF8;
        padding-left: 0.8rem;
        margin: 2rem 0 1rem 0;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
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
    .result-prominent {
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid #38BDF8;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #34D399;
        font-family: 'JetBrains Mono', monospace;
    }
    .result-label {
        font-size: 0.9rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .summary-box {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.6rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Session setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

session_dir = config.OUTPUT_DIR / f"session_{st.session_state.session_id}"
session_dir.mkdir(parents=True, exist_ok=True)

# Check Backend API Health
backend_url = get_backend_url()
backend_online, backend_info = check_backend_health(backend_url)

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="header-title">DEPTHWIZARD</div>
    <div class="header-subtitle">Single-View Height Estimation and 3D Flythrough</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR (Minimal Settings & Health Indicator)
# ----------------------------------------------------
st.sidebar.title("DepthWizard Controls")
if backend_online:
    st.sidebar.markdown("• **System Status:** `System Ready ✓` 🟢")
    st.sidebar.markdown(f"• **AI Backend:** `{backend_url}`")
    st.sidebar.markdown(f"• **Device:** `{backend_info.get('device_label', 'CPU')}`")
elif LOCAL_PIPELINE_AVAILABLE:
    dev_status = get_device_status()
    st.sidebar.markdown("• **System Status:** `Local Pipeline Active` 🟢")
    st.sidebar.markdown(f"• **Device:** `{dev_status['status_label']}`")
else:
    st.sidebar.markdown("• **System Status:** `Backend Offline` 🔴")

st.sidebar.markdown("**Depth Model:** `Depth Anything V2`")
st.sidebar.markdown("---")

with st.sidebar.expander("Advanced Settings"):
    fov_setting = st.slider("Vertical FOV (degrees)", min_value=30.0, max_value=120.0, value=float(config.DEFAULT_FOV_DEG), step=1.0)
    voxel_setting = st.slider("3D Voxel Size (m)", min_value=0.01, max_value=0.10, value=float(config.VOXEL_DOWN_SIZE), step=0.01)

# ----------------------------------------------------
# 01 — INPUT IMAGE
# ----------------------------------------------------
st.markdown('<div class="section-header">01 — INPUT IMAGE</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "UPLOAD RGB IMAGE",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a single outdoor photograph containing a reference object and target object."
)

input_image_path = session_dir / "input_image.jpg"

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    valid, msg, pil_img, np_img = validate_image(bytes_data)
    if valid:
        pil_img.save(input_image_path)
    else:
        st.error(f"Invalid image: {msg}")
        st.stop()
else:
    # Default sample image fallback
    sample_src = config.BASE_DIR / "sample_images" / "sih_demo.jpg"
    if sample_src.exists():
        pil_img, np_img = load_image(str(sample_src))
        pil_img.save(input_image_path)

if not input_image_path.exists():
    st.info("Please upload an outdoor RGB image to begin.")
    st.stop()

pil_img, np_img = load_image(str(input_image_path))
img_w, img_h = pil_img.size

col_img_prev, col_img_meta = st.columns([2, 1])
with col_img_prev:
    st.image(pil_img, caption=f"Uploaded Outdoor RGB Image ({img_w} x {img_h} px)", use_container_width=True)
with col_img_meta:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{img_w} x {img_h}</div><div class="metric-label">Resolution</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><div class="metric-value">RGB</div><div class="metric-label">Color Channels</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Read image bytes for API upload
with open(input_image_path, "rb") as f:
    img_bytes = f.read()

# ----------------------------------------------------
# 02 — DEPTH ESTIMATION
# ----------------------------------------------------
st.markdown('<div class="section-header">02 — DEPTH ESTIMATION</div>', unsafe_allow_html=True)

gen_depth_btn = st.button("Generate Depth", type="primary", use_container_width=True)

if gen_depth_btn or "dist_depth" not in st.session_state:
    if backend_online:
        with st.spinner("Executing Depth Anything V2 via AI Backend..."):
            success, api_res = analyze_image_remote(
                image_bytes=img_bytes,
                filename="input_image.jpg",
                fov_deg=fov_setting,
                voxel_size=voxel_setting,
                backend_url=backend_url
            )
            if success:
                st.session_state.api_res = api_res
                st.session_state.dist_depth_stats = api_res.get("depth_stats", {})
            else:
                st.error(f"Backend processing error: {api_res.get('error')}")
                st.stop()
    elif LOCAL_PIPELINE_AVAILABLE:
        with st.spinner("Processing Depth Anything V2 (Local Fallback)..."):
            raw_depth = estimate_depth(np_img)
            resized_raw_depth = resize_depth_to_image(raw_depth, (img_h, img_w))
            dist_depth = convert_to_distance_like_depth(resized_raw_depth)
            
            depth_npy_path = session_dir / "depth.npy"
            np.save(depth_npy_path, dist_depth)
            npy_p, png_p = save_depth(dist_depth, str(session_dir))
            
            st.session_state.dist_depth = dist_depth
            st.session_state.depth_png_path = png_p
            st.session_state.dist_depth_stats = visualize_depth(dist_depth)
    else:
        st.error("AI Processing Backend is currently offline. Please launch the backend service.")
        st.stop()

depth_stats = st.session_state.get("dist_depth_stats", {"min": 0.0, "max": 1.0, "mean": 0.5})

st.caption("RELATIVE DEPTH — Depth Anything V2")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.image(pil_img, caption="Original RGB Image", use_container_width=True)
with col_d2:
    if "api_res" in st.session_state and backend_online:
        depth_url = f"{backend_url}{st.session_state.api_res['media_urls']['depth_png']}"
        st.image(depth_url, caption="Relative Depth Map", use_container_width=True)
    elif "depth_png_path" in st.session_state and os.path.exists(st.session_state.depth_png_path):
        st.image(Image.open(st.session_state.depth_png_path), caption="Relative Depth Map", use_container_width=True)

cm1, cm2, cm3 = st.columns(3)
cm1.markdown(f'<div class="metric-card"><div class="metric-value">{depth_stats.get("min", 0.0):.2f}</div><div class="metric-label">Depth Minimum</div></div>', unsafe_allow_html=True)
cm2.markdown(f'<div class="metric-card"><div class="metric-value">{depth_stats.get("max", 0.0):.2f}</div><div class="metric-label">Depth Maximum</div></div>', unsafe_allow_html=True)
cm3.markdown(f'<div class="metric-card"><div class="metric-value">{depth_stats.get("mean", 0.0):.2f}</div><div class="metric-label">Depth Mean</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------
# 03 — REFERENCE CALIBRATION
# ----------------------------------------------------
st.markdown('<div class="section-header">03 — REFERENCE CALIBRATION</div>', unsafe_allow_html=True)

col_cal1, col_cal2 = st.columns(2)

with col_cal1:
    ref_obj_choice = st.selectbox(
        "Reference Object",
        ["Person (1.70 m)", "Car (1.50 m)", "Door (2.00 m)", "Custom Reference Object"]
    )
    
    if "Person" in ref_obj_choice:
        default_ref_h = 1.70
    elif "Car" in ref_obj_choice:
        default_ref_h = 1.50
    elif "Door" in ref_obj_choice:
        default_ref_h = 2.00
    else:
        default_ref_h = 1.70

    ref_height_input = st.number_input("Reference Height (m)", min_value=0.1, max_value=50.0, value=default_ref_h, step=0.05)

    default_rx = int(img_w * 0.35)
    default_ry1 = int(img_h * 0.30)
    default_ry2 = int(img_h * 0.75)

    ref_x = st.slider("Reference X (px)", 0, img_w - 1, default_rx)
    ref_top_y = st.slider("Reference Top Y (px)", 0, img_h - 1, default_ry1)
    ref_bot_y = st.slider("Reference Bottom Y (px)", 0, img_h - 1, default_ry2)

ref_top_pt = (ref_x, ref_top_y)
ref_bot_pt = (ref_x, ref_bot_y)

if backend_online and "api_res" in st.session_state:
    calib_dict = st.session_state.api_res.get("calibration", {})
elif LOCAL_PIPELINE_AVAILABLE and "dist_depth" in st.session_state:
    calib_obj = calibrate_scene(
        depth_map=st.session_state.dist_depth,
        reference_top=ref_top_pt,
        reference_bottom=ref_bot_pt,
        reference_height_m=ref_height_input,
        fov_deg=fov_setting
    )
    calib_dict = calib_obj.to_dict()
else:
    calib_dict = {"reference_pixel_height": 0.0, "reference_depth": 0.0, "scale_factor": 1.0}

with col_cal2:
    st.markdown("**Calibration Results**")
    st.write(f"• **Reference Pixel Height:** `{calib_dict.get('reference_pixel_height', 0.0):.1f} px`")
    st.write(f"• **Reference Relative Depth:** `{calib_dict.get('reference_depth', 0.0):.4f}`")
    st.write(f"• **Scale Factor ($S_{{calib}}$):** `{calib_dict.get('scale_factor', 1.0):.6f}`")
    st.write(f"• **Calibration Status:** `Calibrated ✓`")

st.markdown("---")

# ----------------------------------------------------
# 04 — HEIGHT ESTIMATION
# ----------------------------------------------------
st.markdown('<div class="section-header">04 — HEIGHT ESTIMATION</div>', unsafe_allow_html=True)
st.caption("TARGET OBJECT")

col_tgt1, col_tgt2 = st.columns(2)

with col_tgt1:
    default_tx = int(img_w * 0.70)
    default_ty1 = int(img_h * 0.20)
    default_ty2 = int(img_h * 0.90)

    tgt_x = st.slider("Target X (px)", 0, img_w - 1, default_tx)
    tgt_top_y = st.slider("Target Top Y (px)", 0, img_h - 1, default_ty1)
    tgt_bot_y = st.slider("Target Bottom Y (px)", 0, img_h - 1, default_ty2)

tgt_top_pt = (tgt_x, tgt_top_y)
tgt_bot_pt = (tgt_x, tgt_bot_y)

if backend_online and "api_res" in st.session_state:
    height_dict = st.session_state.api_res.get("height_result", {})
elif LOCAL_PIPELINE_AVAILABLE and "dist_depth" in st.session_state:
    height_obj = estimate_height(
        depth_map=st.session_state.dist_depth,
        target_top=tgt_top_pt,
        target_bottom=tgt_bot_pt,
        calibration=calib_obj
    )
    height_dict = height_obj.to_dict()
else:
    height_dict = {"target_pixel_height": 0.0, "target_depth": 0.0, "estimated_height_m": 0.0}

est_height_val = height_dict.get("estimated_height_m", 0.0)

with col_tgt2:
    st.write(f"• **Target Pixel Height:** `{height_dict.get('target_pixel_height', 0.0):.1f} px`")
    st.write(f"• **Target Relative Depth:** `{height_dict.get('target_depth', 0.0):.4f}`")
    
    st.markdown(f"""
    <div class="result-prominent">
        <div class="result-label">Approximate Estimated Height</div>
        <div class="result-value">{est_height_val:.2f} m</div>
    </div>
    """, unsafe_allow_html=True)

# Render Reference and Target Keypoints Overlay Image
overlay_img = cv2.cvtColor(np_img.copy(), cv2.COLOR_RGB2BGR)

# Reference Line (Green)
cv2.line(overlay_img, (ref_x, ref_top_y), (ref_x, ref_bot_y), (0, 255, 0), 3)
cv2.circle(overlay_img, (ref_x, ref_top_y), 6, (0, 255, 0), -1)
cv2.circle(overlay_img, (ref_x, ref_bot_y), 6, (0, 255, 0), -1)
cv2.putText(overlay_img, f"REF: {ref_height_input:.2f}m", (ref_x + 8, ref_top_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Target Line (Cyan)
cv2.line(overlay_img, (tgt_x, tgt_top_y), (tgt_x, tgt_bot_y), (255, 255, 0), 3)
cv2.circle(overlay_img, (tgt_x, tgt_top_y), 6, (255, 255, 0), -1)
cv2.circle(overlay_img, (tgt_x, tgt_bot_y), 6, (255, 255, 0), -1)
cv2.putText(overlay_img, f"TARGET: {est_height_val:.2f}m", (tgt_x + 8, tgt_top_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
st.image(overlay_rgb, caption="Calibrated Keypoints Overlay (Green = Reference Object, Cyan = Target Object)", use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 05 — EVALUATION
# ----------------------------------------------------
st.markdown('<div class="section-header">05 — EVALUATION</div>', unsafe_allow_html=True)

known_tgt_input = st.number_input("Known Target Height (optional ground-truth in meters)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

if known_tgt_input > 0.0:
    if LOCAL_PIPELINE_AVAILABLE:
        eval_res = evaluate_height(
            image_name="uploaded_image",
            estimated_height_m=est_height_val,
            known_height_m=known_tgt_input,
            reference_height_m=ref_height_input
        )
        abs_err = eval_res.absolute_error_m
        pct_err = eval_res.percentage_error
    else:
        abs_err = abs(est_height_val - known_tgt_input)
        pct_err = (abs_err / known_tgt_input) * 100.0

    ev1, ev2, ev3, ev4 = st.columns(4)
    ev1.markdown(f'<div class="metric-card"><div class="metric-value">{est_height_val:.2f} m</div><div class="metric-label">Estimated Height</div></div>', unsafe_allow_html=True)
    ev2.markdown(f'<div class="metric-card"><div class="metric-value">{known_tgt_input:.2f} m</div><div class="metric-label">Known Height</div></div>', unsafe_allow_html=True)
    ev3.markdown(f'<div class="metric-card"><div class="metric-value">{abs_err:.2f} m</div><div class="metric-label">Absolute Error</div></div>', unsafe_allow_html=True)
    ev4.markdown(f'<div class="metric-card"><div class="metric-value">{pct_err:.2f} %</div><div class="metric-label">Percentage Error</div></div>', unsafe_allow_html=True)
else:
    st.info("Ground-truth height not provided.")

st.markdown("---")

# ----------------------------------------------------
# 06 — 3D RECONSTRUCTION
# ----------------------------------------------------
st.markdown('<div class="section-header">06 — 3D RECONSTRUCTION</div>', unsafe_allow_html=True)

if backend_online and "api_res" in st.session_state:
    pc_data = st.session_state.api_res.get("pointcloud", {})
    pts_sample = pc_data.get("sample_points", [])
    cols_sample = pc_data.get("sample_colors", [])
    num_pts = pc_data.get("num_points", len(pts_sample))
    ply_download_url = f"{backend_url}{st.session_state.api_res['media_urls']['ply_file']}"
elif LOCAL_PIPELINE_AVAILABLE and "dist_depth" in st.session_state:
    metric_depth = st.session_state.dist_depth * calib_obj.scale_factor
    ply_path = session_dir / "scene.ply"

    points_3d, colors_3d, pc_stats = create_point_cloud(
        rgb_img=np_img,
        depth_map=metric_depth,
        intrinsics=calib_obj.camera_intrinsics,
        voxel_size=voxel_setting
    )
    save_point_cloud_ply(points_3d, colors_3d, str(ply_path))
    num_pts = len(points_3d)
    pts_sample = points_3d.tolist()
    cols_sample = colors_3d.tolist()
    ply_download_url = None
else:
    num_pts = 0
    pts_sample, cols_sample = [], []
    ply_download_url = None

st.write(f"• **Point Count:** `{num_pts:,}` 3D metric vertices")

# Interactive Three.js WebGL Point Cloud Component
def render_threejs_pointcloud(positions, colors):
    if len(positions) == 0:
        return "<p style='color:white;'>Empty Point Cloud</p>"
    
    pts_js = json.dumps(positions)
    cols_js = json.dumps(colors)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #05070D; }}
            #canvas3d {{ width: 100%; height: 450px; border-radius: 8px; }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas3d"></div>
        <script>
            const container = document.getElementById('canvas3d');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x05070D);

            const camera = new THREE.PerspectiveCamera(60, container.clientWidth / 450, 0.1, 1000);
            camera.position.set(0, 1.5, 4);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, 450);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            const positions = {pts_js};
            const colors = {cols_js};

            const geometry = new THREE.BufferGeometry();
            const posArray = new Float32Array(positions.length * 3);
            const colArray = new Float32Array(colors.length * 3);

            for(let i=0; i<positions.length; i++) {{
                posArray[i*3] = positions[i][0];
                posArray[i*3+1] = positions[i][1];
                posArray[i*3+2] = positions[i][2];

                colArray[i*3] = colors[i][0];
                colArray[i*3+1] = colors[i][1];
                colArray[i*3+2] = colors[i][2];
            }}

            geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colArray, 3));

            const material = new THREE.PointsMaterial({{ size: 0.035, vertexColors: true }});
            const pointCloud = new THREE.Points(geometry, material);
            scene.add(pointCloud);

            const gridHelper = new THREE.GridHelper(10, 20, 0x38BDF8, 0x1E293B);
            gridHelper.position.y = -2;
            scene.add(gridHelper);

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    return html_code

components.html(render_threejs_pointcloud(pts_sample, cols_sample), height=470)

if ply_download_url:
    st.markdown(f"[💾 Download PLY Point Cloud]({ply_download_url})")
elif 'ply_path' in locals() and ply_path.exists():
    with open(ply_path, "rb") as f:
        st.download_button(
            label="Download PLY Point Cloud",
            data=f.read(),
            file_name=f"depthwizard_scene_{st.session_state.session_id}.ply",
            mime="application/octet-stream"
        )

st.markdown("---")

# ----------------------------------------------------
# 07 — VIRTUAL FLYTHROUGH
# ----------------------------------------------------
st.markdown('<div class="section-header">07 — VIRTUAL FLYTHROUGH</div>', unsafe_allow_html=True)

if backend_online and "api_res" in st.session_state:
    mp4_url = f"{backend_url}{st.session_state.api_res['media_urls']['flythrough_mp4']}"
    st.video(mp4_url)
    st.markdown(f"[💾 Download Flythrough MP4]({mp4_url})")
elif LOCAL_PIPELINE_AVAILABLE and "dist_depth" in st.session_state:
    mp4_path = session_dir / "flythrough.mp4"
    if 'points_3d' in locals() and len(points_3d) > 0:
        if not mp4_path.exists():
            with st.spinner("Rendering 3D camera flythrough MP4 video..."):
                generate_flythrough(
                    points=points_3d,
                    colors=colors_3d,
                    output_path=str(mp4_path),
                    duration_sec=config.FLYTHROUGH_DURATION,
                    fps=config.FLYTHROUGH_FPS
                )

        if mp4_path.exists():
            st.video(str(mp4_path))
            with open(mp4_path, "rb") as vf:
                st.download_button(
                    label="Download Flythrough MP4",
                    data=vf.read(),
                    file_name=f"depthwizard_flythrough_{st.session_state.session_id}.mp4",
                    mime="video/mp4"
                )

st.markdown("---")

# ----------------------------------------------------
# 08 — FINAL RESULT
# ----------------------------------------------------
st.markdown('<div class="section-header">08 — FINAL RESULT</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="summary-box">
    <h3>Final Reconstruction & Estimation Summary</h3>
    <p>• <strong>Estimated Height:</strong> <span style="color:#34D399; font-weight:700; font-size:1.2rem;">{est_height_val:.2f} m</span></p>
    <p>• <strong>3D Reconstruction:</strong> Generated ✓ ({num_pts:,} vertices)</p>
    <p>• <strong>Virtual Flythrough:</strong> Generated ✓ (MP4 Video output)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------
# TECHNICAL DOCUMENTATION SECTIONS
# ----------------------------------------------------
st.markdown("#### NOVELTY")
st.write("Reference-assisted scale calibration combined with single-view height estimation, automatic 3D reconstruction and virtual flythrough from a single RGB image.")

with st.expander("HOW IT WORKS"):
    st.write("1. A single RGB image is processed using Depth Anything V2.")
    st.write("2. A known reference object provides a scale constraint.")
    st.write("3. The target object's approximate height is estimated.")
    st.write("4. RGB and depth are converted into a 3D point cloud.")
    st.write("5. A virtual camera generates a flythrough of the reconstructed scene.")

st.markdown("#### LIMITATIONS")
st.write("- Monocular depth provides relative depth.")
st.write("- Reference-assisted calibration determines approximate scale.")
st.write("- Perspective and occlusion can affect height estimation.")
st.write("- Single-view 3D reconstruction is incomplete.")
st.write("- Results are approximate rather than survey-grade measurements.")

# ----------------------------------------------------
# VERCEL & SERVERLESS EXPORT COMPATIBILITY
# Export top-level app, application, and handler symbols for platform inspection.
# ----------------------------------------------------
def handler(request=None, response=None):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": "<h1>DEPTHWIZARD Application</h1><p>Run via Streamlit: <code>streamlit run app.py</code></p>"
    }

app = handler
application = handler
