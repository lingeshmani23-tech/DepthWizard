"""
DEPTHWIZARD — AI Computation Backend API Server
FastAPI Service for Depth Anything V2, Scale Calibration, Height Solver, 3D Point Cloud, and Flythrough MP4
"""

import sys
import os
import uuid
import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config
from src.image_utils import load_image, validate_image, resize_depth_to_image
from src.depth_engine import estimate_depth, convert_to_distance_like_depth, save_depth, visualize_depth, get_device_status
from src.calibration import calibrate_scene
from src.height_estimator import estimate_height
from src.evaluation import evaluate_height
from src.pointcloud import create_point_cloud, save_point_cloud_ply
from src.flythrough import generate_flythrough

app = FastAPI(
    title="DepthWizard AI Computation Backend",
    description="REST API for Single-View Depth Estimation, Scale Calibration, Metric Height Estimation, 3D Reconstruction, and Camera Flythrough Rendering.",
    version="2.0.0"
)

# Enable CORS for frontend requests
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Preload Depth Anything V2 model once into memory on server startup.
    """
    print("[Backend Startup] Preloading Depth Anything V2 model weights...")
    dev_status = get_device_status()
    print(f"[Backend Startup] Compute Device: {dev_status['status_label']}")


@app.get("/")
def read_root():
    return {
        "service": "DepthWizard AI Computation Backend",
        "status": "online",
        "endpoints": ["/api/health", "/api/analyze"]
    }


@app.get("/api/health")
def health_check():
    """
    Return backend status, loaded model, and active compute device (GPU vs CPU).
    """
    dev_status = get_device_status()
    return {
        "status": "ok",
        "model": "Depth Anything V2",
        "device": dev_status["device"],
        "device_label": dev_status["status_label"],
        "gpu_available": dev_status["is_gpu"]
    }


@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    reference_object: str = Form("Person"),
    reference_height_m: float = Form(1.70),
    reference_x: int = Form(0),
    reference_top_y: int = Form(0),
    reference_bot_y: int = Form(0),
    target_x: int = Form(0),
    target_top_y: int = Form(0),
    target_bot_y: int = Form(0),
    known_target_height_m: Optional[float] = Form(0.0),
    fov_deg: float = Form(60.0),
    voxel_size: float = Form(0.03)
):
    """
    Execute full DepthWizard processing pipeline:
    RGB Upload -> Depth Anything V2 -> Scale Calibration -> Height Solver -> Open3D 3D Cloud -> Flythrough MP4
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = config.OUTPUT_DIR / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Read & Validate Uploaded Image
        image_bytes = await image.read()
        valid, msg, pil_img, np_img = validate_image(image_bytes)

        if not valid:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {msg}")

        img_h, img_w = np_img.shape[:2]
        input_image_path = session_dir / "input_image.jpg"
        pil_img.save(input_image_path)

        # 2. Depth Anything V2 Inference
        raw_depth = estimate_depth(np_img)
        resized_raw_depth = resize_depth_to_image(raw_depth, (img_h, img_w))
        dist_depth = convert_to_distance_like_depth(resized_raw_depth)
        
        depth_npy_path = session_dir / "depth.npy"
        np.save(depth_npy_path, dist_depth)
        _, depth_png_path = save_depth(dist_depth, str(session_dir))
        depth_stats = visualize_depth(dist_depth)

        # Default Keypoint Auto-Placement if 0 passed
        rx = reference_x if reference_x > 0 else int(img_w * 0.35)
        ry1 = reference_top_y if reference_top_y > 0 else int(img_h * 0.30)
        ry2 = reference_bot_y if reference_bot_y > 0 else int(img_h * 0.75)

        tx = target_x if target_x > 0 else int(img_w * 0.70)
        ty1 = target_top_y if target_top_y > 0 else int(img_h * 0.20)
        ty2 = target_bot_y if target_bot_y > 0 else int(img_h * 0.90)

        # 3. Reference Calibration
        ref_top_pt = (rx, ry1)
        ref_bot_pt = (rx, ry2)

        calibration = calibrate_scene(
            depth_map=dist_depth,
            reference_top=ref_top_pt,
            reference_bottom=ref_bot_pt,
            reference_height_m=reference_height_m,
            fov_deg=fov_deg
        )

        # 4. Target Height Estimation
        tgt_top_pt = (tx, ty1)
        tgt_bot_pt = (tx, ty2)

        height_res = estimate_height(
            depth_map=dist_depth,
            target_top=tgt_top_pt,
            target_bottom=tgt_bot_pt,
            calibration=calibration
        )

        # 5. Accuracy Evaluation (if ground truth supplied)
        evaluation_dict = None
        if known_target_height_m and known_target_height_m > 0:
            eval_res = evaluate_height(
                image_name=image.filename or "uploaded_image",
                estimated_height_m=height_res.estimated_height_m,
                known_height_m=known_target_height_m,
                reference_height_m=reference_height_m
            )
            evaluation_dict = eval_res.to_dict()

        # 6. 3D Point Cloud Reconstruction
        metric_depth = dist_depth * calibration.scale_factor
        ply_path = session_dir / "scene.ply"

        points_3d, colors_3d, pc_stats = create_point_cloud(
            rgb_img=np_img,
            depth_map=metric_depth,
            intrinsics=calibration.camera_intrinsics,
            voxel_size=voxel_size
        )
        save_point_cloud_ply(points_3d, colors_3d, str(ply_path))

        # 7. Virtual Flythrough MP4 Video Generation
        mp4_path = session_dir / "flythrough.mp4"
        generate_flythrough(
            points=points_3d,
            colors=colors_3d,
            output_path=str(mp4_path),
            duration_sec=config.FLYTHROUGH_DURATION,
            fps=config.FLYTHROUGH_FPS
        )

        # Base URL for static assets
        base_url = "/api/media"
        
        # Sample points array for WebGL Three.js interactive visualizer (subsampled to 12k max)
        max_pts = 12000
        if len(points_3d) > max_pts:
            idx = np.random.choice(len(points_3d), max_pts, replace=False)
            pts_sample = points_3d[idx].tolist()
            cols_sample = colors_3d[idx].tolist()
        else:
            pts_sample = points_3d.tolist()
            cols_sample = colors_3d.tolist()

        return JSONResponse(content={
            "status": "success",
            "session_id": session_id,
            "image_resolution": [img_w, img_h],
            "depth_stats": depth_stats,
            "calibration": calibration.to_dict(),
            "height_result": height_res.to_dict(),
            "evaluation": evaluation_dict,
            "pointcloud": {
                "num_points": len(points_3d),
                "ply_url": f"{base_url}/{session_id}/scene.ply",
                "sample_points": pts_sample,
                "sample_colors": cols_sample
            },
            "media_urls": {
                "depth_png": f"{base_url}/{session_id}/depth_visualization.png",
                "ply_file": f"{base_url}/{session_id}/scene.ply",
                "flythrough_mp4": f"{base_url}/{session_id}/flythrough.mp4"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/api/media/{session_id}/{filename}")
def get_media_file(session_id: str, filename: str):
    """
    Serve generated static media assets (depth image, PLY cloud, MP4 video).
    """
    file_path = config.OUTPUT_DIR / f"session_{session_id}" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Requested asset missing")

    if filename.endswith(".png"):
        mime = "image/png"
    elif filename.endswith(".mp4"):
        mime = "video/mp4"
    elif filename.endswith(".ply"):
        mime = "application/octet-stream"
    else:
        mime = "application/octet-stream"

    return FileResponse(str(file_path), media_type=mime)
