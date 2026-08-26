"""
DEPTHWIZARD — Primary SIH Demonstration Entry Point
Executes zero-manual-input automated demonstration pipeline using preloaded config (demo_config.json).
Outputs formatted terminal logs, generates full report, and automatically opens FINAL_REPORT.html in browser.
"""

import sys
import os
import json
import time
import csv
import shutil
import webbrowser
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Ensure UTF-8 output encoding for Windows console compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import config
from src.image_utils import load_image, resize_depth_to_image
from src.depth_engine import estimate_depth, convert_to_distance_like_depth, save_depth, visualize_depth
from src.calibration import calibrate_scene, save_calibration_result
from src.height_estimator import estimate_height, save_height_result
from src.evaluation import evaluate_height, save_evaluation_csv
from src.pointcloud import create_point_cloud, save_point_cloud_ply
from src.flythrough import generate_flythrough
from src.visualization import generate_all_report_figures
from generate_report import generate_html_report


def print_banner():
    print("=" * 50)
    print("DEPTHWIZARD")
    print("Single-View Height Estimation & 3D Flythrough")
    print("SIH Demonstration Mode")
    print("=" * 50)


def run_sih_demo():
    print_banner()

    # 1. Load Demo Config & Image
    print("[1/8] Loading demo image.............. ", end="", flush=True)
    config_path = BASE_DIR / "demo_config.json"
    if not config_path.exists():
        print("✗")
        raise FileNotFoundError(f"Missing demo configuration at {config_path}")

    with open(config_path) as f:
        demo_cfg = json.load(f)

    img_path = BASE_DIR / demo_cfg["image"]
    if not img_path.exists():
        print("✗")
        raise FileNotFoundError(f"Demo image missing at {img_path}")

    pil_img, np_img = load_image(str(img_path))
    h, w = np_img.shape[:2]
    print("[OK]")

    # 2. Check GPU
    print("[2/8] Checking GPU.................... ", end="", flush=True)
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    print(f"[OK] ({gpu_name})")

    # Output Folder Setup
    report_dir = BASE_DIR / "outputs" / "demo_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    ref_top = tuple(demo_cfg["reference"]["top"])
    ref_bot = tuple(demo_cfg["reference"]["bottom"])
    ref_h = float(demo_cfg["reference"]["height_m"])

    tgt_top = tuple(demo_cfg["target"]["top"])
    tgt_bot = tuple(demo_cfg["target"]["bottom"])
    known_tgt_h = float(demo_cfg["target"]["known_height_m"])

    fov_deg = float(demo_cfg.get("camera", {}).get("fov_deg", config.DEFAULT_FOV_DEG))
    voxel_size = float(demo_cfg.get("processing", {}).get("voxel_size", 0.03))
    fps = int(demo_cfg.get("processing", {}).get("fps", 24))
    duration = int(demo_cfg.get("processing", {}).get("duration", 8))

    # Check Demo Mode (LIVE vs FAST)
    mode = getattr(config, "DEMO_MODE", "LIVE").upper()
    cache_npy = cache_dir / "depth.npy"
    cache_ply = cache_dir / "pointcloud.ply"

    use_fast = (mode == "FAST") and cache_npy.exists() and cache_ply.exists()

    # 3. Depth Estimation
    print("[3/8] Running Depth Anything V2...... ", end="", flush=True)
    if use_fast:
        dist_depth = np.load(cache_npy)
    else:
        raw_depth = estimate_depth(np_img)
        resized_raw_depth = resize_depth_to_image(raw_depth, (h, w))
        dist_depth = convert_to_distance_like_depth(resized_raw_depth)
        np.save(cache_npy, dist_depth)
    
    depth_stats = visualize_depth(dist_depth)
    print("[OK]")

    # 4. Reference Calibration
    print("[4/8] Calibrating scene............... ", end="", flush=True)
    calibration = calibrate_scene(
        depth_map=dist_depth,
        reference_top=ref_top,
        reference_bottom=ref_bot,
        reference_height_m=ref_h,
        fov_deg=fov_deg
    )
    print("[OK]")

    # 5. Target Height Estimation & Evaluation
    print("[5/8] Estimating target height........ ", end="", flush=True)
    height_res = estimate_height(
        depth_map=dist_depth,
        target_top=tgt_top,
        target_bottom=tgt_bot,
        calibration=calibration
    )
    
    eval_res = evaluate_height(
        image_name="sih_demo.jpg",
        estimated_height_m=height_res.estimated_height_m,
        known_height_m=known_tgt_h,
        reference_height_m=ref_h
    )
    print("[OK]")

    # 6. 3D Point Cloud Reconstruction
    print("[6/8] Building 3D point cloud........ ", end="", flush=True)
    metric_depth = dist_depth * calibration.scale_factor
    points_3d, colors_3d, pc_stats = create_point_cloud(
        rgb_img=np_img,
        depth_map=metric_depth,
        intrinsics=calibration.camera_intrinsics,
        voxel_size=voxel_size
    )

    ply_path = report_dir / "scene.ply"
    save_point_cloud_ply(points_3d, colors_3d, str(ply_path))
    save_point_cloud_ply(points_3d, colors_3d, str(cache_ply))
    print("[OK]")

    # 7. Render Virtual Flythrough
    print("[7/8] Rendering flythrough............ ", end="", flush=True)
    mp4_path = report_dir / "flythrough.mp4"
    generate_flythrough(
        points=points_3d,
        colors=colors_3d,
        output_path=str(mp4_path),
        duration_sec=duration,
        fps=fps
    )
    print("[OK]")

    # 8. Generate Final Report & Dashboard
    print("[8/8] Generating final report......... ", end="", flush=True)
    
    fig_paths = generate_all_report_figures(
        rgb_img=np_img,
        depth_map=dist_depth,
        reference_top=ref_top,
        reference_bottom=ref_bot,
        reference_height_m=ref_h,
        target_top=tgt_top,
        target_bottom=tgt_bot,
        estimated_height_m=height_res.estimated_height_m,
        known_target_height_m=known_tgt_h,
        points_3d=points_3d,
        colors_3d=colors_3d,
        flythrough_mp4_path=str(mp4_path),
        output_dir=str(report_dir)
    )

    results_data = {
        "input_image": "sample_images/sih_demo.jpg",
        "image_resolution": f"{w}x{h}",
        "reference": {
            "name": demo_cfg["reference"]["name"],
            "height_m": ref_h,
            "top": list(ref_top),
            "bottom": list(ref_bot)
        },
        "target": {
            "name": demo_cfg["target"]["name"],
            "known_height_m": known_tgt_h,
            "top": list(tgt_top),
            "bottom": list(tgt_bot)
        },
        "depth": depth_stats,
        "calibration": calibration.to_dict(),
        "height_estimation": height_res.to_dict(),
        "evaluation": eval_res.to_dict(),
        "pointcloud": {
            "num_points": len(points_3d),
            "ply_path": str(ply_path),
            "bounds": pc_stats.get("bounds", {})
        },
        "flythrough": {
            "mp4_path": str(mp4_path),
            "duration": duration,
            "fps": fps
        }
    }

    json_path = report_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    csv_path = report_dir / "results.csv"
    save_evaluation_csv([eval_res], str(csv_path))

    html_report_path = generate_html_report(results_data, str(report_dir))
    print("[OK]")

    # Terminal Final Results Summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Reference Object:    {demo_cfg['reference']['name']}")
    print(f"Reference Height:    {ref_h:.2f} m")
    print(f"Target Object:       {demo_cfg['target']['name']}")
    print(f"Estimated Height:    {height_res.estimated_height_m:.2f} m")
    print(f"Known Height:        {known_tgt_h:.2f} m")
    print(f"Absolute Error:      {eval_res.absolute_error_m:.3f} m")
    print(f"Percentage Error:    {eval_res.percentage_error:.2f} %")
    print(f"Point Cloud:         {len(points_3d):,} points")
    print(f"Flythrough:          outputs/demo_report/flythrough.mp4")
    print("=" * 50)
    print("DEMO COMPLETE")
    print("=" * 50)

    # Automatically open HTML report in browser
    report_uri = Path(html_report_path).resolve().as_uri()
    print(f"\nOpening SIH Demonstration Report in browser: {html_report_path}")
    webbrowser.open(report_uri)


if __name__ == "__main__":
    run_sih_demo()
