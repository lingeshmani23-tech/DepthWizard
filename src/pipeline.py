"""
Complete Pipeline Orchestration Engine
Sequentially executes depth estimation, reference calibration, height solver,
evaluation, 3D point cloud generation, and flythrough MP4 rendering.
"""

import os
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

from .image_utils import load_image, resize_depth_to_image
from .depth_engine import estimate_depth, convert_to_distance_like_depth, save_depth, visualize_depth
from .calibration import calibrate_scene, save_calibration_result, CalibrationResult
from .height_estimator import estimate_height, save_height_result, HeightResult
from .evaluation import evaluate_height, save_evaluation_csv, EvaluationResult
from .pointcloud import create_point_cloud, save_point_cloud_ply
from .flythrough import generate_flythrough


@dataclass
class PipelineResult:
    image_name: str
    output_dir: str
    original_image_path: str
    depth_npy_path: str
    depth_png_path: str
    calibration_json_path: str
    height_json_path: str
    pointcloud_ply_path: str
    flythrough_mp4_path: str
    calibration: CalibrationResult
    height_result: HeightResult
    evaluation: Optional[EvaluationResult]
    point_cloud_stats: Dict[str, Any]
    processing_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.evaluation:
            d["evaluation"] = self.evaluation.to_dict()
        d["calibration"] = self.calibration.to_dict()
        d["height_result"] = self.height_result.to_dict()
        return d


def run_pipeline(
    image_input: Any,
    image_name: str,
    reference_top: Tuple[int, int],
    reference_bottom: Tuple[int, int],
    reference_height_m: float,
    target_top: Tuple[int, int],
    target_bottom: Tuple[int, int],
    known_target_height_m: Optional[float] = None,
    fov_deg: float = 60.0,
    voxel_size: float = 0.03,
    flythrough_duration: int = 6,
    flythrough_fps: int = 24,
    base_output_dir: str = "outputs",
    progress_callback: Optional[Any] = None
) -> PipelineResult:
    """
    Execute complete end-to-end processing pipeline for a single RGB image.
    """
    t_start = time.time()
    
    # 1. Output folder setup
    clean_name = os.path.splitext(os.path.basename(image_name))[0]
    output_dir = os.path.abspath(os.path.join(base_output_dir, clean_name))
    os.makedirs(output_dir, exist_ok=True)

    if progress_callback:
        progress_callback(10, "Loading and preprocessing image...")

    # 2. Image Loading
    pil_img, np_img = load_image(image_input)
    orig_path = os.path.join(output_dir, "original.png")
    pil_img.save(orig_path)

    if progress_callback:
        progress_callback(30, "Running Depth Anything V2 monocular depth inference...")

    # 3. Depth Inference & Normalization
    raw_depth = estimate_depth(np_img)
    resized_raw_depth = resize_depth_to_image(raw_depth, np_img.shape[:2])
    dist_depth = convert_to_distance_like_depth(resized_raw_depth)

    depth_npy_path, depth_png_path = save_depth(dist_depth, output_dir)

    if progress_callback:
        progress_callback(50, "Calibrating scene scale using reference object...")

    # 4. Reference Object Scale Calibration
    calibration = calibrate_scene(
        depth_map=dist_depth,
        reference_top=reference_top,
        reference_bottom=reference_bottom,
        reference_height_m=reference_height_m,
        fov_deg=fov_deg
    )
    calibration_json_path = save_calibration_result(calibration, output_dir)

    if progress_callback:
        progress_callback(65, "Estimating target height...")

    # 5. Target Object Height Estimation
    height_res = estimate_height(
        depth_map=dist_depth,
        target_top=target_top,
        target_bottom=target_bottom,
        calibration=calibration
    )
    height_json_path = save_height_result(height_res, output_dir)

    # 6. Evaluation (Optional)
    eval_res = None
    if known_target_height_m is not None and known_target_height_m > 0:
        eval_res = evaluate_height(
            image_name=clean_name,
            estimated_height_m=height_res.estimated_height_m,
            known_height_m=known_target_height_m,
            reference_height_m=reference_height_m
        )
        csv_path = os.path.join(base_output_dir, "evaluation_summary.csv")
        save_evaluation_csv([eval_res], csv_path)

    if progress_callback:
        progress_callback(80, "Constructing 3D point cloud...")

    # 7. Metric 3D Depth Map -> Point Cloud
    metric_depth = dist_depth * calibration.scale_factor
    points, colors, pc_stats = create_point_cloud(
        rgb_img=np_img,
        depth_map=metric_depth,
        intrinsics=calibration.camera_intrinsics,
        voxel_size=voxel_size
    )
    ply_path = os.path.join(output_dir, "scene.ply")
    save_point_cloud_ply(points, colors, ply_path)

    if progress_callback:
        progress_callback(90, "Rendering 3D Virtual Flythrough video...")

    # 8. Render 3D Flythrough Video
    mp4_path = os.path.join(output_dir, "flythrough.mp4")
    generate_flythrough(
        points=points,
        colors=colors,
        output_path=mp4_path,
        duration_sec=flythrough_duration,
        fps=flythrough_fps
    )

    if progress_callback:
        progress_callback(100, "Complete pipeline finished!")

    t_elapsed = round(time.time() - t_start, 2)

    return PipelineResult(
        image_name=clean_name,
        output_dir=output_dir,
        original_image_path=orig_path,
        depth_npy_path=depth_npy_path,
        depth_png_path=depth_png_path,
        calibration_json_path=calibration_json_path,
        height_json_path=height_json_path,
        pointcloud_ply_path=ply_path,
        flythrough_mp4_path=mp4_path,
        calibration=calibration,
        height_result=height_res,
        evaluation=eval_res,
        point_cloud_stats=pc_stats,
        processing_time_sec=t_elapsed
    )
