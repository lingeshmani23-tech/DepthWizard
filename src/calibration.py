"""
Reference Object Calibration Module
Calculates the physical scale factor s mapping relative depth to metric meters.
"""

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class CalibrationResult:
    reference_top: Tuple[int, int]
    reference_bottom: Tuple[int, int]
    reference_pixel_height: float
    reference_depth: float
    reference_height_m: float
    scale_factor: float
    focal_length_px: float
    camera_intrinsics: Dict[str, float]
    calibration_method: str = "Pinhole Back-Projection Calibration"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_focal_length(img_shape: Tuple[int, int], fov_deg: float = 60.0) -> Tuple[float, float, float, float]:
    """
    Estimate pinhole camera intrinsic parameters (fx, fy, cx, cy) from image dimensions and FOV.
    """
    h, w = img_shape[:2]
    fov_rad = math.radians(fov_deg)
    fy = h / (2.0 * math.tan(fov_rad / 2.0))
    fx = fy * (w / float(h))
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


def back_project_point(u: float, v: float, z_rel: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Back-projects 2D pixel coordinate (u, v) with relative depth z_rel into 3D camera coordinate space.
    """
    x = (u - cx) * z_rel / fx
    y = (v - cy) * z_rel / fy
    z = z_rel
    return np.array([x, y, z], dtype=np.float32)


def sample_depth_at_point(depth_map: np.ndarray, x: int, y: int, radius: int = 2) -> float:
    """
    Sample relative depth using a small spatial kernel around point (x, y) for noise robustness.
    """
    h, w = depth_map.shape[:2]
    x_clamped = max(0, min(w - 1, int(x)))
    y_clamped = max(0, min(h - 1, int(y)))

    y_min = max(0, y_clamped - radius)
    y_max = min(h, y_clamped + radius + 1)
    x_min = max(0, x_clamped - radius)
    x_max = min(w, x_clamped + radius + 1)

    patch = depth_map[y_min:y_max, x_min:x_max]
    return float(np.median(patch))


def calibrate_scene(
    depth_map: np.ndarray,
    reference_top: Tuple[int, int],
    reference_bottom: Tuple[int, int],
    reference_height_m: float,
    fov_deg: float = 60.0
) -> CalibrationResult:
    """
    Calculates scene metric scale s such that Metric Depth = s * Relative Depth.
    """
    if reference_height_m <= 0:
        raise ValueError(f"Reference height must be > 0. Got {reference_height_m}")

    h, w = depth_map.shape[:2]
    x1, y1 = reference_top
    x2, y2 = reference_bottom

    # Validate coordinate bounds
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))

    if y2 <= y1:
        # Swap if top/bottom inverted
        y1, y2 = y2, y1
        x1, x2 = x2, x1

    pixel_height = float(math.hypot(x2 - x1, y2 - y1))
    if pixel_height < 1.0:
        pixel_height = 1.0

    # Sample depths
    z_top_rel = sample_depth_at_point(depth_map, x1, y1)
    z_bot_rel = sample_depth_at_point(depth_map, x2, y2)
    reference_depth_rel = (z_top_rel + z_bot_rel) / 2.0

    # Compute intrinsics
    fx, fy, cx, cy = compute_focal_length((h, w), fov_deg=fov_deg)

    # 3D Back projection in relative space
    p1_3d_rel = back_project_point(x1, y1, z_top_rel, fx, fy, cx, cy)
    p2_3d_rel = back_project_point(x2, y2, z_bot_rel, fx, fy, cx, cy)

    rel_3d_distance = float(np.linalg.norm(p1_3d_rel - p2_3d_rel))

    if rel_3d_distance < 1e-6:
        scale_factor = 1.0
    else:
        # Physical metric height constraint: s * rel_3d_distance = reference_height_m
        scale_factor = float(reference_height_m / rel_3d_distance)

    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "fov_deg": fov_deg}

    return CalibrationResult(
        reference_top=(int(x1), int(y1)),
        reference_bottom=(int(x2), int(y2)),
        reference_pixel_height=round(pixel_height, 2),
        reference_depth=round(reference_depth_rel, 4),
        reference_height_m=float(reference_height_m),
        scale_factor=round(scale_factor, 6),
        focal_length_px=round(fy, 2),
        camera_intrinsics=intrinsics
    )


def save_calibration_result(result: CalibrationResult, output_dir: str) -> str:
    """
    Save calibration JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "calibration.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return json_path
