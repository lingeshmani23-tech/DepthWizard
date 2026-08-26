"""
Target Object Height Estimation Module
Computes estimated target height using pinhole back-projection geometry and calibrated metric scale.
"""

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any
import numpy as np

from .calibration import CalibrationResult, back_project_point, sample_depth_at_point


@dataclass
class HeightResult:
    target_top: Tuple[int, int]
    target_bottom: Tuple[int, int]
    target_pixel_height: float
    target_depth: float
    estimated_height_m: float
    height_ratio_estimate_m: float
    calibration_used: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_height(
    depth_map: np.ndarray,
    target_top: Tuple[int, int],
    target_bottom: Tuple[int, int],
    calibration: CalibrationResult
) -> HeightResult:
    """
    Estimate target height using reference-assisted calibration.

    Mathematical Formulation:
    1. Metric 3D Back-Projection:
       P1_metric = scale_factor * back_project(x1, y1, Z_target_top)
       P2_metric = scale_factor * back_project(x2, y2, Z_target_bottom)
       H_target = || P1_metric - P2_metric ||_2

    2. Perspective Relative Height Ratio Check:
       H_target_ratio ~ (P_target / P_reference) * (Z_target / Z_reference) * H_reference
    """
    h, w = depth_map.shape[:2]
    x1, y1 = target_top
    x2, y2 = target_bottom

    # Validate coordinate bounds
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))

    if y2 <= y1:
        y1, y2 = y2, y1
        x1, x2 = x2, x1

    pixel_height = float(math.hypot(x2 - x1, y2 - y1))
    if pixel_height < 1.0:
        pixel_height = 1.0

    # Sample relative depth
    z_top_rel = sample_depth_at_point(depth_map, x1, y1)
    z_bot_rel = sample_depth_at_point(depth_map, x2, y2)
    target_depth_rel = (z_top_rel + z_bot_rel) / 2.0

    intrinsics = calibration.camera_intrinsics
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    s = calibration.scale_factor

    # Back-project points in relative camera space
    p1_rel = back_project_point(x1, y1, z_top_rel, fx, fy, cx, cy)
    p2_rel = back_project_point(x2, y2, z_bot_rel, fx, fy, cx, cy)

    # Apply metric scale factor s
    p1_metric = s * p1_rel
    p2_metric = s * p2_rel

    # 3D Euclidean distance in metric meters
    estimated_height_m = float(np.linalg.norm(p1_metric - p2_metric))

    # Analytical ratio approximation: H_target ~ (P_target / P_ref) * (Z_target / Z_ref) * H_ref
    p_ref = calibration.reference_pixel_height
    z_ref = calibration.reference_depth
    h_ref = calibration.reference_height_m
    ratio_estimate = float((pixel_height / max(1.0, p_ref)) * (target_depth_rel / max(1e-4, z_ref)) * h_ref)

    return HeightResult(
        target_top=(int(x1), int(y1)),
        target_bottom=(int(x2), int(y2)),
        target_pixel_height=round(pixel_height, 2),
        target_depth=round(target_depth_rel, 4),
        estimated_height_m=round(estimated_height_m, 2),
        height_ratio_estimate_m=round(ratio_estimate, 2),
        calibration_used=calibration.to_dict()
    )


def save_height_result(result: HeightResult, output_dir: str) -> str:
    """
    Save height result JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "height_result.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return json_path
