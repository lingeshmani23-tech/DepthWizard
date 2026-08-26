"""
DEPTHWIZARD — Height Estimation Engine
Implements Reference-Assisted Scale Calibration and Pinhole 3D Back-Projection.
Takes reference top/bottom points, known reference height, and target top/bottom points.
Saves numerical output to output/height_result.json.
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def pixel_to_3d(u, v, Z, W, H, fov_deg=config.DEFAULT_FOV_DEG):
    """
    Back-projects pixel coordinate (u, v) with depth Z into 3D metric camera coordinates (X, Y, Z).
    """
    fov_rad = math.radians(fov_deg)
    fy = H / (2.0 * math.tan(fov_rad / 2.0))
    fx = fy  # Assume square pixels
    cx = W / 2.0
    cy = H / 2.0

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float64)

def calculate_3d_distance(pt1_2d, pt2_2d, depth_map, fov_deg=config.DEFAULT_FOV_DEG):
    """
    Computes 3D Euclidean distance between two pixel coordinates (u, v) using the depth map.
    """
    H, W = depth_map.shape
    u1, v1 = int(round(pt1_2d[0])), int(round(pt1_2d[1]))
    u2, v2 = int(round(pt2_2d[0])), int(round(pt2_2d[1]))

    # Clamp coordinates to image boundaries
    u1, v1 = max(0, min(W - 1, u1)), max(0, min(H - 1, v1))
    u2, v2 = max(0, min(W - 1, u2)), max(0, min(H - 1, v2))

    Z1 = float(depth_map[v1, u1])
    Z2 = float(depth_map[v2, u2])

    P1 = pixel_to_3d(u1, v1, Z1, W, H, fov_deg)
    P2 = pixel_to_3d(u2, v2, Z2, W, H, fov_deg)

    dist_3d = np.linalg.norm(P1 - P2)
    return float(dist_3d), P1, P2

def estimate_height(
    image_path=None,
    depth_npy_path=None,
    reference_top=(200, 150),
    reference_bottom=(200, 380),
    reference_real_height_m=1.70,
    target_top=(450, 100),
    target_bottom=(450, 480),
    fov_deg=config.DEFAULT_FOV_DEG,
    output_dir=None
):
    """
    Performs reference-assisted scale calibration and target height estimation.
    """
    if image_path is None:
        image_path = config.INPUT_DIR / "test.jpg"
    else:
        image_path = Path(image_path)

    if depth_npy_path is None:
        depth_npy_path = config.OUTPUT_DIR / "depth.npy"
    else:
        depth_npy_path = Path(depth_npy_path)

    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not depth_npy_path.exists():
        raise FileNotFoundError(f"Depth map array not found at {depth_npy_path}")

    depth_map = np.load(depth_npy_path)
    
    # 1. Compute 3D distance of reference object
    ref_meas, ref_p1, ref_p2 = calculate_3d_distance(reference_top, reference_bottom, depth_map, fov_deg)
    
    if ref_meas <= 0 or reference_real_height_m <= 0:
        raise ValueError("Reference measurement or reference height must be positive.")

    # 2. Compute calibration scale factor
    calibration_scale = reference_real_height_m / ref_meas

    # 3. Compute 3D distance of target object
    target_meas, tgt_p1, tgt_p2 = calculate_3d_distance(target_top, target_bottom, depth_map, fov_deg)

    # 4. Apply scale calibration to estimate target height
    estimated_target_height_m = target_meas * calibration_scale

    result = {
        "reference_height_m": float(reference_real_height_m),
        "reference_measurement": float(ref_meas),
        "calibration_scale": float(calibration_scale),
        "target_measurement": float(target_meas),
        "estimated_target_height_m": float(estimated_target_height_m),
        "reference_coords": {
            "top": [int(reference_top[0]), int(reference_top[1])],
            "bottom": [int(reference_bottom[0]), int(reference_bottom[1])]
        },
        "target_coords": {
            "top": [int(target_top[0]), int(target_top[1])],
            "bottom": [int(target_bottom[0]), int(target_bottom[1])]
        }
    }

    json_path = output_dir / "height_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print("--- Height Estimation Results ---")
    print(f"Reference Known Real Height: {reference_real_height_m:.2f} m")
    print(f"Reference Raw Depth Measurement: {ref_meas:.4f} m")
    print(f"Calculated Calibration Scale: {calibration_scale:.4f}")
    print(f"Target Raw Depth Measurement: {target_meas:.4f} m")
    print(f"Estimated Target Height: {estimated_target_height_m:.2f} m")
    print(f"Saved result to {json_path.name}")

    # Generate visual overlay image
    if image_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            # Draw Reference Line (Green)
            cv2.line(img, (int(reference_top[0]), int(reference_top[1])), 
                     (int(reference_bottom[0]), int(reference_bottom[1])), (0, 255, 0), 3)
            cv2.circle(img, (int(reference_top[0]), int(reference_top[1])), 6, (0, 255, 0), -1)
            cv2.circle(img, (int(reference_bottom[0]), int(reference_bottom[1])), 6, (0, 255, 0), -1)
            cv2.putText(img, f"Ref: {reference_real_height_m:.2f}m", 
                        (int(reference_top[0]) + 10, int(reference_top[1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw Target Line (Cyan)
            cv2.line(img, (int(target_top[0]), int(target_top[1])), 
                     (int(target_bottom[0]), int(target_bottom[1])), (255, 255, 0), 3)
            cv2.circle(img, (int(target_top[0]), int(target_top[1])), 6, (255, 255, 0), -1)
            cv2.circle(img, (int(target_bottom[0]), int(target_bottom[1])), 6, (255, 255, 0), -1)
            cv2.putText(img, f"Target Est: {estimated_target_height_m:.2f}m", 
                        (int(target_top[0]) + 10, int(target_top[1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            overlay_path = output_dir / "height_overlay.png"
            cv2.imwrite(str(overlay_path), img)

    return result

if __name__ == "__main__":
    estimate_height()
