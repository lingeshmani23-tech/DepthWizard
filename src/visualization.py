"""
DEPTHWIZARD — SIH Visualization Engine
Generates annotated SIH report figures:
- 01_original.png
- 02_depth.png (Titled "Depth Anything V2 — Relative Depth" with colorbar)
- 03_calibration.png
- 04_height_estimation.png (Green reference & Cyan target callouts)
- 05_pointcloud.png
- 06_flythrough_preview.png
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_all_report_figures(
    rgb_img: np.ndarray,
    depth_map: np.ndarray,
    reference_top: Tuple[int, int],
    reference_bottom: Tuple[int, int],
    reference_height_m: float,
    target_top: Tuple[int, int],
    target_bottom: Tuple[int, int],
    estimated_height_m: float,
    known_target_height_m: Optional[float],
    points_3d: np.ndarray,
    colors_3d: np.ndarray,
    flythrough_mp4_path: str,
    output_dir: str
) -> Dict[str, str]:
    """
    Renders and saves all 6 required SIH demonstration figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "original": os.path.join(output_dir, "01_original.png"),
        "depth": os.path.join(output_dir, "02_depth.png"),
        "calibration": os.path.join(output_dir, "03_calibration.png"),
        "height": os.path.join(output_dir, "04_height_estimation.png"),
        "pointcloud": os.path.join(output_dir, "05_pointcloud.png"),
        "flythrough": os.path.join(output_dir, "06_flythrough_preview.png"),
    }

    # 1. Save 01_original.png
    Image.fromarray(rgb_img).save(paths["original"])

    # 2. Save 02_depth.png (Relative Depth visualization with matplotlib colorbar)
    d_min, d_max = depth_map.min(), depth_map.max()
    norm_depth = (depth_map - d_min) / (d_max - d_min + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    im = ax.imshow(norm_depth, cmap="inferno")
    ax.set_title("Depth Anything V2 — Relative Depth", fontsize=14, fontweight="bold", pad=12, color="#0F172A")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Depth (Disparity Scale)", fontsize=11, fontweight="semibold")
    
    # Add explicit footer note
    fig.text(0.5, 0.02, "Note: Raw output represents Relative Depth. Absolute metric scale is calibrated via reference object geometry.",
             ha="center", fontsize=9, fontstyle="italic", color="#475569")
    plt.tight_layout()
    plt.savefig(paths["depth"], bbox_inches="tight")
    plt.close(fig)

    # 3. Save 03_calibration.png (Reference Object Only Highlighted)
    calib_img = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_RGB2BGR)
    rx1, ry1 = int(reference_top[0]), int(reference_top[1])
    rx2, ry2 = int(reference_bottom[0]), int(reference_bottom[1])
    
    # Draw reference line & dots (Green)
    cv2.line(calib_img, (rx1, ry1), (rx2, ry2), (0, 230, 0), 3)
    cv2.circle(calib_img, (rx1, ry1), 7, (0, 255, 0), -1)
    cv2.circle(calib_img, (rx2, ry2), 7, (0, 255, 0), -1)
    
    # Text callout box
    cv2.rectangle(calib_img, (rx1 + 10, ry1 - 10), (rx1 + 220, ry1 + 35), (0, 0, 0), -1)
    cv2.putText(calib_img, f"Ref Height: {reference_height_m:.2f} m", (rx1 + 15, ry1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imwrite(paths["calibration"], calib_img)

    # 4. Save 04_height_estimation.png (Reference Green & Target Cyan Annotated Overlay)
    height_img = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_RGB2BGR)
    tx1, ty1 = int(target_top[0]), int(target_top[1])
    tx2, ty2 = int(target_bottom[0]), int(target_bottom[1])

    # Reference (Green)
    cv2.line(height_img, (rx1, ry1), (rx2, ry2), (0, 230, 0), 3)
    cv2.circle(height_img, (rx1, ry1), 7, (0, 255, 0), -1)
    cv2.circle(height_img, (rx2, ry2), 7, (0, 255, 0), -1)
    cv2.putText(height_img, f"REF: {reference_height_m:.2f}m", (rx1 - 70, ry1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Target (Cyan)
    cv2.line(height_img, (tx1, ty1), (tx2, ty2), (255, 255, 0), 3)
    cv2.circle(height_img, (tx1, ty1), 7, (255, 255, 0), -1)
    cv2.circle(height_img, (tx2, ty2), 7, (255, 255, 0), -1)
    cv2.putText(height_img, f"TARGET EST: {estimated_height_m:.2f}m", (tx1 - 80, ty1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if known_target_height_m:
        cv2.putText(height_img, f"Known: {known_target_height_m:.2f}m", (tx1 - 80, ty2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 250), 2)

    cv2.imwrite(paths["height"], height_img)

    # 5. Save 05_pointcloud.png (Rendered 3D Point Cloud Snapshot)
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.grid(False)
    ax.axis('off')

    # Subsample points for plot clarity
    sub_count = min(8000, len(points_3d))
    idx = np.random.choice(len(points_3d), sub_count, replace=False)
    sub_pts = points_3d[idx]
    sub_colors = colors_3d[idx] if len(colors_3d) > 0 else "cyan"

    ax.scatter(
        sub_pts[:, 0],
        sub_pts[:, 2],
        -sub_pts[:, 1],
        c=sub_colors,
        s=1.2,
        alpha=0.85
    )
    ax.view_init(elev=20, azim=-35)
    plt.title(f"3D Point Cloud Reconstruction ({len(points_3d):,} Metric Vertices)", color="#F3F4F6", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(paths["pointcloud"], bbox_inches="tight")
    plt.close(fig)

    # 6. Save 06_flythrough_preview.png (Extract middle frame from flythrough video)
    if os.path.exists(flythrough_mp4_path):
        cap = cv2.VideoCapture(flythrough_mp4_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(paths["flythrough"], frame)
        cap.release()

    if not os.path.exists(paths["flythrough"]):
        # Fallback to copy 05_pointcloud.png if video frame extraction fails
        Image.fromarray(rgb_img).save(paths["flythrough"])

    return paths
