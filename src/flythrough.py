"""
3D Virtual Camera Flythrough Video Generator Engine
Renders realistic 3D camera flythrough trajectories through RGB-colored 3D reconstructions.
Exports H.264 MP4 videos using true RGB image colors (zero depth heatmap colormaps).
"""

import os
import math
from typing import Tuple, Optional
import numpy as np
import cv2
import imageio


def generate_flythrough(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    duration_sec: int = 6,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720)
) -> str:
    """
    Renders an MP4 flythrough animation of the 3D RGB point cloud / mesh along a smooth camera path:
    - Orbit around scene centroid
    - Elevation change & smooth pan
    - Realistic perspective rendering with true RGB colors
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_frames = duration_sec * fps

    width, height = resolution

    if len(points) == 0:
        # Generate placeholder video if points empty
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, "Empty 3D Scene", (width // 3, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for _ in range(num_frames):
            writer.write(frame)
        writer.release()
        return output_path

    # Subsample points for smooth 3D rendering if needed
    max_pts = 25000
    if len(points) > max_pts:
        indices = np.random.choice(len(points), max_pts, replace=False)
        pts_render = points[indices]
        cols_render = colors[indices]
    else:
        pts_render = points
        cols_render = colors

    # Convert normalized RGB [0, 1] to BGR [0, 255] for OpenCV/3D projection
    cols_render_bgr = (np.clip(cols_render[:, ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)

    # Compute scene center and spatial extent
    center = np.mean(pts_render, axis=0)
    ptp = np.ptp(pts_render, axis=0)
    extent = max(1.0, float(np.max(ptp)))

    focal_px = width * 0.85
    cx, cy = width / 2.0, height / 2.0

    frames = []

    for f_idx in range(num_frames):
        t = f_idx / float(max(1, num_frames - 1))  # Normalized time in [0, 1]

        # Smooth camera trajectory parameters:
        # Radius: Gentle zoom in and out
        radius = extent * (1.15 - 0.25 * math.sin(math.pi * t))
        # Azimuth: Smooth orbit rotation from -35 to +35 degrees
        azimuth = math.radians(-35.0 + 70.0 * math.sin(2.0 * math.pi * t * 0.5))
        # Elevation: Gentle pitch angle change from 12 to 24 degrees
        elevation = math.radians(18.0 + 8.0 * math.cos(2.0 * math.pi * t))

        cam_x = center[0] + radius * math.sin(azimuth) * math.cos(elevation)
        cam_y = center[1] + radius * math.sin(elevation)
        cam_z = center[2] - radius * math.cos(azimuth) * math.cos(elevation)

        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        # Look-at matrix towards scene center
        forward = center - cam_pos
        norm_fwd = np.linalg.norm(forward)
        if norm_fwd > 1e-6:
            forward /= norm_fwd
        else:
            forward = np.array([0, 0, 1], dtype=np.float32)

        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(up, forward)
        norm_r = np.linalg.norm(right)
        if norm_r > 1e-6:
            right /= norm_r
        else:
            right = np.array([1, 0, 0], dtype=np.float32)

        true_up = np.cross(forward, right)

        # World to Camera transformation matrix (3x3)
        R = np.vstack([right, true_up, forward])
        pts_rel = pts_render - cam_pos
        pts_cam = np.dot(pts_rel, R.T)

        # Project points to 2D screen coordinates
        valid_z = pts_cam[:, 2] > 0.05
        if not np.any(valid_z):
            frame = np.full((height, width, 3), (13, 15, 5), dtype=np.uint8)  # Sleek #05070D dark
            frames.append(frame)
            continue

        p_cam = pts_cam[valid_z]
        c_bgr = cols_render_bgr[valid_z]

        # Sort points back-to-front for proper depth painter's rendering
        sort_idx = np.argsort(-p_cam[:, 2])
        p_cam = p_cam[sort_idx]
        c_bgr = c_bgr[sort_idx]

        u_proj = (p_cam[:, 0] * focal_px / p_cam[:, 2]) + cx
        v_proj = (-p_cam[:, 1] * focal_px / p_cam[:, 2]) + cy  # Flip Y for screen

        frame = np.full((height, width, 3), (13, 15, 5), dtype=np.uint8)  # Sleek #05070D background

        # Draw points with distance-based dynamic sizing
        u_int = u_proj.astype(np.int32)
        v_int = v_proj.astype(np.int32)

        valid_scr = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
        u_scr = u_int[valid_scr]
        v_scr = v_int[valid_scr]
        c_scr = c_bgr[valid_scr]
        z_scr = p_cam[valid_scr, 2]

        for u_p, v_p, col, z_p in zip(u_scr, v_scr, c_scr, z_scr):
            pt_size = max(1, int(4.5 / math.sqrt(max(0.4, z_p))))
            if pt_size == 1:
                frame[v_p, u_p] = col
            else:
                cv2.circle(frame, (int(u_p), int(v_p)), pt_size, col.tolist(), -1)

        # Subtle dark gradient overlay footer
        cv2.rectangle(frame, (0, height - 40), (width, height), (8, 10, 15), -1)
        cv2.putText(frame, "DEPTHWIZARD — 3D RGB Camera Traversal", (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(frame, f"3D Scene Frame {f_idx + 1}/{num_frames}", (width - 240, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (243, 244, 246), 1, cv2.LINE_AA)

        # Append RGB frame
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Write MP4 using imageio libx264
    try:
        imageio.mimsave(output_path, frames, fps=fps, codec="libx264")
    except Exception as e:
        print(f"[Flythrough Warning] imageio libx264 encoder exception: {e}. Falling back to default writer.")
        imageio.mimsave(output_path, frames, fps=fps)

    return output_path
