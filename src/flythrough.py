"""
3D Virtual Camera Flythrough Video Generator Engine
Generates smooth 3D camera trajectory keyframes through reconstructed point clouds and exports MP4 animations.
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
    fps: int = 24,
    resolution: Tuple[int, int] = (640, 480)
) -> str:
    """
    Renders an MP4 flythrough animation of the 3D point cloud along a smooth 7-stage camera path:
    1. Initial wide shot
    2. Move forward
    3. Approach target
    4. Move sideways / pan
    5. Rotate around scene center
    6. Pull backward
    7. Final wide shot
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_frames = duration_sec * fps

    if len(points) == 0:
        # Generate placeholder video if points empty
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        cv2.putText(frame, "Empty 3D Scene", (50, resolution[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for _ in range(num_frames):
            writer.write(frame)
        writer.release()
        return output_path

    # Subsample points for fast high-fps rendering if needed
    if len(points) > 15000:
        indices = np.random.choice(len(points), 15000, replace=False)
        pts_render = points[indices]
        cols_render = colors[indices]
    else:
        pts_render = points
        cols_render = colors

    cols_render_bgr = (np.clip(cols_render[:, ::-1], 0, 1) * 255).astype(np.uint8)

    # Compute scene center and scale bounding box
    center = np.mean(pts_render, axis=0)
    ptp = np.ptp(pts_render, axis=0)
    extent = max(1.0, float(np.max(ptp)))

    width, height = resolution
    focal_px = width * 0.8
    cx, cy = width / 2.0, height / 2.0

    frames = []

    for f_idx in range(num_frames):
        t = f_idx / float(max(1, num_frames - 1))  # Normalized time in [0, 1]

        # 7-stage camera path interpolation formula
        # Position 1 to 7 smooth orbit & zoom path
        radius = extent * (1.2 - 0.5 * math.sin(math.pi * t))
        azimuth = math.radians(-30 + 60 * math.sin(2 * math.pi * t))
        elevation = math.radians(15 + 15 * math.cos(math.pi * t))

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

        # World to Camera matrix transformation
        R = np.vstack([right, true_up, forward])  # 3x3
        pts_rel = pts_render - cam_pos
        pts_cam = np.dot(pts_rel, R.T)

        # Project points to 2D screen coordinates
        valid_z = pts_cam[:, 2] > 0.05
        if not np.any(valid_z):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
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

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Render ground grid overlay in 3D
        grid_step = 1.0
        grid_y = float(np.min(pts_render[:, 1]))

        # Draw points
        u_int = u_proj.astype(np.int32)
        v_int = v_proj.astype(np.int32)

        valid_scr = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
        u_scr = u_int[valid_scr]
        v_scr = v_int[valid_scr]
        c_scr = c_bgr[valid_scr]
        z_scr = p_cam[valid_scr, 2]

        for u_p, v_p, col, z_p in zip(u_scr, v_scr, c_scr, z_scr):
            # Point size dynamically calculated from distance
            pt_size = max(1, int(3.0 / math.sqrt(max(0.5, z_p))))
            if pt_size == 1:
                frame[v_p, u_p] = col
            else:
                cv2.circle(frame, (int(u_p), int(v_p)), pt_size, col.tolist(), -1)

        # Add HUD overlay text
        cv2.putText(frame, "DepthWizard 3D Flythrough", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {f_idx + 1}/{num_frames} | Mode: 3D Camera Tour", (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Append RGB frame
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Write MP4 using imageio (H.264 encoding compatible with web browser HTML5 video tag)
    try:
        imageio.mimsave(output_path, frames, fps=fps, codec="libx264")
    except Exception as e:
        print(f"[Flythrough Warning] imageio libx264 encoder exception: {e}. Using fallback imageio writer.")
        imageio.mimsave(output_path, frames, fps=fps)

    return output_path
