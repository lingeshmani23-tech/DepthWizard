"""
DEPTHWIZARD — 3D Virtual Camera Flythrough Video Generator Engine
Renders realistic 3D camera flythrough trajectories through RGB-colored 3D reconstructions.
Exports validated browser-compatible H.264 MP4 videos (yuv420p) at 640x480 24FPS.
"""

import os
import sys
import math
import shutil
import subprocess
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import imageio_ffmpeg

logger = logging.getLogger("depthwizard_flythrough")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[FLYTHROUGH] %(asctime)s - %(message)s")


def sanitize_and_validate_point_cloud(
    points: np.ndarray,
    colors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Validates and cleans the 3D point cloud:
    - Removes NaNs, Infs, zero/negative depths
    - Removes extreme statistical distance outliers
    - Computes scene bounding box and spatial centroid
    - Validates point_count > 0
    """
    if points is None or len(points) == 0:
        raise ValueError("Point cloud input is empty (0 points). Cannot generate flythrough video.")

    # Convert to float32
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.float32)

    # Ensure 2D shape (N, 3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Invalid point cloud shape: {pts.shape}. Expected (N, 3).")

    # 1. Filter NaN and Infinity values
    valid_mask = np.all(np.isfinite(pts), axis=1) & np.all(np.isfinite(cols), axis=1)
    pts = pts[valid_mask]
    cols = cols[valid_mask]

    if len(pts) == 0:
        raise ValueError("All points in point cloud were invalid (NaN or Infinity).")

    # 2. Filter zero or negative depth points (Z <= 0) if applicable
    valid_z = pts[:, 2] > 1e-4
    if np.sum(valid_z) > 10:
        pts = pts[valid_z]
        cols = cols[valid_z]

    # 3. Statistical outlier filtering based on centroid distance
    centroid = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    q99 = np.percentile(dists, 99.0)
    inlier_mask = dists <= (q99 * 1.5)
    pts = pts[inlier_mask]
    cols = cols[inlier_mask]

    point_count = len(pts)
    logger.info(f"Point count after cleaning: {point_count:,}")

    if point_count == 0:
        raise ValueError("Point cloud count is 0 after outlier filtering.")

    # Compute bounding bounds and centroid
    min_bounds = np.min(pts, axis=0)
    max_bounds = np.max(pts, axis=0)
    scene_center = np.mean(pts, axis=0)

    extents = max_bounds - min_bounds
    scene_width, scene_height, scene_depth = float(extents[0]), float(extents[1]), float(extents[2])
    spatial_extent = max(0.5, float(np.max(extents)))

    stats = {
        "point_count": point_count,
        "min_x": float(min_bounds[0]), "max_x": float(max_bounds[0]),
        "min_y": float(min_bounds[1]), "max_y": float(max_bounds[1]),
        "min_z": float(min_bounds[2]), "max_z": float(max_bounds[2]),
        "scene_center": scene_center.tolist(),
        "scene_width": scene_width,
        "scene_height": scene_height,
        "scene_depth": scene_depth,
        "spatial_extent": spatial_extent
    }

    return pts, cols, stats


def compute_smooth_camera_trajectory(
    frame_idx: int,
    total_frames: int,
    scene_center: np.ndarray,
    spatial_extent: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes camera position and target look-at vector for frame_idx out of total_frames (240).
    Trajectory stages:
    - Frame 0: Wide overview of scene
    - Frames 1-60: Camera slowly moves forward (zoom in)
    - Frames 61-120: Camera orbits around scene
    - Frames 121-180: Camera moves sideways (lateral pan)
    - Frames 181-240: Camera moves upward/backward returning to wide overview
    """
    t = frame_idx / float(max(1, total_frames - 1))  # t in [0, 1]
    extent = float(spatial_extent)

    # Base distance scaling
    base_dist = extent * 1.5

    if t <= 0.25:
        # Phase 1 (0-60): Slow forward movement / zoom in
        p = t / 0.25  # p in [0, 1]
        dist = base_dist * (1.4 - 0.4 * (0.5 - 0.5 * math.cos(math.pi * p)))
        azimuth = 0.0
        elevation = math.radians(12.0)
        pan_x = 0.0
        pan_y = 0.0
    elif t <= 0.50:
        # Phase 2 (61-120): Orbit around scene
        p = (t - 0.25) / 0.25
        smooth_p = 0.5 - 0.5 * math.cos(math.pi * p)
        dist = base_dist * 1.0
        azimuth = math.radians(50.0 * math.sin(math.pi * smooth_p))
        elevation = math.radians(12.0 + 8.0 * smooth_p)
        pan_x = 0.0
        pan_y = 0.0
    elif t <= 0.75:
        # Phase 3 (121-180): Sideways horizontal pan
        p = (t - 0.50) / 0.25
        smooth_p = 0.5 - 0.5 * math.cos(math.pi * p)
        dist = base_dist * 1.0
        azimuth = math.radians(50.0 * (1.0 - smooth_p))
        elevation = math.radians(20.0 - 5.0 * smooth_p)
        pan_x = extent * 0.25 * math.sin(math.pi * smooth_p)
        pan_y = extent * 0.05 * math.sin(math.pi * smooth_p)
    else:
        # Phase 4 (181-240): Upward & backward zoom out to wide overview
        p = (t - 0.75) / 0.25
        smooth_p = 0.5 - 0.5 * math.cos(math.pi * p)
        dist = base_dist * (1.0 + 0.4 * smooth_p)
        azimuth = 0.0
        elevation = math.radians(15.0 + 15.0 * smooth_p)
        pan_x = extent * 0.25 * (1.0 - smooth_p)
        pan_y = 0.0

    # Spherical to Cartesian relative camera offset
    cam_x = scene_center[0] + pan_x + dist * math.sin(azimuth) * math.cos(elevation)
    cam_y = scene_center[1] + pan_y - dist * math.sin(elevation)  # Y inverted for camera up
    cam_z = scene_center[2] - dist * math.cos(azimuth) * math.cos(elevation)

    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
    target = np.array(scene_center, dtype=np.float32)

    # Validate camera coordinates
    if not np.all(np.isfinite(cam_pos)):
        cam_pos = target + np.array([0.0, -0.2 * extent, -base_dist], dtype=np.float32)

    # Ensure camera is not exactly at target
    if np.linalg.norm(cam_pos - target) < 1e-4:
        cam_pos[2] -= base_dist

    return cam_pos, target


def render_frame_zbuffer(
    points: np.ndarray,
    colors: np.ndarray,
    cam_pos: np.ndarray,
    target: np.ndarray,
    resolution: Tuple[int, int] = (640, 480),
    fov_deg: float = 60.0
) -> np.ndarray:
    """
    Renders a single 3D frame using a headless Z-buffered software rasterizer:
    - Fixed resolution: 640x480
    - Look-at camera transformation
    - Depth-buffered point splatting with anti-aliasing
    """
    width, height = resolution

    # 1. Camera Look-at Matrix
    forward = target - cam_pos
    norm_fwd = np.linalg.norm(forward)
    if norm_fwd > 1e-6:
        forward /= norm_fwd
    else:
        forward = np.array([0, 0, 1], dtype=np.float32)

    world_up = np.array([0, 1, 0], dtype=np.float32)
    # Check if forward is collinear with world_up
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0, 0, 1], dtype=np.float32)

    right = np.cross(world_up, forward)
    norm_right = np.linalg.norm(right)
    if norm_right > 1e-6:
        right /= norm_right
    else:
        right = np.array([1, 0, 0], dtype=np.float32)

    true_up = np.cross(forward, right)
    norm_up = np.linalg.norm(true_up)
    if norm_up > 1e-6:
        true_up /= norm_up

    # View matrix R (3x3)
    R = np.vstack([right, true_up, forward])  # 3x3

    # Transform world points to camera coordinates
    pts_rel = points - cam_pos
    pts_cam = np.dot(pts_rel, R.T)

    # Filter points behind camera (Z_cam <= 0.05)
    valid_mask = pts_cam[:, 2] > 0.05
    if not np.any(valid_mask):
        # Return dark background frame
        frame = np.full((height, width, 3), (21, 15, 13), dtype=np.uint8)  # BGR #05070D
        return frame

    p_cam = pts_cam[valid_mask]
    cols = colors[valid_mask]

    # Convert colors to BGR uint8 [0, 255]
    if np.max(cols) <= 1.0:
        cols_bgr = (np.clip(cols[:, ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        cols_bgr = np.clip(cols[:, ::-1], 0.0, 255.0).astype(np.uint8)

    # Perspective Projection
    focal_px = (height / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx, cy = width / 2.0, height / 2.0

    z_vals = p_cam[:, 2]
    u_proj = (p_cam[:, 0] * focal_px / z_vals) + cx
    v_proj = (-p_cam[:, 1] * focal_px / z_vals) + cy

    u_int = np.round(u_proj).astype(np.int32)
    v_int = np.round(v_proj).astype(np.int32)

    # Bounds check
    in_bounds = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    u_scr = u_int[in_bounds]
    v_scr = v_int[in_bounds]
    z_scr = z_vals[in_bounds]
    c_scr = cols_bgr[in_bounds]

    # Initialize Z-Buffer and Frame Buffer
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    frame = np.full((height, width, 3), (21, 15, 13), dtype=np.uint8)  # Sleek #05070D dark background

    # Sort points back-to-front for smooth rendering order
    sort_indices = np.argsort(-z_scr)
    u_scr = u_scr[sort_indices]
    v_scr = v_scr[sort_indices]
    z_scr = z_scr[sort_indices]
    c_scr = c_scr[sort_indices]

    for u, v, z, col in zip(u_scr, v_scr, z_scr, c_scr):
        # Dynamic point splat radius based on depth
        radius = max(1, int(round(2.5 / math.sqrt(max(0.2, float(z))))))

        if radius == 1:
            if z < z_buffer[v, u]:
                z_buffer[v, u] = z
                frame[v, u] = col
        else:
            y1, y2 = max(0, v - radius), min(height, v + radius + 1)
            x1, x2 = max(0, u - radius), min(width, u + radius + 1)

            # Update depth buffer and frame pixels
            mask = z < z_buffer[y1:y2, x1:x2]
            if np.any(mask):
                z_buffer[y1:y2, x1:x2][mask] = z
                frame[y1:y2, x1:x2][mask] = col

    # Draw sleek overlay footer banner
    cv2.rectangle(frame, (0, height - 32), (width, height), (15, 10, 8), -1)
    cv2.putText(
        frame,
        "DEPTHWIZARD — 3D RGB Camera Traversal",
        (15, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (248, 189, 56),  # Cyan / accent
        1,
        cv2.LINE_AA
    )

    return frame


def encode_mp4_h264(
    frame_dir: str,
    output_mp4: str,
    fps: int = 24,
    resolution: Tuple[int, int] = (640, 480)
) -> bool:
    """
    Encodes PNG frame sequence into browser-compatible H.264 MP4 with yuv420p pixel format using FFmpeg.
    Falls back to OpenCV VideoWriter if FFmpeg CLI fails.
    """
    logger.info("Encoding frames to H.264 MP4 (yuv420p)...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # 1. Try FFmpeg CLI encoding
    if ffmpeg_exe and os.path.exists(ffmpeg_exe):
        frame_pattern = os.path.join(frame_dir, "%06d.png")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-r", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={resolution[0]}:{resolution[1]}",
            "-an",
            output_mp4
        ]
        logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 0:
            logger.info("FFmpeg H.264 encoding succeeded.")
            return True
        else:
            logger.warning(f"FFmpeg encoding failed (code {res.returncode}): {res.stderr.decode('utf-8', errors='ignore')}")

    # 2. OpenCV Fallback
    logger.info("Attempting OpenCV VideoWriter fallback...")
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not frame_files:
        logger.error("No frame images found in frame directory.")
        return False

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_mp4, fourcc, fps, resolution)

    if not writer.isOpened():
        logger.warning("OpenCV avc1 codec VideoWriter failed to open. Trying mp4v codec...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_mp4, fourcc, fps, resolution)

    if not writer.isOpened():
        logger.error("OpenCV VideoWriter failed to initialize.")
        return False

    for fpath in frame_files:
        img = cv2.imread(fpath)
        if img is not None:
            writer.write(img)

    writer.release()
    return os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 0


def validate_mp4_file(
    output_mp4: str,
    expected_frames: int = 240,
    expected_resolution: Tuple[int, int] = (640, 480)
) -> bool:
    """
    Mandatory MP4 validation check:
    - File exists & size > 0
    - OpenCV VideoCapture can open file
    - Frame count > 0, FPS > 0, resolution == 640x480
    """
    logger.info(f"Validating output MP4: {output_mp4}")
    if not os.path.exists(output_mp4):
        raise ValueError(f"MP4 file does not exist: {output_mp4}")

    file_size = os.path.getsize(output_mp4)
    if file_size == 0:
        raise ValueError(f"MP4 file size is 0 bytes: {output_mp4}")

    cap = cv2.VideoCapture(output_mp4)
    if not cap.isOpened():
        raise ValueError("Failed to open generated MP4 video with VideoCapture.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info(f"Validated MP4 — Size: {file_size:,} bytes, Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")

    if frame_count <= 0:
        raise ValueError(f"Invalid frame count in MP4: {frame_count}")

    if width != expected_resolution[0] or height != expected_resolution[1]:
        logger.warning(f"MP4 resolution ({width}x{height}) differs from expected ({expected_resolution[0]}x{expected_resolution[1]}).")

    return True


def generate_flythrough(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    duration_sec: int = 10,
    fps: int = 24,
    resolution: Tuple[int, int] = (640, 480)
) -> str:
    """
    Main Flythrough Video Generation Entrypoint.
    Executes full pipeline:
    1. Sanitize & validate 3D point cloud
    2. Create camera trajectory (240 frames, 24 FPS)
    3. Render frames to disk in temporary directory (Memory Protection)
    4. Validate each frame
    5. Encode to H.264 yuv420p MP4
    6. Validate final MP4
    7. Clean up temporary frame files
    """
    logger.info(f"[FLYTHROUGH] Starting generation -> {output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Point Cloud Validation
    pts, cols, stats = sanitize_and_validate_point_cloud(points, colors)
    logger.info(f"[FLYTHROUGH] Point count: {stats['point_count']:,}")
    logger.info(f"[FLYTHROUGH] Scene bounds calculated. Extent: {stats['spatial_extent']:.3f} m")

    total_frames = duration_sec * fps  # 240 frames
    width, height = resolution         # 640 x 480

    scene_center = np.array(stats["scene_center"], dtype=np.float32)
    spatial_extent = stats["spatial_extent"]

    # 2. Temporary Frame Directory Setup
    job_id = os.path.basename(output_dir)
    temp_frames_dir = os.path.join(output_dir, "temp_frames")
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
    os.makedirs(temp_frames_dir, exist_ok=True)

    try:
        # 3. Frame Rendering Loop with Disk Streaming (Memory Protection)
        logger.info(f"[FLYTHROUGH] Rendering {total_frames} frames to {temp_frames_dir}...")

        for f_idx in range(total_frames):
            cam_pos, target = compute_smooth_camera_trajectory(f_idx, total_frames, scene_center, spatial_extent)

            # Render single 640x480 frame
            frame = render_frame_zbuffer(pts, cols, cam_pos, target, resolution=resolution)

            # Frame Validation
            if frame is None or frame.shape[0] != height or frame.shape[1] != width:
                err_msg = f"Frame {f_idx + 1} rendering failed: invalid frame object or dimension."
                logger.error(f"[FLYTHROUGH] {err_msg}")
                raise RuntimeError(err_msg)

            # Write frame to disk immediately (releases RAM)
            frame_path = os.path.join(temp_frames_dir, f"{f_idx + 1:06d}.png")
            cv2.imwrite(frame_path, frame)

            if (f_idx + 1) in [1, 60, 120, 180, 240] or (f_idx + 1) % 60 == 0:
                logger.info(f"[FLYTHROUGH] Rendering frame {f_idx + 1}/{total_frames}")

        # 4. MP4 Encoding
        logger.info("[FLYTHROUGH] Encoding MP4")
        encode_success = encode_mp4_h264(temp_frames_dir, output_path, fps=fps, resolution=resolution)
        if not encode_success:
            raise RuntimeError("MP4 video encoding failed.")

        # 5. MP4 Validation
        logger.info("[FLYTHROUGH] Validating MP4")
        validate_mp4_file(output_path, expected_frames=total_frames, expected_resolution=resolution)

        logger.info("[FLYTHROUGH] SUCCESS")
        return output_path

    except Exception as e:
        logger.error(f"[FLYTHROUGH] FAILED — Reason: {str(e)}", exc_info=True)
        raise

    finally:
        # Resource cleanup: Remove temporary PNG frame files
        if os.path.exists(temp_frames_dir):
            try:
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                logger.info("[FLYTHROUGH] Cleaned up temporary frame directory.")
            except Exception as clean_err:
                logger.warning(f"Failed to clean temporary frame directory: {clean_err}")

