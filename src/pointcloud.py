"""
3D Point Cloud Generator Module
Applies pinhole camera back-projection, voxel grid downsampling, outlier filtering,
and exports ASCII/binary PLY files for 3D viewers.
"""

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np


def create_point_cloud(
    rgb_img: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: Dict[str, float],
    voxel_size: float = 0.03,
    max_depth: float = 50.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Construct 3D Point Cloud (XYZ coordinates and normalized RGB colors) from RGB-D pair.

    Returns:
        Tuple of (points_xyz (N, 3), colors_rgb (N, 3), stats_dict)
    """
    h, w = depth_map.shape[:2]
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Meshgrid of pixel coordinates (u, v)
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Flatten arrays
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    z_flat = depth_map.flatten()
    rgb_flat = (rgb_img.reshape(-1, 3) / 255.0).astype(np.float32)

    # Valid depth filtering (eliminate NaN, Inf, non-positive, extreme values)
    valid_mask = (
        np.isfinite(z_flat) &
        (z_flat > 1e-4) &
        (z_flat <= max_depth)
    )

    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    colors_valid = rgb_flat[valid_mask]

    # Pinhole 3D projection formulas
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy

    # Note: OpenCV image Y goes downward, flip Y for standard 3D viewer orientation (Y up)
    y_valid = -y_valid

    points_valid = np.column_stack([x_valid, y_valid, z_valid]).astype(np.float32)

    # Voxel grid downsampling
    if voxel_size > 0 and len(points_valid) > 0:
        points_valid, colors_valid = voxel_downsample(points_valid, colors_valid, voxel_size)

    # Statistics
    if len(points_valid) > 0:
        stats = {
            "point_count": len(points_valid),
            "x_range": (float(np.min(points_valid[:, 0])), float(np.max(points_valid[:, 0]))),
            "y_range": (float(np.min(points_valid[:, 1])), float(np.max(points_valid[:, 1]))),
            "z_range": (float(np.min(points_valid[:, 2])), float(np.max(points_valid[:, 2]))),
            "width_m": float(np.ptp(points_valid[:, 0])),
            "height_m": float(np.ptp(points_valid[:, 1])),
            "depth_m": float(np.ptp(points_valid[:, 2]))
        }
    else:
        stats = {
            "point_count": 0,
            "x_range": (0.0, 0.0),
            "y_range": (0.0, 0.0),
            "z_range": (0.0, 0.0),
            "width_m": 0.0,
            "height_m": 0.0,
            "depth_m": 0.0
        }

    return points_valid, colors_valid, stats


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample 3D point cloud onto a uniform spatial grid.
    """
    if len(points) == 0:
        return points, colors

    # Compute voxel index for each point
    min_bound = np.min(points, axis=0)
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(np.int32)

    # Dictionary hash table for unique voxels
    voxel_dict = {}
    for idx, (v_idx, pt, col) in enumerate(zip(voxel_indices, points, colors)):
        key = tuple(v_idx)
        if key not in voxel_dict:
            voxel_dict[key] = ([pt], [col])
        else:
            voxel_dict[key][0].append(pt)
            voxel_dict[key][1].append(col)

    downsampled_points = []
    downsampled_colors = []
    for pts, cols in voxel_dict.values():
        downsampled_points.append(np.mean(pts, axis=0))
        downsampled_colors.append(np.mean(cols, axis=0))

    return np.array(downsampled_points, dtype=np.float32), np.array(downsampled_colors, dtype=np.float32)


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, output_path: str) -> str:
    """
    Export 3D point cloud to ASCII PLY file format readable by MeshLab, Blender, Open3D, etc.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_points = len(points)
    colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

    header = f"""ply
format ascii 1.0
comment Exported by DepthWizard 3D Engine
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(num_points):
            pt = points[i]
            col = colors_uint8[i]
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")

    return output_path
