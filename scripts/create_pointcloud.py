"""
DEPTHWIZARD — 3D Point Cloud Generator
Back-projects RGB-D image into 3D metric camera coordinates (X, Y, Z) with RGB colors.
Exports clean point cloud to output/pointcloud.ply.
"""

import sys
import math
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def save_ply_ascii(filepath, points, colors):
    """
    Saves points (N, 3) and colors (N, 3 uint8) to a PLY file.
    """
    num_points = len(points)
    header = f"""ply
format ascii 1.0
comment Created by DepthWizard
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filepath, "w") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")

def generate_pointcloud(
    image_path=None,
    depth_npy_path=None,
    output_ply_path=None,
    fov_deg=config.DEFAULT_FOV_DEG,
    max_depth=config.MAX_DEPTH_METERS,
    downsample_factor=2
):
    """
    Generates a 3D Point Cloud from RGB image and depth numpy array.
    """
    if image_path is None:
        image_path = config.INPUT_DIR / "test.jpg"
    else:
        image_path = Path(image_path)

    if depth_npy_path is None:
        depth_npy_path = config.OUTPUT_DIR / "depth.npy"
    else:
        depth_npy_path = Path(depth_npy_path)

    if output_ply_path is None:
        output_ply_path = config.OUTPUT_DIR / "pointcloud.ply"
    else:
        output_ply_path = Path(output_ply_path)

    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image missing at {image_path}")
    if not depth_npy_path.exists():
        raise FileNotFoundError(f"Depth file missing at {depth_npy_path}")

    # Load RGB image and Depth Array
    rgb_img = Image.open(image_path).convert("RGB")
    depth_map = np.load(depth_npy_path)

    W, H = rgb_img.size
    depth_H, depth_W = depth_map.shape

    if (W, H) != (depth_W, depth_H):
        rgb_img = rgb_img.resize((depth_W, depth_H), Image.Resampling.BILINEAR)
        W, H = depth_W, depth_H

    rgb_array = np.array(rgb_img)

    # Intrinsics calculation
    fov_rad = math.radians(fov_deg)
    fy = H / (2.0 * math.tan(fov_rad / 2.0))
    fx = fy
    cx = W / 2.0
    cy = H / 2.0

    # Downsample step for point cloud efficiency
    u_coords = np.arange(0, W, downsample_factor)
    v_coords = np.arange(0, H, downsample_factor)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    # Subsample depth map and RGB image
    Z = depth_map[v_grid, u_grid]
    colors = rgb_array[v_grid, u_grid]

    # Filter invalid, non-finite, zero or extreme depth points
    valid_mask = (Z > 0.1) & (Z <= max_depth) & np.isfinite(Z)

    u_valid = u_grid[valid_mask]
    v_valid = v_grid[valid_mask]
    Z_valid = Z[valid_mask]
    colors_valid = colors[valid_mask]

    # Back-projection to 3D metric coordinates
    X_valid = (u_valid - cx) * Z_valid / fx
    Y_valid = (v_valid - cy) * Z_valid / fy

    points_3d = np.stack([X_valid, Y_valid, Z_valid], axis=-1)

    print(f"Generated 3D Point Cloud: {len(points_3d)} valid points")
    print(f"X range: [{np.min(X_valid):.2f}, {np.max(X_valid):.2f}] m")
    print(f"Y range: [{np.min(Y_valid):.2f}, {np.max(Y_valid):.2f}] m")
    print(f"Z range: [{np.min(Z_valid):.2f}, {np.max(Z_valid):.2f}] m")

    # Save to PLY format using Trimesh or pure ASCII PLY
    try:
        pc = trimesh.PointCloud(vertices=points_3d, colors=colors_valid)
        pc.export(str(output_ply_path))
    except Exception as e:
        print(f"Trimesh export fallback notice ({e}), saving ASCII PLY...")
        save_ply_ascii(output_ply_path, points_3d, colors_valid)

    print(f"Saved point cloud to {output_ply_path.name}")

    return {
        "ply_path": output_ply_path,
        "num_points": len(points_3d),
        "bounds": {
            "min": [float(np.min(X_valid)), float(np.min(Y_valid)), float(np.min(Z_valid))],
            "max": [float(np.max(X_valid)), float(np.max(Y_valid)), float(np.max(Z_valid))]
        }
    }

if __name__ == "__main__":
    generate_pointcloud()
