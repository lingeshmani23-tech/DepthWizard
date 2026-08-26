"""
DEPTHWIZARD — Virtual Camera Flythrough Generator
Renders a smooth 3D camera flythrough trajectory from the 3D point cloud.
Exports playable H.264 / MP4 video to output/flythrough.mp4.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def load_ply_points(ply_path):
    """
    Loads 3D vertices and RGB colors from a PLY point cloud file.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"Point cloud file missing at {ply_path}")

    mesh_or_pc = trimesh.load(str(ply_path))
    if isinstance(mesh_or_pc, trimesh.PointCloud):
        vertices = np.array(mesh_or_pc.vertices)
        colors = np.array(mesh_or_pc.colors)[:, :3] / 255.0 if len(mesh_or_pc.colors) > 0 else None
    elif hasattr(mesh_or_pc, 'vertices'):
        vertices = np.array(mesh_or_pc.vertices)
        colors = np.array(mesh_or_pc.visual.vertex_colors)[:, :3] / 255.0 if hasattr(mesh_or_pc.visual, 'vertex_colors') and len(mesh_or_pc.visual.vertex_colors) > 0 else None
    else:
        raise ValueError("Could not parse vertices from PLY file")

    return vertices, colors

def render_flythrough(
    ply_path=None,
    output_mp4_path=None,
    duration_sec=config.FLYTHROUGH_DURATION,
    fps=config.FLYTHROUGH_FPS,
    resolution=(640, 480)
):
    """
    Generates and saves a virtual camera flythrough MP4 video.
    """
    if ply_path is None:
        ply_path = config.OUTPUT_DIR / "pointcloud.ply"
    else:
        ply_path = Path(ply_path)

    if output_mp4_path is None:
        output_mp4_path = config.OUTPUT_DIR / "flythrough.mp4"
    else:
        output_mp4_path = Path(output_mp4_path)

    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading point cloud from {ply_path.name}...")
    vertices, colors = load_ply_points(ply_path)

    # Subsample for rendering speed
    max_render_points = 5000
    if len(vertices) > max_render_points:
        indices = np.random.choice(len(vertices), max_render_points, replace=False)
        vertices = vertices[indices]
        if colors is not None:
            colors = colors[indices]

    if colors is None:
        colors = np.ones((len(vertices), 3)) * 0.7  # Default fallback color

    # Compute bounding center and ranges
    center = np.mean(vertices, axis=0)
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

    num_frames = int(duration_sec * fps)
    print(f"Rendering flythrough: {num_frames} frames at {fps} FPS ({duration_sec}s)...")

    # Setup Matplotlib 3D figure
    dpi = 100
    fig_w, fig_h = resolution[0] / dpi, resolution[1] / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Set background styling to sleek dark mode
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.grid(False)
    ax.axis('off')

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, resolution)

    for frame_idx in range(num_frames):
        t = frame_idx / float(num_frames)

        # Smooth flythrough camera trajectory:
        # 1. Pan azimuth angle smoothly from -30 deg to +30 deg and back
        azim = -30.0 + 60.0 * np.sin(2 * np.pi * t * 0.5)
        # 2. Oscillate elevation angle between +10 deg and +25 deg
        elev = 15.0 + 10.0 * np.cos(2 * np.pi * t)
        # 3. Zoom / forward motion
        zoom_dist = max_range * (1.2 - 0.3 * np.sin(np.pi * t))

        ax.clear()
        ax.set_facecolor('#0B0F19')
        ax.grid(False)
        ax.axis('off')

        # Scatter plot 3D points
        # Invert Y so up is UP in 3D plot
        ax.scatter(
            vertices[:, 0],
            vertices[:, 2],  # Z becomes depth axis
            -vertices[:, 1], # -Y becomes height axis
            c=colors,
            s=1.5,
            alpha=0.85
        )

        ax.view_init(elev=elev, azim=azim)
        
        # Set equal aspect ratio limits
        ax.set_xlim(center[0] - zoom_dist, center[0] + zoom_dist)
        ax.set_ylim(center[2] - zoom_dist, center[2] + zoom_dist)
        ax.set_zlim(-center[1] - zoom_dist, -center[1] + zoom_dist)

        fig.canvas.draw()

        # Extract RGB buffer from Matplotlib canvas
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        bgr_resized = cv2.resize(bgr, resolution)

        out_video.write(bgr_resized)

        if (frame_idx + 1) % 48 == 0 or frame_idx == num_frames - 1:
            print(f"Rendered frame {frame_idx + 1}/{num_frames}")

    out_video.release()
    plt.close(fig)

    print(f"Flythrough rendering complete! Saved to {output_mp4_path.name}")
    return output_mp4_path

if __name__ == "__main__":
    render_flythrough()
