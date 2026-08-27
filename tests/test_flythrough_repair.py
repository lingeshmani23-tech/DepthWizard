"""
DEPTHWIZARD — 3D Flythrough Repair Verification Unit & Integration Test
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import cv2

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flythrough import (
    sanitize_and_validate_point_cloud,
    compute_smooth_camera_trajectory,
    render_frame_zbuffer,
    generate_flythrough,
    validate_mp4_file
)
from src.pointcloud import save_point_cloud_ply, load_point_cloud_ply


class TestFlythroughRepair(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Create synthetic 3D point cloud (5000 points)
        np.random.seed(42)
        n_pts = 5000
        x = np.random.uniform(-2.0, 2.0, n_pts)
        y = np.random.uniform(-1.5, 1.5, n_pts)
        z = np.random.uniform(1.0, 5.0, n_pts)
        self.points = np.column_stack([x, y, z]).astype(np.float32)
        self.colors = np.random.uniform(0.1, 0.9, (n_pts, 3)).astype(np.float32)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_point_cloud_cleaning_and_validation(self):
        """Test point cloud cleaning, NaN filtering, and bounding calculation."""
        # Add NaNs and Infs
        pts_dirty = self.points.copy()
        pts_dirty[0, 0] = np.nan
        pts_dirty[1, 1] = np.inf

        cols_dirty = self.colors.copy()
        cols_dirty[2, 0] = np.nan

        cleaned_pts, cleaned_cols, stats = sanitize_and_validate_point_cloud(pts_dirty, cols_dirty)
        self.assertGreater(len(cleaned_pts), 0)
        self.assertEqual(len(cleaned_pts), len(cleaned_cols))
        self.assertFalse(np.any(np.isnan(cleaned_pts)))
        self.assertFalse(np.any(np.isinf(cleaned_pts)))
        self.assertIn("scene_center", stats)
        self.assertGreater(stats["spatial_extent"], 0.0)

    def test_empty_point_cloud_rejection(self):
        """Test that empty point cloud input raises explicit ValueError."""
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        empty_cols = np.zeros((0, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            sanitize_and_validate_point_cloud(empty_pts, empty_cols)

    def test_camera_trajectory(self):
        """Test camera trajectory calculations for all 240 frames."""
        center = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        extent = 3.0
        for f_idx in range(240):
            cam_pos, target = compute_smooth_camera_trajectory(f_idx, 240, center, extent)
            self.assertTrue(np.all(np.isfinite(cam_pos)), f"Frame {f_idx} camera position contains NaN/Inf")
            self.assertTrue(np.all(np.isfinite(target)), f"Frame {f_idx} target contains NaN/Inf")
            self.assertGreater(np.linalg.norm(cam_pos - target), 1e-3)

    def test_single_frame_rendering(self):
        """Test rendering of a single 640x480 Z-buffered frame."""
        cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 3.0], dtype=np.float32)

        frame = render_frame_zbuffer(self.points, self.colors, cam_pos, target, resolution=(640, 480))
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_full_flythrough_video_generation_and_validation(self):
        """Test full MP4 video generation (240 frames) and MP4 validation."""
        output_mp4 = os.path.join(self.test_dir, "flythrough.mp4")
        res_path = generate_flythrough(
            points=self.points,
            colors=self.colors,
            output_path=output_mp4,
            duration_sec=10,
            fps=24,
            resolution=(640, 480)
        )
        self.assertTrue(os.path.exists(res_path))
        self.assertGreater(os.path.getsize(res_path), 0)

        # Validate MP4 file
        valid = validate_mp4_file(output_mp4, expected_frames=240, expected_resolution=(640, 480))
        self.assertTrue(valid)

        # Check OpenCV video properties
        cap = cv2.VideoCapture(output_mp4)
        self.assertTrue(cap.isOpened())
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.assertEqual(frame_count, 240)
        self.assertEqual(width, 640)
        self.assertEqual(height, 480)
        self.assertEqual(fps, 24.0)

    def test_ply_save_and_load(self):
        """Test saving and re-loading point cloud PLY files for video regeneration."""
        ply_path = os.path.join(self.test_dir, "test_scene.ply")
        save_point_cloud_ply(self.points, self.colors, ply_path)
        self.assertTrue(os.path.exists(ply_path))

        loaded_pts, loaded_cols = load_point_cloud_ply(ply_path)
        self.assertEqual(len(loaded_pts), len(self.points))
        self.assertEqual(len(loaded_cols), len(self.colors))


if __name__ == "__main__":
    unittest.main()
