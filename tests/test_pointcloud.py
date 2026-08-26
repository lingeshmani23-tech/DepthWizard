import unittest
import numpy as np
from src.pointcloud import create_point_cloud

class TestPointCloud(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((50, 50, 3), 128, dtype=np.uint8)
        self.depth = np.full((50, 50), 3.0, dtype=np.float32)
        self.intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 25.0, "cy": 25.0}

    def test_create_point_cloud(self):
        pts, cols, stats = create_point_cloud(
            rgb_img=self.rgb,
            depth_map=self.depth,
            intrinsics=self.intrinsics,
            voxel_size=0.1
        )
        self.assertGreater(len(pts), 0)
        self.assertEqual(pts.shape[1], 3)
        self.assertEqual(cols.shape[1], 3)
        self.assertIn("point_count", stats)

if __name__ == "__main__":
    unittest.main()
