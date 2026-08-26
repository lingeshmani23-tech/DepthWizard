import unittest
import numpy as np
from src.depth_engine import estimate_depth, convert_to_distance_like_depth, normalize_depth, visualize_depth

class TestDepthEngine(unittest.TestCase):
    def setUp(self):
        self.dummy_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        self.dummy_rgb[20:80, 20:80] = [100, 150, 200]

    def test_depth_estimation_shape(self):
        raw_depth = estimate_depth(self.dummy_rgb)
        self.assertEqual(raw_depth.shape, (100, 100))
        self.assertTrue(np.issubdtype(raw_depth.dtype, np.floating))

    def test_convert_to_distance_like_depth(self):
        raw_depth = estimate_depth(self.dummy_rgb)
        dist_depth = convert_to_distance_like_depth(raw_depth)
        self.assertEqual(dist_depth.shape, (100, 100))
        self.assertTrue(np.all(dist_depth > 0))

    def test_depth_stats(self):
        raw_depth = estimate_depth(self.dummy_rgb)
        stats = visualize_depth(raw_depth)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)

if __name__ == "__main__":
    unittest.main()
