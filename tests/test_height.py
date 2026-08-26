import unittest
import numpy as np
from src.calibration import calibrate_scene
from src.height_estimator import estimate_height, HeightResult

class TestHeightEstimator(unittest.TestCase):
    def setUp(self):
        self.depth_map = np.ones((100, 100), dtype=np.float32) * 5.0
        self.cal = calibrate_scene(
            depth_map=self.depth_map,
            reference_top=(30, 20),
            reference_bottom=(30, 80),
            reference_height_m=1.80
        )

    def test_estimate_height(self):
        h_res = estimate_height(
            depth_map=self.depth_map,
            target_top=(70, 20),
            target_bottom=(70, 80),
            calibration=self.cal
        )
        self.assertIsInstance(h_res, HeightResult)
        self.assertAlmostEqual(h_res.estimated_height_m, 1.80, delta=0.1)

if __name__ == "__main__":
    unittest.main()
