import unittest
import numpy as np
from src.calibration import calibrate_scene, CalibrationResult

class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.depth_map = np.ones((100, 100), dtype=np.float32) * 5.0  # Constant depth 5 meters proxy

    def test_calibrate_scene(self):
        cal = calibrate_scene(
            depth_map=self.depth_map,
            reference_top=(50, 10),
            reference_bottom=(50, 90),
            reference_height_m=1.70,
            fov_deg=60.0
        )
        self.assertIsInstance(cal, CalibrationResult)
        self.assertGreater(cal.scale_factor, 0.0)
        self.assertEqual(cal.reference_height_m, 1.70)
        self.assertEqual(cal.reference_pixel_height, 80.0)

    def test_invalid_reference_height(self):
        with self.assertRaises(ValueError):
            calibrate_scene(
                depth_map=self.depth_map,
                reference_top=(50, 10),
                reference_bottom=(50, 90),
                reference_height_m=-1.0
            )

if __name__ == "__main__":
    unittest.main()
