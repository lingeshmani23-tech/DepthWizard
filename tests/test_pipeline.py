import unittest
import os
import numpy as np
from src.pipeline import run_pipeline, PipelineResult

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        self.rgb[10:90, 10:90] = [120, 160, 200]

    def test_run_pipeline(self):
        res = run_pipeline(
            image_input=self.rgb,
            image_name="test_scene.png",
            reference_top=(30, 20),
            reference_bottom=(30, 80),
            reference_height_m=1.70,
            target_top=(70, 10),
            target_bottom=(70, 90),
            known_target_height_m=2.20,
            flythrough_duration=2,
            flythrough_fps=10,
            base_output_dir="outputs/test_pipeline_run"
        )
        self.assertIsInstance(res, PipelineResult)
        self.assertTrue(os.path.exists(res.depth_npy_path))
        self.assertTrue(os.path.exists(res.calibration_json_path))
        self.assertTrue(os.path.exists(res.height_json_path))
        self.assertTrue(os.path.exists(res.pointcloud_ply_path))
        self.assertTrue(os.path.exists(res.flythrough_mp4_path))

if __name__ == "__main__":
    unittest.main()
