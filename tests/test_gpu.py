import unittest
from src.gpu_utils import get_device, get_gpu_info

class TestGPUUtils(unittest.TestCase):
    def test_get_device(self):
        device = get_device()
        self.assertIn(device, ["cuda", "cpu"])

    def test_get_gpu_info(self):
        info = get_gpu_info()
        self.assertIn("pytorch_version", info)
        self.assertIn("cuda_available", info)
        self.assertIn("device_str", info)

if __name__ == "__main__":
    unittest.main()
