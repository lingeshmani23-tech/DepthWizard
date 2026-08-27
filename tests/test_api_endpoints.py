"""
DEPTHWIZARD — Backend API Routes Test Script
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
import config


class TestBackendAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.sample_img_path = config.BASE_DIR / "sample_images" / "sih_demo.jpg"

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)

    def test_full_pipeline_analyze_and_endpoints(self):
        if not self.sample_img_path.exists():
            self.skipTest("Sample image missing for API test.")

        with open(self.sample_img_path, "rb") as f:
            files = {"image": ("sih_demo.jpg", f, "image/jpeg")}
            data = {
                "reference_object": "Person",
                "reference_height_m": 1.70,
                "reference_x": 0,
                "reference_top_y": 0,
                "reference_bot_y": 0,
                "target_x": 0,
                "target_top_y": 0,
                "target_bot_y": 0,
                "fov_deg": 60.0,
                "voxel_size": 0.05
            }
            resp = self.client.post("/api/analyze", files=files, data=data)

        self.assertEqual(resp.status_code, 200, f"Analyze failed: {resp.text}")
        res_json = resp.json()
        self.assertEqual(res_json["status"], "success")

        session_id = res_json["session_id"]
        self.assertIn("flythrough", res_json)
        self.assertIn("url", res_json["flythrough"])
        self.assertIn("download_url", res_json["flythrough"])

        flythrough_url = res_json["flythrough"]["url"]
        download_url = res_json["flythrough"]["download_url"]

        # 1. Test streamable inline MP4 route GET /outputs/{job_id}/flythrough.mp4
        resp_stream = self.client.get(flythrough_url)
        self.assertEqual(resp_stream.status_code, 200)
        self.assertEqual(resp_stream.headers["content-type"], "video/mp4")
        self.assertGreater(len(resp_stream.content), 0)

        # 2. Test attachment download MP4 route GET /download/{job_id}/flythrough
        resp_dl = self.client.get(download_url)
        self.assertEqual(resp_dl.status_code, 200)
        self.assertEqual(resp_dl.headers["content-type"], "video/mp4")
        self.assertIn("attachment", resp_dl.headers.get("content-disposition", ""))
        self.assertGreater(len(resp_dl.content), 0)

        # 3. Test video regeneration endpoint POST /api/regenerate_flythrough
        resp_regen = self.client.post("/api/regenerate_flythrough", data={"session_id": session_id})
        self.assertEqual(resp_regen.status_code, 200)
        regen_json = resp_regen.json()
        self.assertEqual(regen_json["status"], "success")
        self.assertIn("flythrough", regen_json)


if __name__ == "__main__":
    unittest.main()
