"""
DEPTHWIZARD — Virtual Camera Flythrough Generator Script
Renders a smooth 3D camera flythrough trajectory from the 3D RGB point cloud / mesh using true RGB colors.
Exports playable H.264 MP4 video.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.flythrough import generate_flythrough
from src.pointcloud import create_point_cloud
from src.image_utils import load_image

if __name__ == "__main__":
    test_img = config.BASE_DIR / "sample_images" / "sih_demo.jpg"
    if test_img.exists():
        pil_img, np_img = load_image(str(test_img))
        dummy_depth = np.ones((np_img.shape[0], np_img.shape[1]), dtype=np.float32) * 5.0
        intr = {"fx": 500.0, "fy": 500.0, "cx": np_img.shape[1]/2.0, "cy": np_img.shape[0]/2.0}
        pts, cols, _ = create_point_cloud(np_img, dummy_depth, intr)
        out_mp4 = config.OUTPUT_DIR / "flythrough.mp4"
        generate_flythrough(pts, cols, str(out_mp4))
        print(f"Flythrough generated at {out_mp4}")
