"""
DepthWizard Central Configuration Module
"""

import os
from pathlib import Path
import torch

# Base Directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
SCRIPTS_DIR = BASE_DIR / "scripts"
APP_DIR = BASE_DIR / "app"
ASSETS_DIR = BASE_DIR / "assets"
EVALUATION_DIR = BASE_DIR / "evaluation"
MODELS_DIR = BASE_DIR / "models"

# Ensure all essential directories exist
for folder in [INPUT_DIR, OUTPUT_DIR, SCRIPTS_DIR, APP_DIR, ASSETS_DIR, EVALUATION_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Model Settings
MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Camera Intrinsics Assumptions (for monocular 3D back-projection when intrinsics are uncalibrated)
# Default Vertical FOV ~ 60 degrees
DEFAULT_FOV_DEG = 60.0

# 3D Point Cloud Parameters
VOXEL_DOWN_SIZE = 0.05  # meter grid downsampling for Open3D
MAX_DEPTH_METERS = 80.0  # filter out sky / extreme background noise

# Camera Flythrough Parameters
FLYTHROUGH_DURATION = 8  # seconds
FLYTHROUGH_FPS = 24
FLYTHROUGH_NUM_FRAMES = FLYTHROUGH_DURATION * FLYTHROUGH_FPS
