"""
DEPTHWIZARD — Single-View Height Estimation & 3D Flythrough
Smart India Hackathon (SIH) Image Upload Demonstration Application
Entry point for Streamlit: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Import main interactive upload app logic
from app.app import *
