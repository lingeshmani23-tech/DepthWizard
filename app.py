"""
DEPTHWIZARD — Optional Interactive Streamlit Interface
Entry point for optional interactive Streamlit UI:
Run via: streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Import interactive Streamlit logic from app/app.py
from app.app import *
