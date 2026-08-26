"""
DEPTHWIZARD — Streamlit Runner Script
Executes streamlit application from app/app.py
"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
app_path = BASE_DIR / "app" / "app.py"

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"Launching Streamlit App: {' '.join(cmd)}")
    subprocess.run(cmd)
