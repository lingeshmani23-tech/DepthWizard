"""
DEPTHWIZARD — Reusable Frontend API Client
Manages HTTP communication between the Public Frontend and the AI Computation Backend API.
"""

import os
import io
import json
from typing import Dict, Any, Tuple, Optional
import requests

def get_backend_url() -> str:
    """
    Retrieve backend API base URL from environment variable BACKEND_API_URL.
    Defaults to http://localhost:8000 for local development.
    """
    url = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")
    return url


def check_backend_health(backend_url: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Query backend health status endpoint GET /api/health.
    """
    if backend_url is None:
        backend_url = get_backend_url()

    try:
        resp = requests.get(f"{backend_url}/api/health", timeout=4)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"Backend returned status code {resp.status_code}"}
    except Exception as e:
        return False, {"error": f"Could not connect to backend API at {backend_url}: {str(e)}"}


def analyze_image_remote(
    image_bytes: bytes,
    filename: str = "uploaded_image.jpg",
    reference_object: str = "Person",
    reference_height_m: float = 1.70,
    reference_x: int = 0,
    reference_top_y: int = 0,
    reference_bot_y: int = 0,
    target_x: int = 0,
    target_top_y: int = 0,
    target_bot_y: int = 0,
    known_target_height_m: float = 0.0,
    fov_deg: float = 60.0,
    voxel_size: float = 0.03,
    backend_url: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Send RGB image and parameters to Backend API POST /api/analyze.
    """
    if backend_url is None:
        backend_url = get_backend_url()

    endpoint = f"{backend_url}/api/analyze"

    files = {
        "image": (filename, image_bytes, "image/jpeg")
    }

    data = {
        "reference_object": reference_object,
        "reference_height_m": str(reference_height_m),
        "reference_x": str(reference_x),
        "reference_top_y": str(reference_top_y),
        "reference_bot_y": str(reference_bot_y),
        "target_x": str(target_x),
        "target_top_y": str(target_top_y),
        "target_bot_y": str(target_bot_y),
        "known_target_height_m": str(known_target_height_m),
        "fov_deg": str(fov_deg),
        "voxel_size": str(voxel_size)
    }

    try:
        resp = requests.post(endpoint, files=files, data=data, timeout=120)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"Backend API error ({resp.status_code}): {resp.text}"}
    except Exception as e:
        return False, {"error": f"Failed to communicate with AI Backend: {str(e)}"}
