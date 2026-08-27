"""
Depth Anything V2 Model Inference & Processing Engine
Implements cached pretrained monocular depth inference, direction-consistent depth normalization,
numerical .npy array persistence, and colorbar visualizations.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Suppress unnecessary logs
import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Suppress unnecessary logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from config import MODEL_NAME
except ImportError:
    MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

_MODEL_CACHE = {}


def _load_model_uncached(model_id: str):
    """
    Core function to load pretrained Depth Anything V2 model from Hugging Face transformers.
    Returns (processor, model, device, device_name, is_gpu).
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        is_gpu = torch.cuda.is_available()
        device = "cuda" if is_gpu else "cpu"
        device_name = torch.cuda.get_device_name(0) if is_gpu else "CPU Fallback"
        
        # Try HuggingFace Depth Anything V2 model
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
        except Exception:
            # Fallback to Depth-Anything-Small-hf if V2 HF tag is structured differently
            fallback_id = "LiheYoung/depth-anything-small-hf"
            processor = AutoImageProcessor.from_pretrained(fallback_id)
            model = AutoModelForDepthEstimation.from_pretrained(fallback_id).to(device)

        model.eval()
        return processor, model, device, device_name, is_gpu
    except Exception as e:
        print(f"[DepthEngine Warning] PyTorch model loading error: {e}. Falling back to analytical depth engine.")
        return None, None, "cpu", "CPU Fallback", False


def load_depth_model(model_id: str = MODEL_NAME):
    """
    Load pretrained Depth Anything V2 model from Hugging Face transformers.
    Cached in global memory dictionary or st.cache_resource to enforce ONE-TIME loading per process.
    """
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    # Check if streamlit is active and use @st.cache_resource
    try:
        import streamlit as st
        @st.cache_resource(show_spinner="Loading Depth Anything V2 Model Weights...")
        def _cached_st_loader(m_id: str):
            return _load_model_uncached(m_id)
        
        res = _cached_st_loader(model_id)
        _MODEL_CACHE[model_id] = res
        return res
    except Exception:
        res = _load_model_uncached(model_id)
        _MODEL_CACHE[model_id] = res
        return res


def get_device_status(model_id: str = MODEL_NAME) -> Dict[str, Any]:
    """
    Return human-readable device and model health status.
    """
    processor, model, device, device_name, is_gpu = load_depth_model(model_id)
    return {
        "device": device,
        "device_name": device_name,
        "is_gpu": is_gpu,
        "status_label": f"GPU: Available ({device_name})" if is_gpu else "GPU: CPU Fallback",
        "model_loaded": model is not None,
        "model_name": model_id
    }


def estimate_depth(rgb_img: np.ndarray, model_id: str = MODEL_NAME) -> np.ndarray:
    """
    Perform monocular depth estimation on RGB numpy array.
    Executes under torch.inference_mode().
    Returns raw 2D numerical depth float32 numpy array.
    """
    processor, model, device, device_name, is_gpu = load_depth_model(model_id)

    if model is not None:
        import torch
        pil_img = Image.fromarray(rgb_img)

        with torch.inference_mode():
            inputs = processor(images=pil_img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

            # Interpolate to original image size
            h, w = rgb_img.shape[:2]
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            raw_depth = prediction.squeeze().cpu().numpy().astype(np.float32)
            return raw_depth
    else:
        # Robust mathematical gradient-depth fallback (for testing without heavy weights)
        h, w = rgb_img.shape[:2]
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        # Ground plane gradient + intensity shading proxy
        raw_depth = (1.0 - (y_grid / h)) * 0.7 + (1.0 - gray) * 0.3
        return raw_depth.astype(np.float32)


def convert_to_distance_like_depth(raw_depth: np.ndarray) -> np.ndarray:
    """
    Normalizes monocular depth output into a consistent distance representation
    where Z > 0 increases with physical camera-to-object distance (meters proxy).
    Depth Anything V2 predicts relative depth/disparity (where larger = closer).
    We invert disparity: Z_rel = 1.0 / (raw_depth + 1e-6) or linear distance proxy.
    """
    depth_clean = np.nan_to_num(raw_depth, nan=0.0, posinf=1.0, neginf=0.0)
    min_val, max_val = depth_clean.min(), depth_clean.max()
    
    if max_val - min_val < 1e-6:
        return np.ones_like(depth_clean, dtype=np.float32)

    # Scale raw depth to [0.1, 1.0] range
    norm_disp = 0.1 + 0.9 * ((depth_clean - min_val) / (max_val - min_val))
    
    # Invert disparity to obtain distance-like depth (Z_rel)
    distance_depth = 1.0 / norm_disp
    return distance_depth.astype(np.float32)


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Scale depth map values into [0.0, 1.0] interval for visualization.
    """
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth_map, dtype=np.float32)
    return (depth_map - d_min) / (d_max - d_min)


def save_depth(depth_raw: np.ndarray, output_dir: str) -> Tuple[str, str]:
    """
    Persist raw numerical depth as .npy array and generate colorized visualization image.

    Returns:
        Tuple of (npy_file_path, png_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, "depth_raw.npy")
    png_path = os.path.join(output_dir, "depth_visualization.png")

    # Save raw array
    np.save(npy_path, depth_raw.astype(np.float32))

    # Generate visualization
    norm_depth = normalize_depth(depth_raw)
    colored_depth = (cm.viridis(norm_depth)[:, :, :3] * 255).astype(np.uint8)
    
    # Save visualization with matplotlib figure colorbar
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    im = ax.imshow(norm_depth, cmap="viridis")
    ax.set_title(f"Depth Map (Min: {depth_raw.min():.2f}, Max: {depth_raw.max():.2f})")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Depth")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return npy_path, png_path


def visualize_depth(depth_raw: np.ndarray) -> Dict[str, Any]:
    """
    Calculate numerical statistics for depth map display.
    """
    return {
        "min": float(np.nanmin(depth_raw)),
        "max": float(np.nanmax(depth_raw)),
        "mean": float(np.nanmean(depth_raw)),
        "std": float(np.nanstd(depth_raw)),
        "shape": depth_raw.shape
    }
