"""
DEPTHWIZARD — Depth Estimation Engine
Loads Depth Anything V2 Metric Outdoor Small model, performs CUDA inference,
saves numerical metric depth (output/depth.npy) and colorized visualization (output/depth.png).
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def load_depth_model(model_name=config.MODEL_NAME, device=config.DEVICE):
    """
    Loads Depth Anything V2 model and image processor.
    """
    print(f"Loading model: {model_name}")
    print(f"Target Device: {device}")
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    return processor, model

def run_depth_estimation(image_path=None, output_dir=None, device=None):
    """
    Runs metric depth estimation on an input image.
    Saves depth.npy and depth.png.
    """
    if image_path is None:
        image_path = config.INPUT_DIR / "test.jpg"
    else:
        image_path = Path(image_path)
        
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = config.DEVICE
        
    print(f"Device: {device}")
    print(f"Input: {image_path}")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found at {image_path}")
        
    raw_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_image.size
    print(f"Image size: {orig_w}x{orig_h}")
    
    processor, model = load_depth_model(config.MODEL_NAME, device=device)
    
    # Process inputs
    inputs = processor(images=raw_image, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        
        # Resize predicted depth map to match original image dimensions
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    depth_np = prediction.cpu().numpy()
    
    # Ensure non-negative metric depth
    depth_np = np.maximum(depth_np, 0.0)
    
    depth_h, depth_w = depth_np.shape
    min_d = float(np.min(depth_np))
    max_d = float(np.max(depth_np))
    mean_d = float(np.mean(depth_np))
    
    print(f"Depth size: {depth_w}x{depth_h}")
    print(f"Minimum depth: {min_d:.4f} m")
    print(f"Maximum depth: {max_d:.4f} m")
    print(f"Mean depth: {mean_d:.4f} m")
    
    # Save depth.npy
    npy_path = output_dir / "depth.npy"
    np.save(npy_path, depth_np)
    print(f"Saved {npy_path.name}")
    
    # Generate & Save colorized depth map (depth.png) using inferno colormap
    norm_depth = (depth_np - min_d) / (max_d - min_d + 1e-8)
    depth_colored = (plt.cm.inferno(norm_depth)[:, :, :3] * 255).astype(np.uint8)
    
    png_path = output_dir / "depth.png"
    Image.fromarray(depth_colored).save(png_path)
    print(f"Saved {png_path.name}")
    
    return {
        "depth_np": depth_np,
        "npy_path": npy_path,
        "png_path": png_path,
        "min_depth": min_d,
        "max_depth": max_d,
        "mean_depth": mean_d,
        "image_size": (orig_w, orig_h)
    }

if __name__ == "__main__":
    run_depth_estimation()
