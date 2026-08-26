"""
Image Loading and Preprocessing Module
Handles multi-format RGB image loading, resolution normalization, and depth array scaling.
"""

import os
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageOps
import cv2


def load_image(image_input: Union[str, bytes, Image.Image]) -> Tuple[Image.Image, np.ndarray]:
    """
    Load an image from file path, raw bytes, or PIL Image object.
    Enforces RGB conversion and removes EXIF orientation rotations.

    Returns:
        Tuple of (PIL.Image in RGB, numpy uint8 array in RGB)
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, (str, os.PathLike)):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        import io
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, np.ndarray):
        np_img = image_input.copy()
        pil_img = Image.fromarray(np_img)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Auto-rotate based on EXIF tag if present
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")

    np_img = np.array(pil_img)
    return pil_img, np_img


def validate_image(image_input: Union[str, bytes, Image.Image]) -> Tuple[bool, str, Optional[Image.Image], Optional[np.ndarray]]:
    """
    Validates uploaded outdoor image.
    Verifies decoding, RGB 3-channel layout, resolution, and absence of corruption.

    Returns:
        Tuple of (is_valid: bool, status_message: str, pil_img, np_img)
    """
    try:
        pil_img, np_img = load_image(image_input)
        h, w, c = np_img.shape

        if c != 3:
            return False, f"Image must be 3-channel RGB. Found {c} channels.", None, None

        if w < 64 or h < 64:
            return False, f"Image resolution ({w}x{h}) is too low for depth estimation. Minimum is 64x64.", None, None

        return True, "✓ Image successfully loaded", pil_img, np_img
    except Exception as e:
        return False, f"Image validation error: {str(e)}", None, None


def preprocess_image(np_img: np.ndarray, max_dim: int = 1024) -> Tuple[np.ndarray, float]:
    """
    Downscale RGB image for fast inference if max dimension exceeds max_dim.

    Returns:
        Tuple of (resized_np_img, scale_factor)
    """
    h, w = np_img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_dim:
        return np_img, 1.0

    scale = max_dim / float(long_side)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def resize_depth_to_image(depth_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize relative depth map back to exact original image dimensions (height, width).
    GuaranteES: RGB width == depth width and RGB height == depth height.
    """
    target_h, target_w = target_shape[:2]
    if depth_map.shape[:2] == (target_h, target_w):
        return depth_map.astype(np.float32)

    resized_depth = cv2.resize(depth_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return resized_depth
