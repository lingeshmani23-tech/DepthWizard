"""
GPU Verification & System Diagnostics Module
"""

import sys
from typing import Dict, Any

def get_device() -> str:
    """
    Check PyTorch CUDA availability and return active compute device string ('cuda' or 'cpu').
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_gpu_info() -> Dict[str, Any]:
    """
    Query system GPU, CUDA version, PyTorch version, and VRAM statistics.
    Returns structured dictionary with diagnostic status.
    """
    info = {
        "pytorch_version": "N/A",
        "cuda_available": False,
        "cuda_version": "None",
        "gpu_name": "None",
        "total_memory_gb": 0.0,
        "allocated_memory_gb": 0.0,
        "device_str": "cpu"
    }

    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["device_str"] = "cuda"
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            
            # Memory stats
            props = torch.cuda.get_device_properties(0)
            info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["allocated_memory_gb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
    except Exception as e:
        info["error"] = str(e)

    return info
