"""
Hardware & Environment Utilities
Detects compute device (CUDA/MPS/CPU), clamps OpenBLAS threads, and reports hardware specs.
"""
import os
import platform
import sys
from typing import Dict, Any
import torch


def clamp_blas_threads(num_threads: int = 4):
    """Clamps OpenBLAS/OMP/MKL threads on CPU to prevent Windows memory exhaustion."""
    t_str = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = t_str
    os.environ["OMP_NUM_THREADS"] = t_str
    os.environ["MKL_NUM_THREADS"] = t_str
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def detect_device(requested: str = "auto") -> str:
    """Detects available compute device with robust fallbacks."""
    if requested != "auto":
        return requested.lower()

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_hardware_info() -> Dict[str, Any]:
    """Inspects host system hardware specs."""
    info = {
        "os": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    return info

