"""Device detection and environment-adaptive configuration."""

import torch


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Return optimal dtype for given device."""
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def get_amp_enabled(device: torch.device) -> bool:
    """Whether automatic mixed precision is available."""
    return device.type == "cuda"


def device_info() -> dict:
    """Return a summary of the current compute environment."""
    device = get_device()
    info = {
        "device": str(device),
        "dtype": str(get_dtype(device)),
        "amp_enabled": get_amp_enabled(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
        info["gpu_count"] = torch.cuda.device_count()
    return info
