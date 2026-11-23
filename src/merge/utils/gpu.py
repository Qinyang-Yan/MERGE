"""GPU discovery utilities."""

from typing import Dict


def get_device(prefer_gpu: bool = True) -> str:
    """Return ``cuda`` if available and requested, otherwise ``cpu``."""
    import torch

    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


def describe_gpu() -> Dict[str, object]:
    """Return a serializable summary of CUDA availability and devices."""
    import torch

    info: Dict[str, object] = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        info["device_count"] = int(torch.cuda.device_count())
        info["devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
            }
            for i in range(torch.cuda.device_count())
        ]
    else:
        info["device_count"] = 0
        info["devices"] = []
    return info
