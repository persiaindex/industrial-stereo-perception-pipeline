from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class DeviceInfo:
    device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None

def get_device_info(preferred: str = "auto") -> DeviceInfo:
    preferred = preferred.lower().strip()
    if preferred not in {"auto", "cuda", "cpu"}:
        raise ValueError("preferred must be one of: auto, cuda, cpu")

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else None

    if preferred == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA.")

    device = "cpu" if preferred == "cpu" else (
        "cuda" if preferred == "cuda" or (preferred == "auto" and cuda_available) else "cpu"
    )

    return DeviceInfo(device, cuda_available, cuda_count, cuda_name)
