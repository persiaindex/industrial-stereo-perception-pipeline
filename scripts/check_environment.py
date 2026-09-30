import platform
import sys
import cv2
import numpy as np
import torch
from stereo_perception.common.device import get_device_info

def main():
    info = get_device_info("auto")
    print("=== Environment ===")
    print("Python:", sys.version.split()[0])
    print("OS:", platform.platform())
    print("NumPy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("PyTorch:", torch.__version__)
    print("Device:", info.device)
    print("CUDA available:", info.cuda_available)
    print("CUDA devices:", info.cuda_device_count)
    print("CUDA device name:", info.cuda_device_name)

    x = torch.tensor([1.0, 2.0, 3.0], device=info.device)
    print("Tensor smoke result:", float(x.square().sum().cpu()))

if __name__ == "__main__":
    main()
