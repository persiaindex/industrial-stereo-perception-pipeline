# Industrial Stereo Perception Pipeline

One-month portfolio project for modern industrial computer vision / 3D perception.

## Month-1 target
Real cameras -> PyTorch detection -> stereo calibration -> disparity/depth ->
3D object position -> tracking -> evaluation -> ONNX-oriented deployment.

## Day 1
- project structure
- environment verification
- CPU/CUDA detection
- camera probing
- initial tests

## Setup
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision
pip install -e ".[dev]"
python scripts/check_environment.py
pytest
python scripts/probe_cameras.py
```

If Windows Application Control blocks a PyTorch DLL again, stop and diagnose the exact
policy/DLL issue rather than disabling security controls blindly.
