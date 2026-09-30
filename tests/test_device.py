import pytest
from stereo_perception.common.device import get_device_info

def test_auto_device():
    assert get_device_info("auto").device in {"cpu", "cuda"}

def test_force_cpu():
    assert get_device_info("cpu").device == "cpu"

def test_invalid_device():
    with pytest.raises(ValueError):
        get_device_info("gpu")
