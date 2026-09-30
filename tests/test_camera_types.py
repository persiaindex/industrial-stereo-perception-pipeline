from stereo_perception.cameras.probe import CameraProbeResult

def test_camera_probe_result():
    r = CameraProbeResult(0, True, True, (480, 640, 3))
    assert r.index == 0
    assert r.shape == (480, 640, 3)
