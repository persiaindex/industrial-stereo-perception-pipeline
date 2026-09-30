from dataclasses import dataclass
import cv2

@dataclass(frozen=True)
class CameraProbeResult:
    index: int
    opened: bool
    frame_ok: bool
    shape: tuple[int, ...] | None

def probe_camera(index: int, backend: int | None = None) -> CameraProbeResult:
    cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return CameraProbeResult(index, False, False, None)

        ok, frame = False, None
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                break

        return CameraProbeResult(
            index=index,
            opened=True,
            frame_ok=bool(ok and frame is not None),
            shape=tuple(frame.shape) if ok and frame is not None else None,
        )
    finally:
        cap.release()
