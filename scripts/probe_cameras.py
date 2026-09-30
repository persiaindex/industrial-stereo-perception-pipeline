import os
import cv2
from stereo_perception.cameras.probe import probe_camera

def main():
    backend = cv2.CAP_DSHOW if os.name == "nt" else None
    for index in range(5):
        result = probe_camera(index, backend)
        print(
            f"camera={result.index} opened={result.opened} "
            f"frame_ok={result.frame_ok} shape={result.shape}"
        )

if __name__ == "__main__":
    main()
