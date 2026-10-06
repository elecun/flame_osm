"""
Camera worker thread using OpenCV.
Captures frames from USB cameras or generates synthetic test frames if cameras are missing.
Draws camera ID in the top-left corner and manages AVI video recording.
"""

import time
import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker


class CameraWorker(QThread):
    """
    Worker thread that continuously captures video frames from a camera device.
    """
    # Emits (camera_id, frame_bgr)
    frame_ready = pyqtSignal(int, np.ndarray)

    def __init__(
        self,
        camera_id: int,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        codec: str = "XVID",
        mock_if_missing: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_id = camera_id
        self.target_fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.mock_if_missing = mock_if_missing

        self._running = False
        self._is_mock = False
        self._cap = None
        self._writer = None
        self._recording = False
        self._record_path = None
        self._mutex = QMutex()
        self._frame_count = 0

    def init_capture(self) -> bool:
        """
        Attempts to open the physical camera via cv2.VideoCapture.
        Falls back to mock mode if unavailable.
        """
        try:
            # Try default backend or platform backend
            cap = cv2.VideoCapture(self.camera_id)
            if cap.isOpened():
                # Attempt to set desired resolution and fps
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._cap = cap
                self._is_mock = False
                print(f"[Camera {self.camera_id}] Physical camera opened successfully.")
                return True
        except Exception as e:
            print(f"[Camera {self.camera_id}] Could not open physical camera: {e}")

        if self.mock_if_missing:
            self._is_mock = True
            print(f"[Camera {self.camera_id}] Missing physical camera. Using synthetic mock feed.")
            return True
        return False

    def start_recording(self, output_dir: str):
        """
        Starts recording incoming frames to cam_<id>.avi.
        """
        with QMutexLocker(self._mutex):
            os.makedirs(output_dir, exist_ok=True)
            self._record_path = os.path.join(output_dir, f"cam_{self.camera_id}.avi")

            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                self._record_path,
                fourcc,
                self.target_fps,
                (self.width, self.height),
            )
            if not self._writer.isOpened():
                # Fallback to MJPG codec if XVID is unavailable on current platform
                alt_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._writer = cv2.VideoWriter(
                    self._record_path,
                    alt_fourcc,
                    self.target_fps,
                    (self.width, self.height),
                )
            self._recording = True
            print(f"[Camera {self.camera_id}] Started recording to {self._record_path}")

    def stop_recording(self):
        """
        Stops recording and safely finalizes the AVI file.
        """
        with QMutexLocker(self._mutex):
            if self._recording:
                self._recording = False
                if self._writer is not None:
                    try:
                        self._writer.release()
                    except Exception as e:
                        print(f"[Camera {self.camera_id}] Error releasing writer: {e}")
                    finally:
                        self._writer = None
                print(f"[Camera {self.camera_id}] Stopped recording safely.")

    def run(self):
        self._running = True
        if not self.init_capture():
            print(f"[Camera {self.camera_id}] Initialization failed.")
            return

        frame_duration = 1.0 / max(self.target_fps, 1)

        while self._running:
            loop_start = time.time()
            raw_frame = None

            if not self._is_mock and self._cap is not None:
                ret, captured = self._cap.read()
                if ret and captured is not None:
                    if captured.shape[1] != self.width or captured.shape[0] != self.height:
                        raw_frame = cv2.resize(captured, (self.width, self.height))
                    else:
                        raw_frame = captured
                else:
                    raw_frame = self._generate_no_camera_frame()
            else:
                raw_frame = self._generate_no_camera_frame()

            # Record RAW original frame directly to video file (no overlays on recorded data)
            with QMutexLocker(self._mutex):
                if self._recording and self._writer is not None:
                    try:
                        self._writer.write(raw_frame)
                    except Exception as e:
                        print(f"[Camera {self.camera_id}] Write error: {e}")

            # For GUI preview: overlay Camera ID on copy
            preview_frame = raw_frame.copy()
            self._overlay_camera_id(preview_frame)

            # Emit frame to GUI for real-time display
            self.frame_ready.emit(self.camera_id, preview_frame)
            self._frame_count += 1

            # Regulate frame rate
            elapsed = time.time() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup on exit
        self.stop_recording()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _overlay_camera_id(self, frame: np.ndarray):
        """
        Draws Camera ID badge in top-left corner.
        """
        text = f"CAM {self.camera_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        # Draw dark background box for high visibility
        box_x1, box_y1 = 8, 8
        box_x2, box_y2 = box_x1 + tw + 14, box_y1 + th + 14
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 220, 0), 1)

        # Draw Camera ID text
        text_pos = (box_x1 + 7, box_y1 + th + 6)
        cv2.putText(frame, text, text_pos, font, scale, (0, 255, 128), thickness, cv2.LINE_AA)

    def _generate_no_camera_frame(self) -> np.ndarray:
        """
        Generates a black canvas with 'no camera' centered for disconnected feeds.
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        text = "no camera"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = (self.width - tw) // 2
        ty = (self.height + th) // 2
        cv2.putText(frame, text, (tx, ty), font, scale, (120, 120, 130), thickness, cv2.LINE_AA)
        return frame

    def stop(self):
        self._running = False
        self.wait()

    @property
    def is_recording(self) -> bool:
        return self._recording
