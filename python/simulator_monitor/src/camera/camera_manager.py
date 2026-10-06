"""
Manager to orchestrate up to 4 camera capture workers and synchronized recording.
"""

from typing import List, Dict
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from src.camera.camera_thread import CameraWorker
from src.config import CameraConfig


class CameraManager(QObject):
    """
    Coordinates multiple CameraWorker threads.
    """
    frame_received = pyqtSignal(int, np.ndarray)

    def __init__(self, config: CameraConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.workers: Dict[int, CameraWorker] = {}

    def start_cameras(self):
        """
        Initializes and starts capture threads for configured camera IDs (up to 4).
        """
        self.stop_cameras()

        camera_ids = self.config.camera_ids[:4]
        for cam_id in camera_ids:
            worker = CameraWorker(
                camera_id=cam_id,
                fps=self.config.fps,
                width=self.config.width,
                height=self.config.height,
                codec=self.config.codec,
                mock_if_missing=self.config.mock_if_missing,
            )
            worker.frame_ready.connect(self.frame_received.emit)
            self.workers[cam_id] = worker
            worker.start()

    def start_recording(self, output_dir: str):
        """
        Starts AVI recording across all camera workers.
        """
        for worker in self.workers.values():
            worker.start_recording(output_dir)

    def stop_recording(self):
        """
        Stops recording on all cameras without interrupting real-time preview.
        """
        for worker in self.workers.values():
            worker.stop_recording()

    def stop_cameras(self):
        """
        Safely stops all camera capture threads.
        """
        for worker in self.workers.values():
            worker.stop()
        self.workers.clear()

    @property
    def is_recording(self) -> bool:
        return any(w.is_recording for w in self.workers.values())
