"""USB camera capture and recording manager (up to 4 cameras)."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from config import CameraConfig


def now_ms() -> int:
    """Return current Unix epoch timestamp in milliseconds."""
    return time.time_ns() // 1_000_000


class CameraWorker(threading.Thread):
    """Captures frames from a single USB camera and optionally records them."""

    def __init__(
        self,
        slot_index: int,
        device_index: int,
        config: CameraConfig,
        on_frame,
    ) -> None:
        super().__init__(daemon=True, name=f"CameraWorker-{slot_index}")
        self.slot_index = slot_index
        self.device_index = device_index
        self.config = config
        self._on_frame = on_frame

        self._stop_event = threading.Event()
        self._record_lock = threading.Lock()
        self._recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._csv_file = None
        self._csv_writer = None
        self._frame_count = 0

        self._cap: Optional[cv2.VideoCapture] = None
        self.connected = False
        self.actual_width = config.width
        self.actual_height = config.height
        self.actual_fps = float(config.fps)

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False

        if len(self.config.fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False

        self.actual_height, self.actual_width = frame.shape[:2]
        # Use configured FPS (e.g. 30) because V4L2 CAP_PROP_FPS may report max 60fps
        self.actual_fps = float(self.config.fps)
        self._cap = cap
        self.connected = True
        return True

    def start_recording(self, session_dir: Path) -> None:
        with self._record_lock:
            if self._recording or not self.connected:
                return
            avi_path = session_dir / f"camera_{self.slot_index}.avi"
            csv_path = session_dir / f"camera_{self.slot_index}.csv"

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._writer = cv2.VideoWriter(
                str(avi_path),
                fourcc,
                self.actual_fps,
                (self.actual_width, self.actual_height),
            )
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["frame_index", "timestamp_ms"])
            self._frame_count = 0
            self._recording = True

    def stop_recording(self) -> None:
        with self._record_lock:
            self._recording = False
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def stop(self) -> None:
        self._stop_event.set()
        self.stop_recording()

    def run(self) -> None:
        if self._cap is None:
            return
        try:
            while not self._stop_event.is_set():
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                ts = now_ms()
                with self._record_lock:
                    if self._recording and self._writer is not None:
                        self._writer.write(frame)
                        self._csv_writer.writerow([self._frame_count, ts])
                        self._frame_count += 1

                self._on_frame(self.slot_index, frame)
        finally:
            self.stop_recording()
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self.connected = False


class CameraManager(QObject):
    """Manages up to 4 USB cameras for live monitoring and synchronized recording."""

    frame_ready = pyqtSignal(int, object)  # slot_index, BGR numpy.ndarray
    status_changed = pyqtSignal(bool, int, str)  # connected, count, message

    def __init__(self, config: CameraConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.config = config
        self._workers: list[CameraWorker] = []

    @property
    def connected_count(self) -> int:
        return sum(1 for w in self._workers if w.connected)

    @property
    def is_connected(self) -> bool:
        return self.connected_count > 0

    def _emit_frame(self, slot_index: int, frame: np.ndarray) -> None:
        self.frame_ready.emit(slot_index, frame)

    def start_cameras(self) -> None:
        self.stop_cameras()
        threading.Thread(
            target=self._discover_and_start_cameras,
            daemon=True,
            name="CameraDiscoveryWorker",
        ).start()

    def _discover_and_start_cameras(self) -> None:
        if self.config.device_indices is not None:
            candidate_indices = self.config.device_indices
        else:
            candidate_indices = sorted(
                int(p.name.replace("video", ""))
                for p in Path("/dev").glob("video*")
                if p.name.replace("video", "").isdigit()
            )

        slot = 0
        for dev_idx in candidate_indices:
            if slot >= self.config.max_cameras:
                break
            worker = CameraWorker(slot, dev_idx, self.config, self._emit_frame)
            if worker.open():
                worker.start()
                self._workers.append(worker)
                slot += 1

        count = len(self._workers)
        if count > 0:
            devs = ", ".join(f"/dev/video{w.device_index}" for w in self._workers)
            self.status_changed.emit(True, count, f"Connected ({count}): {devs}")
        else:
            self.status_changed.emit(False, 0, "No USB cameras detected")

    def start_recording(self, session_dir: Path) -> None:
        for worker in self._workers:
            worker.start_recording(session_dir)

    def stop_recording(self) -> None:
        # Stop recording only; keep camera streams open for live monitoring.
        for worker in self._workers:
            worker.stop_recording()

    def stop_cameras(self) -> None:
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            worker.join(timeout=2.0)
        self._workers.clear()
