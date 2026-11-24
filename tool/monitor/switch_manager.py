"""Advantech USB-4750 handle switch reader and recorder (IDI0, active-low)."""

from __future__ import annotations

import csv
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from config import SwitchConfig

DAQNAVI_PYTHON_PATH = Path("/opt/advantech/examples/Python")
PORT_COUNT = 2


def now_ms() -> int:
    """Return current Unix epoch timestamp in milliseconds."""
    return time.time_ns() // 1_000_000


class SwitchManager(QObject):
    """Monitors Advantech USB-4750 IDI0 and records press/release edges to switch.csv."""

    status_changed = pyqtSignal(bool, str)  # connected, message
    state_changed = pyqtSignal(int, int)  # state (1=pressed, 0=released), timestamp_ms

    def __init__(self, config: SwitchConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._record_lock = threading.Lock()
        self._recording = False
        self._csv_file = None
        self._csv_writer = None
        self._connected = False
        self._controller = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="USB4750SwitchWorker"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.stop_recording()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._connected = False

    def start_recording(self, session_dir: Path) -> None:
        with self._record_lock:
            if self._recording:
                return
            csv_path = session_dir / "switch.csv"
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["timestamp_ms", "state"])
            self._recording = True

    def stop_recording(self) -> None:
        with self._record_lock:
            self._recording = False
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def _open_controller(self):
        if not DAQNAVI_PYTHON_PATH.is_dir():
            raise RuntimeError(
                f"DAQNavi Python API not found at {DAQNAVI_PYTHON_PATH}"
            )
        if str(DAQNAVI_PYTHON_PATH) not in sys.path:
            sys.path.insert(0, str(DAQNAVI_PYTHON_PATH))

        from Automation.BDaq.BDaqApi import BioFailed  # type: ignore[import-not-found]
        from Automation.BDaq.InstantDiCtrl import InstantDiCtrl  # type: ignore[import-not-found]

        controller = InstantDiCtrl(self.config.device)

        def read_channel() -> int:
            result, port_values = controller.readAny(0, PORT_COUNT)
            if BioFailed(result):
                code = getattr(result, "value", result)
                raise RuntimeError(f"DAQNavi read failed (0x{int(code):X})")
            ch = self.config.channel
            raw_bit = (int(port_values[ch // 8]) >> (ch % 8)) & 1
            # Active low: raw 0 => pressed (1), raw 1 => released (0)
            return (1 - raw_bit) if self.config.active_low else raw_bit

        return controller, read_channel

    def _run_loop(self) -> None:
        controller = None
        try:
            controller, read_channel = self._open_controller()
            self._controller = controller
            previous = read_channel()
            self._connected = True
            self.status_changed.emit(True, f"Connected ({self.config.device})")
            self.state_changed.emit(previous, now_ms())

            poll_sec = self.config.poll_interval_ms / 1000.0
            while not self._stop_event.is_set():
                current = read_channel()
                if current != previous:
                    ts = now_ms()
                    previous = current
                    with self._record_lock:
                        if self._recording and self._csv_writer is not None:
                            self._csv_writer.writerow([ts, current])
                            self._csv_file.flush()
                    self.state_changed.emit(current, ts)
                time.sleep(poll_sec)
        except Exception as exc:
            self._connected = False
            self.status_changed.emit(False, f"Error: {exc}")
        finally:
            self.stop_recording()
            if controller is not None:
                try:
                    controller.dispose()
                except Exception:
                    pass
                self._controller = None
            self._connected = False
