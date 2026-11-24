"""Pupil Labs Neon eye tracker discovery and recording control."""

from __future__ import annotations

import csv
import ipaddress
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from config import EyeTrackerConfig


def now_ms() -> int:
    """Return current Unix epoch timestamp in milliseconds."""
    return time.time_ns() // 1_000_000


def _is_full_ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
        return True
    except ValueError:
        return False


class EyeTrackerManager(QObject):
    """Connects to a Pupil Labs Neon device and controls remote recording."""

    status_changed = pyqtSignal(bool, str)  # connected, message

    def __init__(self, config: EyeTrackerConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.config = config
        self._device = None
        self._lock = threading.Lock()
        self._connected = False
        self._recording_id: Optional[str] = None
        self._csv_file = None
        self._csv_writer = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect_async(self) -> None:
        threading.Thread(
            target=self._connect_worker, daemon=True, name="NeonConnectWorker"
        ).start()

    def _connect_worker(self) -> None:
        self.status_changed.emit(False, f"Searching ({self.config.ip})...")
        try:
            from pupil_labs.realtime_api.simple import Device, discover_one_device
        except Exception as exc:
            self._connected = False
            self.status_changed.emit(False, f"Library error: {exc}")
            return

        device = None
        err_msg = ""

        # 1. If a full IPv4 address is configured, connect directly first.
        if _is_full_ipv4(self.config.ip):
            try:
                device = Device(address=self.config.ip, port=self.config.port)
            except Exception as exc:
                err_msg = str(exc)
                device = None

        # 2. Otherwise (e.g. subnet prefix like '192.168.18') or as fallback, use mDNS discovery.
        if device is None:
            try:
                device = discover_one_device(
                    max_search_duration_seconds=self.config.discover_timeout_sec
                )
            except Exception as exc:
                err_msg = str(exc)
                device = None

        with self._lock:
            self._close_locked()
            if device is not None:
                self._device = device
                self._connected = True
                phone_ip = getattr(device, "phone_ip", self.config.ip)
                phone_name = getattr(device, "phone_name", "Neon")
                self.status_changed.emit(True, f"Connected ({phone_name} @ {phone_ip})")
            else:
                self._connected = False
                detail = f" ({err_msg})" if err_msg else ""
                self.status_changed.emit(False, f"Not found{detail}")

    def start_recording(self, session_dir: Path) -> Optional[str]:
        with self._lock:
            rec_id = ""
            if self._device is not None:
                try:
                    self._recording_id = self._device.recording_start()
                    rec_id = str(self._recording_id or "")
                except Exception as exc:
                    self.status_changed.emit(self._connected, f"Record start failed: {exc}")

            csv_path = session_dir / "eyetracker.csv"
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["event", "datetime", "timestamp_ms", "recording_id"])
            self._csv_writer.writerow([
                "record_start",
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
                now_ms(),
                rec_id,
            ])
            self._csv_file.flush()
            return self._recording_id

    def stop_recording(self) -> None:
        with self._lock:
            rec_id = str(self._recording_id or "")
            if self._device is not None:
                try:
                    self._device.recording_stop_and_save()
                except Exception as exc:
                    self.status_changed.emit(self._connected, f"Record stop failed: {exc}")
                finally:
                    self._recording_id = None

            if self._csv_writer is not None and self._csv_file is not None:
                self._csv_writer.writerow([
                    "record_stop",
                    datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    now_ms(),
                    rec_id,
                ])
                self._csv_file.flush()
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def _close_locked(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        self._connected = False

    def close(self) -> None:
        with self._lock:
            self._close_locked()
