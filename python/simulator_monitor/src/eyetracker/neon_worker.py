"""
Pupil Labs Neon Eye Tracker Worker.
Discovers and automatically connects to Pupil Labs Neon eye tracker via network API.
Manages recording start and stop sessions.
Includes robust fallback if device or network is unavailable.
"""

from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

try:
    from pupil_labs.realtime_api.simple import discover_one_device, Device
    PUPIL_AVAILABLE = True
except ImportError:
    PUPIL_AVAILABLE = False
    Device = None


class NeonTrackerWorker(QThread):
    """
    QThread worker for Pupil Labs Neon eye tracker auto-discovery and recording control.
    """
    # Signal emitted when connection status changes: (status_text, is_connected)
    status_changed = pyqtSignal(str, bool)
    recording_changed = pyqtSignal(bool)

    def __init__(
        self,
        auto_discover: bool = True,
        device_address: str = "",
        device_port: int = 8080,
        search_timeout_sec: float = 3.0,
        parent=None,
    ):
        super().__init__(parent)
        self.auto_discover = auto_discover
        self.device_address = device_address.strip()
        self.device_port = device_port
        self.search_timeout = search_timeout_sec

        self._device: Optional[Device] = None
        self._is_connected = False
        self._is_recording = False
        self._recording_id: Optional[str] = None
        self._running = True
        self._mutex = QMutex()

    def run(self):
        """
        Runs auto-discovery upon thread start.
        """
        if not PUPIL_AVAILABLE:
            print("[EyeTracker] pupil-labs-realtime-api is not installed.")
            self.status_changed.emit("Neon: Driver/Library Missing", False)
            return

        self.status_changed.emit("Neon: Searching for device...", False)

        try:
            device = None
            if self.device_address:
                # Direct IP connection
                print(f"[EyeTracker] Attempting direct connection to {self.device_address}:{self.device_port}")
                device = Device(address=self.device_address, port=self.device_port)
            elif self.auto_discover:
                # mDNS / zeroconf discovery
                print(f"[EyeTracker] Discovering Neon device (timeout: {self.search_timeout}s)...")
                device = discover_one_device(max_search_duration_seconds=self.search_timeout)

            with QMutexLocker(self._mutex):
                if device is not None:
                    self._device = device
                    self._is_connected = True
                    phone_name = getattr(device, "phone_name", "Connected")
                    print(f"[EyeTracker] Successfully connected to Neon ({phone_name}).")
                    self.status_changed.emit(f"Neon: Connected ({phone_name})", True)
                else:
                    self._device = None
                    self._is_connected = False
                    print("[EyeTracker] Neon eye tracker not found on network.")
                    self.status_changed.emit("Neon: Device Not Found (Offline)", False)

        except Exception as e:
            print(f"[EyeTracker] Error during Neon discovery: {e}")
            with QMutexLocker(self._mutex):
                self._device = None
                self._is_connected = False
            self.status_changed.emit("Neon: Connection Error", False)

    def start_recording(self) -> bool:
        """
        Starts eye tracker recording on Neon companion device.
        """
        with QMutexLocker(self._mutex):
            if not self._is_connected or self._device is None:
                print("[EyeTracker] Cannot start recording: Neon device is not connected.")
                return False

            try:
                rec_id = self._device.recording_start()
                self._recording_id = rec_id
                self._is_recording = True
                print(f"[EyeTracker] Neon recording started (ID: {rec_id})")
                self.recording_changed.emit(True)
                return True
            except Exception as e:
                print(f"[EyeTracker] Failed to start Neon recording: {e}")
                return False

    def stop_recording(self) -> bool:
        """
        Stops and saves eye tracker recording on Neon companion device.
        """
        with QMutexLocker(self._mutex):
            if not self._is_recording or self._device is None:
                return False

            try:
                self._device.recording_stop_and_save()
                print(f"[EyeTracker] Neon recording stopped and saved (ID: {self._recording_id})")
                self._is_recording = False
                self._recording_id = None
                self.recording_changed.emit(False)
                return True
            except Exception as e:
                print(f"[EyeTracker] Failed to stop Neon recording: {e}")
                self._is_recording = False
                self.recording_changed.emit(False)
                return False

    def stop(self):
        """
        Stops the worker thread cleanly.
        """
        self._running = False
        if self._is_recording:
            self.stop_recording()
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        self.wait(1000)

    @property
    def is_connected(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._is_connected

    @property
    def is_recording(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._is_recording
