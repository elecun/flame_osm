"""
DAQ Worker for Advantech USB-4750-CE Digital Input.
Provides real-time state polling and triggers events when channel 0 is activated.
Includes automatic fallback to simulation mode when hardware or driver is absent.
"""

import time
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.time_utils import get_formatted_timestamp_ms

# Try importing Advantech DAQNavi library
ADV_AVAILABLE = False
try:
    from Automation.BDaq import InstantDiCtrl, DeviceInformation, ErrorCode
    ADV_AVAILABLE = True
except ImportError:
    ADV_AVAILABLE = False


class DAQWorker(QThread):
    """
    QThread worker to poll digital input channel on Advantech USB-4750.
    """
    # Signal emitted on state change or periodic refresh: True -> 'pressed', False -> 'idle'
    state_changed = pyqtSignal(bool)
    # Signal emitted when rising edge / press event occurs with timestamp (ms)
    triggered = pyqtSignal(str)

    def __init__(
        self,
        device_description: str = "USB-4750,BID#0",
        port: int = 0,
        channel: int = 0,
        poll_interval_ms: int = 10,
        mock_mode: str = "auto",
        parent=None,
    ):
        super().__init__(parent)
        self.device_description = device_description
        self.port = port
        self.channel = channel
        self.poll_interval = max(poll_interval_ms, 5) / 1000.0  # seconds
        self.mock_mode = mock_mode.lower()

        self._running = False
        self._is_mock = False
        self._current_state = False  # False = idle, True = pressed
        self._di_ctrl = None

        # Software mock pulse support
        self._mock_pressed_until = 0.0

    def init_device(self) -> bool:
        """
        Attempts to initialize the Advantech device.
        Falls back to mock mode if mock_mode='true' or if hardware initialization fails.
        """
        if self.mock_mode == "true" or not ADV_AVAILABLE:
            self._is_mock = True
            print(f"[DAQ] Operating in MOCK mode (Advantech SDK available: {ADV_AVAILABLE})")
            return True

        try:
            self._di_ctrl = InstantDiCtrl()
            self._di_ctrl.SelectedDevice = DeviceInformation(self.device_description)
            if not self._di_ctrl.Initialized:
                print(f"[DAQ] Device '{self.device_description}' not found. Falling back to MOCK mode.")
                self._is_mock = True
                self._di_ctrl = None
                return True
            print(f"[DAQ] Successfully connected to Advantech device '{self.device_description}'.")
            self._is_mock = False
            return True
        except Exception as e:
            print(f"[DAQ] Failed to initialize hardware ({e}). Falling back to MOCK mode.")
            self._is_mock = True
            self._di_ctrl = None
            return True

    def trigger_mock_press(self, duration_sec: float = 0.2):
        """
        Simulates a switch press for the specified duration (used in mock/testing mode).
        """
        self._mock_pressed_until = time.time() + duration_sec

    def read_hardware_bit(self) -> bool:
        """
        Reads channel 0 bit from Advantech USB-4750.
        """
        if self._is_mock or self._di_ctrl is None:
            return time.time() < self._mock_pressed_until

        try:
            # Read single bit or byte from port
            # In DAQNavi InstantDiCtrl.ReadBit(port, channel) or Read(port)
            err, bit_data = self._di_ctrl.ReadBit(self.port, self.channel)
            if err == ErrorCode.Success:
                # 1 = High / Pressed (or depending on wiring; 1 indicates active input)
                return bool(bit_data)
            else:
                # Try reading whole byte if ReadBit is unsupported
                err, port_data = self._di_ctrl.Read(self.port)
                if err == ErrorCode.Success and len(port_data) > 0:
                    return bool((port_data[0] >> self.channel) & 0x01)
        except Exception as e:
            # Suppress excessive logging during polling
            pass
        return False

    def run(self):
        self.init_device()
        self._running = True
        last_state = False

        # Initial state emission
        self.state_changed.emit(False)

        while self._running:
            new_state = self.read_hardware_bit()

            if new_state != last_state:
                # State transition occurred
                self.state_changed.emit(new_state)
                if new_state:  # Rising edge (idle -> pressed)
                    ts = get_formatted_timestamp_ms()
                    self.triggered.emit(ts)
                last_state = new_state
                self._current_state = new_state

            time.sleep(self.poll_interval)

        # Cleanup hardware on thread exit
        if self._di_ctrl is not None:
            try:
                self._di_ctrl.Cleanup()
            except Exception:
                pass
            self._di_ctrl = None

    def stop(self):
        self._running = False
        self.wait()

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def current_state(self) -> bool:
        return self._current_state
