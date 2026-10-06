"""
Operator Monitoring & Control GUI window.
Manages 4-camera video streams, Advantech USB-4750 DAQ switch monitoring,
Pupil Labs Neon eye tracker integration, sound feedback, scenario selection,
and experiment session recording.
"""

import os
import glob
import cv2
import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QMessageBox
from PyQt6.QtCore import pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut

from src.config import AppConfig
from src.camera.camera_manager import CameraManager
from src.daq.daq_worker import DAQWorker
from src.daq.daq_logger import DAQLogger
from src.eyetracker.neon_worker import NeonTrackerWorker
from src.sound.sound_player import SoundPlayer
from src.gui.subject_window import SubjectWindow
from src.utils.time_utils import get_directory_timestamp, get_formatted_timestamp_ms


class OperatorWindow(QMainWindow):
    """
    Main operator control interface.
    """

    def __init__(self, config: AppConfig, subject_window: SubjectWindow, parent=None):
        super().__init__(parent)
        self.config = config
        self.subject_window = subject_window

        # Load Qt Designer UI file
        ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ui", "operator_window.ui")
        uic.loadUi(ui_path, self)

        # Mapping of camera labels
        self.cam_labels = {
            0: getattr(self, "label_cam_0", None),
            1: getattr(self, "label_cam_1", None),
            2: getattr(self, "label_cam_2", None),
            3: getattr(self, "label_cam_3", None),
        }

        # Initialize Camera Manager
        self.camera_manager = CameraManager(self.config.camera, self)
        self.camera_manager.frame_received.connect(self.on_frame_received)

        # Initialize DAQ Worker & Logger
        self.daq_worker = DAQWorker(
            device_description=self.config.daq.device_description,
            port=self.config.daq.port,
            channel=self.config.daq.channel,
            poll_interval_ms=self.config.daq.poll_interval_ms,
            mock_mode=self.config.daq.mock_mode,
            parent=self,
        )
        self.daq_worker.state_changed.connect(self.on_daq_state_changed)
        self.daq_worker.triggered.connect(self.on_daq_triggered)

        self.daq_logger = DAQLogger()

        # Initialize Sound Player
        sound_path = self.config.sound.sound_file
        if not os.path.isabs(sound_path):
            sound_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), sound_path)
        self.sound_player = SoundPlayer(
            sound_file_path=sound_path,
            enabled=self.config.sound.enabled,
            parent=self,
        )

        # Initialize Pupil Labs Neon Eye Tracker Worker
        self.neon_worker = NeonTrackerWorker(
            auto_discover=self.config.eyetracker.auto_discover,
            device_address=self.config.eyetracker.device_address,
            device_port=self.config.eyetracker.device_port,
            search_timeout_sec=self.config.eyetracker.search_timeout_sec,
            parent=self,
        )
        self.neon_worker.status_changed.connect(self.on_neon_status_changed)

        # Connect Subject Window scenario completion signal -> auto-stop
        if self.subject_window is not None:
            self.subject_window.scenario_completed.connect(self.on_scenario_completed)

        # Session tracking state
        self.is_recording = False
        self.current_session_dir = ""
        self.switch_event_count = 0
        self.session_start_time = None

        # Timer for updating recording duration in status bar
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_session_status)

        # Populate Scenarios Dropdown
        self._populate_scenarios()
        self.combo_scenario.currentTextChanged.connect(self.on_scenario_selected)

        # Sound feedback checkbox
        self.check_sound_feedback.setChecked(self.config.sound.enabled)
        self.check_sound_feedback.toggled.connect(self.sound_player.set_enabled)

        # Connect UI Buttons
        self.btn_start.clicked.connect(self.start_experiment)
        self.btn_stop.clicked.connect(self.stop_experiment)
        self.btn_toggle_subject_fullscreen.clicked.connect(self.toggle_subject_fullscreen)
        self.btn_mock_switch.clicked.connect(self.simulate_switch_press)

        # Space shortcut for mock switch testing
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_space.activated.connect(self.simulate_switch_press)

        # Start Workers
        self.camera_manager.start_cameras()
        self.daq_worker.start()
        self.neon_worker.start()

        # Initial UI state
        self._update_daq_label(False)
        self._update_experiment_state_ui(False)
        self.label_session_info.setText("Status: Standby | Ready for experiment")

    def _populate_scenarios(self):
        """
        Scans scenario directory for *.scenario files and populates dropdown menu.
        """
        self.combo_scenario.clear()
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        scenario_dir = self.config.subject_display.scenario_dir
        if not os.path.isabs(scenario_dir):
            scenario_dir = os.path.join(base_dir, scenario_dir)

        # Find all .scenario files
        search_pattern = os.path.join(scenario_dir, "*.scenario")
        files = sorted(glob.glob(search_pattern))

        # Also search root directory if empty
        if not files:
            files = sorted(glob.glob(os.path.join(base_dir, "*.scenario")))

        for fpath in files:
            fname = os.path.basename(fpath)
            self.combo_scenario.addItem(fname, fpath)

        # Select configured scenario if present
        target_name = os.path.basename(self.config.subject_display.scenario_file)
        idx = self.combo_scenario.findText(target_name)
        if idx >= 0:
            self.combo_scenario.setCurrentIndex(idx)
        elif self.combo_scenario.count() > 0:
            self.combo_scenario.setCurrentIndex(0)

        # Load initially selected scenario into subject window
        self.on_scenario_selected(self.combo_scenario.currentText())

    def on_scenario_selected(self, scenario_name: str):
        """
        Loads the selected scenario file into the SubjectWindow.
        """
        fpath = self.combo_scenario.currentData()
        if fpath and os.path.exists(fpath) and self.subject_window is not None:
            self.subject_window.load_scenario(fpath)
            print(f"[Scenario] Loaded scenario: {fpath}")

    @pyqtSlot(int, np.ndarray)
    def on_frame_received(self, cam_id: int, frame: np.ndarray):
        """
        Updates the corresponding camera QLabel with new video frame.
        """
        lbl = self.cam_labels.get(cam_id)
        if lbl is None or frame is None:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        lbl_size = lbl.size()
        pixmap = QPixmap.fromImage(q_img).scaled(
            lbl_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        lbl.setPixmap(pixmap)

    @pyqtSlot(bool)
    def on_daq_state_changed(self, is_pressed: bool):
        """
        Updates the DAQ status label: 'pressed' or 'idle'.
        """
        self._update_daq_label(is_pressed)

    def _update_daq_label(self, is_pressed: bool):
        if is_pressed:
            self.label_switch_status.setText("pressed")
            self.label_switch_status.setStyleSheet(
                "background-color: #28a745; color: #ffffff; "
                "border: 2px solid #34ce57; border-radius: 5px; "
                "font-size: 13px; font-weight: bold; padding: 2px;"
            )
        else:
            self.label_switch_status.setText("idle")
            self.label_switch_status.setStyleSheet(
                "background-color: #2b2b35; color: #a4a4b4; "
                "border: 2px solid #444455; border-radius: 5px; "
                "font-size: 13px; font-weight: bold; padding: 2px;"
            )

    @pyqtSlot(str, bool)
    def on_neon_status_changed(self, status_text: str, is_connected: bool):
        """
        Updates Pupil Labs Neon connection status indicator.
        """
        self.label_eyetracker_status.setText(status_text)
        if is_connected:
            self.label_eyetracker_status.setStyleSheet(
                "background-color: #1e7e34; color: #ffffff; "
                "border: 1px solid #28a745; border-radius: 5px; "
                "font-size: 12px; font-weight: bold; padding: 2px 6px;"
            )
        elif "Searching" in status_text:
            self.label_eyetracker_status.setStyleSheet(
                "background-color: #856404; color: #fff3cd; "
                "border: 1px solid #ffeeba; border-radius: 5px; "
                "font-size: 12px; font-weight: bold; padding: 2px 6px;"
            )
        else:
            self.label_eyetracker_status.setStyleSheet(
                "background-color: #495057; color: #f8d7da; "
                "border: 1px solid #6c757d; border-radius: 5px; "
                "font-size: 12px; font-weight: bold; padding: 2px 6px;"
            )

    @pyqtSlot(str)
    def on_daq_triggered(self, timestamp_ms: str):
        """
        Handles rising-edge switch trigger event.
        Plays instant sound feedback if enabled and logs to handle_switch.csv if recording.
        """
        # Instant sound feedback (cancels preceding sound immediately)
        if self.check_sound_feedback.isChecked():
            self.sound_player.play()

        # Log to CSV if session is active
        if self.is_recording and self.daq_logger.is_active:
            self.switch_event_count += 1
            self.daq_logger.log_event(
                channel=self.config.daq.channel,
                event="pressed",
                timestamp_str=timestamp_ms,
            )
            print(f"[Event] Switch pressed at {timestamp_ms} (Count: {self.switch_event_count})")

    def simulate_switch_press(self):
        """
        Triggers simulated switch press for test/mock purposes.
        """
        self.daq_worker.trigger_mock_press(duration_sec=0.25)

    def toggle_subject_fullscreen(self):
        """
        Toggles subject window fullscreen mode.
        """
        if self.subject_window is not None:
            self.subject_window.toggle_fullscreen()

    def _update_experiment_state_ui(self, is_recording: bool):
        """
        Updates Start/Stop button states and status indicator mutually exclusively.
        """
        if is_recording:
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.edit_subject_name.setEnabled(False)
            self.combo_scenario.setEnabled(False)
            self.label_rec_state.setText("RECORDING")
            self.label_rec_state.setStyleSheet(
                "background-color: #bd2130; color: #ffffff; border: 1px solid #dc3545; "
                "border-radius: 5px; font-size: 13px; font-weight: bold;"
            )
        else:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.edit_subject_name.setEnabled(True)
            self.combo_scenario.setEnabled(True)
            self.label_rec_state.setText("IDLE")
            self.label_rec_state.setStyleSheet(
                "background-color: #282832; color: #9090a0; border: 1px solid #444452; "
                "border-radius: 5px; font-size: 13px; font-weight: bold;"
            )

    def start_experiment(self):
        """
        Starts experiment session:
        - Validates subject name
        - Creates <output_root_dir>/<subject_name>/<yyyymmddHHMMSS>/
        - Begins AVI recording on all 4 cameras (cam_<id>.avi)
        - Initializes handle_switch.csv
        - Starts Neon eye tracker recording
        - Starts stimulus scenario on subject screen
        """
        raw_subject = self.edit_subject_name.text().strip()
        if not raw_subject:
            raw_subject = "subject_unnamed"
            self.edit_subject_name.setText(raw_subject)

        subject_name = "".join(c for c in raw_subject if c.isalnum() or c in ("-", "_")).strip()
        if not subject_name:
            subject_name = "subject"

        timestamp_dir = get_directory_timestamp()
        session_dir = os.path.join(self.config.output_root_dir, subject_name, timestamp_dir)

        try:
            os.makedirs(session_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Directory Error", f"Failed to create session directory: {e}")
            return

        self.current_session_dir = session_dir
        self.switch_event_count = 0
        self.is_recording = True

        # Initialize CSV logger
        self.daq_logger.start_logging(session_dir, "handle_switch.csv")

        # Start Camera Video Writers
        self.camera_manager.start_recording(session_dir)

        # Start Pupil Labs Neon recording
        self.neon_worker.start_recording()

        # Start Scenario playback on Subject Window
        if self.subject_window is not None:
            # Ensure latest selected scenario is loaded
            fpath = self.combo_scenario.currentData()
            if fpath and os.path.exists(fpath):
                self.subject_window.load_scenario(fpath)
            self.subject_window.start_scenario()

        # Update UI Controls & Indicators
        self._update_experiment_state_ui(True)

        self.status_timer.start(500)
        start_ts = get_formatted_timestamp_ms()
        self.label_session_info.setText(
            f"Recording Active | Dir: {session_dir} | Started: {start_ts}"
        )
        print(f"[Session] Started experiment recording at {session_dir}")

    def stop_experiment(self):
        """
        Stops experiment session:
        - Safely stops video recording
        - Finalizes handle_switch.csv
        - Stops Neon eye tracker recording
        - Stops scenario stimulus on subject window (reverts to black screen)
        - Keeps real-time camera preview active
        """
        if not self.is_recording:
            return

        self.is_recording = False
        self.status_timer.stop()

        # Stop camera video recording (preview remains active!)
        self.camera_manager.stop_recording()

        # Stop Pupil Labs Neon recording
        self.neon_worker.stop_recording()

        # Safely close CSV file
        self.daq_logger.stop_logging()

        # Stop Subject Window scenario (reverts to black screen)
        if self.subject_window is not None:
            self.subject_window.stop_scenario()

        # Update UI Controls & Indicators
        self._update_experiment_state_ui(False)

        stop_ts = get_formatted_timestamp_ms()
        self.label_session_info.setText(
            f"Recording Stopped | Saved: {self.current_session_dir} | "
            f"Total Switch Events: {self.switch_event_count} | Stopped: {stop_ts}"
        )
        print(f"[Session] Stopped recording. Data saved in {self.current_session_dir}")

    @pyqtSlot()
    def on_scenario_completed(self):
        """
        Slot triggered when subject window scenario completes.
        Automatically transitions system to stopped state.
        """
        print("[Session] Scenario playback finished. Automatically stopping experiment.")
        if self.is_recording:
            self.stop_experiment()

    def _update_session_status(self):
        """
        Periodic status refresh while recording.
        """
        if self.is_recording:
            self.label_session_info.setText(
                f"Recording in progress | Dir: {os.path.basename(self.current_session_dir)} | "
                f"Switch Events Logged: {self.switch_event_count}"
            )

    def closeEvent(self, event):
        """
        Safe application exit:
        - Finalizes recordings, Neon, and CSV
        - Stops threads
        - Closes subject window
        """
        print("[App] Closing application safely...")
        if self.is_recording:
            self.stop_experiment()

        self.status_timer.stop()
        self.camera_manager.stop_cameras()
        self.daq_worker.stop()
        self.neon_worker.stop()

        if self.subject_window is not None:
            self.subject_window.close()

        super().closeEvent(event)
