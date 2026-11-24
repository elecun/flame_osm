"""Main monitoring and control window (loaded from ui/monitor_window.ui)."""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QMessageBox

from camera_manager import CameraManager
from config import AppConfig
from eyetracker_manager import EyeTrackerManager
from subject_window import SubjectWindow
from switch_manager import SwitchManager


def _sanitize_subject_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    return cleaned.strip("._")


class MonitorWindow(QMainWindow):
    """Main operator window for live monitoring and synchronized recording."""

    def __init__(
        self,
        ui_path: Path,
        config: AppConfig,
        subject_window: SubjectWindow,
    ) -> None:
        super().__init__()
        uic.loadUi(str(ui_path), self)

        self.config = config
        self.subject_window = subject_window

        self._cam_labels = [
            self.camLabel0,
            self.camLabel1,
            self.camLabel2,
            self.camLabel3,
        ]
        self._preview_scale = self.config.cameras.preview_scale
        init_w = max(1, int(self.config.cameras.width * self._preview_scale))
        init_h = max(1, int(self.config.cameras.height * self._preview_scale))
        for label in self._cam_labels:
            label.setFixedSize(init_w, init_h)

        self._cameras_ready = False
        self._switch_ready = False
        self._eyetracker_ready = False
        self._recording = False
        self._record_start_monotonic: Optional[float] = None
        self._current_session_dir: Optional[Path] = None

        self.subjectNameEdit.setText(self.config.recording.default_subject)

        # Device managers
        self.camera_manager = CameraManager(self.config.cameras, self)
        self.switch_manager = SwitchManager(self.config.switch, self)
        self.eyetracker_manager = EyeTrackerManager(self.config.eyetracker, self)

        # Connect signals
        self.camera_manager.frame_ready.connect(self._on_camera_frame)
        self.camera_manager.status_changed.connect(self._on_camera_status)

        self.switch_manager.status_changed.connect(self._on_switch_status)
        self.switch_manager.state_changed.connect(self._on_switch_state)

        self.eyetracker_manager.status_changed.connect(self._on_eyetracker_status)
        self.subject_window.status_changed.connect(self._on_scenario_status)
        self.subject_window.playback_state_changed.connect(self._on_scenario_playback_state)

        self.recordStartButton.clicked.connect(self._on_record_start)
        self.recordStopButton.clicked.connect(self._on_record_stop)
        self.reconnectButton.clicked.connect(self._connect_all_devices)

        self.scenarioPlayButton.clicked.connect(self.subject_window.test_play)
        self.scenarioPauseButton.clicked.connect(self.subject_window.test_pause)
        self.scenarioResumeButton.clicked.connect(self.subject_window.test_resume)
        self.scenarioStopButton.clicked.connect(self.subject_window.test_stop)

        # Elapsed timer during recording
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(200)
        self._elapsed_timer.timeout.connect(self._update_elapsed_label)

        # Reflect initial scenario preload status
        if self.subject_window.is_video_loaded:
            self._on_scenario_status(True, f"Loaded: {self.config.scenario.video_file}")
        else:
            self._on_scenario_status(False, f"Not found: {self.config.scenario.video_file}")

        # Start device initialization shortly after window shows
        QTimer.singleShot(100, self._connect_all_devices)

    def _connect_all_devices(self) -> None:
        if self._recording:
            return
        self.statusbar.showMessage("Initializing devices...")
        self.camera_manager.start_cameras()
        self.switch_manager.start()
        self.eyetracker_manager.connect_async()

    def _update_record_buttons(self) -> None:
        all_ready = self._cameras_ready and self._switch_ready and self._eyetracker_ready
        self.recordStartButton.setEnabled(all_ready and not self._recording)
        self.recordStopButton.setEnabled(self._recording)
        self.subjectNameEdit.setEnabled(not self._recording)
        self.reconnectButton.setEnabled(not self._recording)
        for btn in (
            self.scenarioPlayButton,
            self.scenarioPauseButton,
            self.scenarioResumeButton,
            self.scenarioStopButton,
        ):
            btn.setEnabled(not self._recording)

    def _on_camera_frame(self, slot_index: int, frame: np.ndarray) -> None:
        if not (0 <= slot_index < len(self._cam_labels)):
            return
        label = self._cam_labels[slot_index]
        h, w = frame.shape[:2]
        preview_w = max(1, int(w * self._preview_scale))
        preview_h = max(1, int(h * self._preview_scale))
        resized = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, preview_w, preview_h, 3 * preview_w, QImage.Format.Format_RGB888
        ).copy()
        if label.width() != preview_w or label.height() != preview_h:
            label.setFixedSize(preview_w, preview_h)
        label.setPixmap(QPixmap.fromImage(qimg))

    def _on_camera_status(self, connected: bool, count: int, message: str) -> None:
        self._cameras_ready = connected
        color = "#66bb6a" if connected else "#ef5350"
        self.cameraStatusLabel.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.cameraStatusLabel.setText(f"Cameras: {message}")
        for idx in range(count, len(self._cam_labels)):
            self._cam_labels[idx].clear()
            self._cam_labels[idx].setText(f"Camera {idx} - No Signal")
        self._update_record_buttons()

    def _on_switch_status(self, connected: bool, message: str) -> None:
        self._switch_ready = connected
        color = "#66bb6a" if connected else "#ef5350"
        self.switchStatusLabel.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.switchStatusLabel.setText(f"USB-4750: {message}")
        self._update_record_buttons()

    def _on_switch_state(self, state: int, timestamp_ms: int) -> None:
        if state == 1:
            self.switchStateLabel.setStyleSheet("color: #ffca28; font-weight: bold;")
            self.switchStateLabel.setText("Handle Switch (IDI0): PRESSED (1)")
        else:
            self.switchStateLabel.setStyleSheet("color: #90caf9; font-weight: bold;")
            self.switchStateLabel.setText("Handle Switch (IDI0): RELEASED (0)")

    def _on_eyetracker_status(self, connected: bool, message: str) -> None:
        self._eyetracker_ready = connected
        color = "#66bb6a" if connected else "#ef5350"
        self.eyeTrackerStatusLabel.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.eyeTrackerStatusLabel.setText(f"Neon Eye Tracker: {message}")
        self._update_record_buttons()

    def _on_scenario_status(self, loaded: bool, message: str) -> None:
        color = "#66bb6a" if loaded else "#ffa726"
        self.scenarioStatusLabel.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.scenarioStatusLabel.setText(f"Scenario: {message}")

    def _on_scenario_playback_state(self, state: str) -> None:
        colors = {
            "Playing": "#66bb6a",
            "Paused": "#ffca28",
            "Stopped": "#aaaaaa",
        }
        color = colors.get(state, "#aaaaaa")
        self.scenarioPlaybackStateLabel.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {color}; padding-left: 12px;"
        )
        self.scenarioPlaybackStateLabel.setText(f"Video: {state}")

    def _on_record_start(self) -> None:
        if self._recording:
            return

        subject_name = _sanitize_subject_name(self.subjectNameEdit.text())
        if not subject_name:
            QMessageBox.warning(
                self,
                "Invalid Subject Name",
                "Please enter a valid subject name before starting recording.",
            )
            return

        timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.config.recording.output_dir / subject_name / timestamp_folder
        session_dir.mkdir(parents=True, exist_ok=True)
        self._current_session_dir = session_dir

        # Start synchronized recording across all devices + stimulus playback
        self.eyetracker_manager.start_recording(session_dir)
        self.switch_manager.start_recording(session_dir)
        self.camera_manager.start_recording(session_dir)
        self.subject_window.start_stimulus(session_dir)

        self._recording = True
        self._record_start_monotonic = time.monotonic()
        self._elapsed_timer.start()
        self.recordingStatusLabel.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ef5350; padding-left: 12px;"
        )
        self.recordingStatusLabel.setText("Status: RECORDING (00:00)")
        self.statusbar.showMessage(f"Recording to: {session_dir}")
        self._update_record_buttons()

    def _on_record_stop(self) -> None:
        if not self._recording:
            return

        self._recording = False
        self._elapsed_timer.stop()

        # Stop stimulus and return subject window to black screen
        self.subject_window.stop_stimulus()
        # Stop recording on cameras (keeps live camera monitoring active)
        self.camera_manager.stop_recording()
        # Stop switch recording
        self.switch_manager.stop_recording()
        # Stop Neon eye tracker recording
        self.eyetracker_manager.stop_recording()

        self.recordingStatusLabel.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #aaaaaa; padding-left: 12px;"
        )
        self.recordingStatusLabel.setText("Status: Idle")
        if self._current_session_dir is not None:
            self.statusbar.showMessage(f"Saved session to: {self._current_session_dir}")
        self._update_record_buttons()

    def _update_elapsed_label(self) -> None:
        if self._record_start_monotonic is None:
            return
        elapsed = int(time.monotonic() - self._record_start_monotonic)
        mins, secs = divmod(elapsed, 60)
        self.recordingStatusLabel.setText(f"Status: RECORDING ({mins:02d}:{secs:02d})")

    def closeEvent(self, event) -> None:
        if self._recording:
            self._on_record_stop()
        self.camera_manager.stop_cameras()
        self.switch_manager.stop()
        self.eyetracker_manager.close()
        self.subject_window.close()
        super().closeEvent(event)
