"""
Subject stimulus presentation window.
Displays real-time dynamic visual stimuli generated via OpenCV per scenario schedule.
Supports full-screen toggle, pure black initial screen, and automatic completion notification.
"""

import os
import cv2
import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from src.config import SubjectDisplayConfig
from src.scenario.scenario_player import ScenarioPlayer


class SubjectWindow(QMainWindow):
    """
    Subject presentation window displaying OpenCV-rendered stimulus.
    """
    # Emitted when scenario reaches completion
    scenario_completed = pyqtSignal()

    def __init__(self, config: SubjectDisplayConfig, parent=None):
        super().__init__(parent)
        self.config = config

        # Load Qt Designer UI file
        ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ui", "subject_window.ui")
        uic.loadUi(ui_path, self)

        # Configure window size
        self.resize(self.config.width, self.config.height)

        # Setup scenario player
        self.player = ScenarioPlayer(width=self.config.width, height=self.config.height)
        scenario_path = self.config.scenario_file
        if not os.path.isabs(scenario_path):
            scenario_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), scenario_path)
        self.player.load_scenario(scenario_path)

        # Fullscreen shortcut (F11)
        self.shortcut_f11 = QShortcut(QKeySequence("F11"), self)
        self.shortcut_f11.activated.connect(self.toggle_fullscreen)

        # Escape shortcut to exit fullscreen
        self.shortcut_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.shortcut_esc.activated.connect(self.exit_fullscreen)

        # Stimulus render timer (~30 FPS)
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._render_next_frame)
        self.render_timer.start(33)

        # Apply initial fullscreen setting if configured
        if self.config.fullscreen:
            self.showFullScreen()
        else:
            self.show()

    def load_scenario(self, file_path: str):
        """
        Loads a new scenario file.
        """
        self.player.load_scenario(file_path)

    def start_scenario(self):
        """
        Starts rendering scenario stimulus.
        """
        self.player.start()

    def stop_scenario(self):
        """
        Stops rendering scenario, reverting to pure black screen.
        """
        self.player.stop()

    def toggle_fullscreen(self):
        """
        Toggles between fullscreen and normal windowed mode.
        """
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()

    def mouseDoubleClickEvent(self, event):
        """
        Double-clicking window toggles fullscreen.
        """
        self.toggle_fullscreen()
        super().mouseDoubleClickEvent(event)

    def _render_next_frame(self):
        """
        Fetches next frame from ScenarioPlayer and renders to QLabel.
        Notifies when scenario has concluded.
        """
        frame = self.player.get_frame()

        if getattr(self.player, "is_finished", False):
            self.player.is_finished = False
            self.scenario_completed.emit()

        if frame is None or not hasattr(self, "label_stimulus"):
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # Convert BGR (OpenCV) to RGB for QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale pixmap smoothly to fit window/label while keeping aspect ratio
        lbl_size = self.label_stimulus.size()
        pixmap = QPixmap.fromImage(q_img).scaled(
            lbl_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.label_stimulus.setPixmap(pixmap)

    def closeEvent(self, event):
        self.render_timer.stop()
        super().closeEvent(event)
