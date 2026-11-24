"""Subject stimulus window (1920x1080 fullscreen on left monitor)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

from PyQt6 import uic
from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import QWidget

from config import ScenarioConfig


def now_ms() -> int:
    """Return current Unix epoch timestamp in milliseconds."""
    return time.time_ns() // 1_000_000


class SubjectWindow(QWidget):
    """Displays black background when idle and plays preloaded stimulus video during recording."""

    status_changed = pyqtSignal(bool, str)  # loaded, message
    playback_state_changed = pyqtSignal(str)  # "Playing", "Paused", "Stopped"

    def __init__(self, ui_path: Path, scenario_config: ScenarioConfig) -> None:
        super().__init__()
        uic.loadUi(str(ui_path), self)

        self.scenario_config = scenario_config
        self._video_loaded = False
        self._prerolling = False
        self._recording = False
        self._csv_file = None
        self._csv_writer = None
        self._frame_index = 0

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)

        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.videoWidget)

        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.errorOccurred.connect(self._on_player_error)

        sink = self.videoWidget.videoSink()
        if sink is not None:
            sink.videoFrameChanged.connect(self._on_video_frame_changed)

        self.show_black_screen()
        self.preload_video()

    @property
    def is_video_loaded(self) -> bool:
        return self._video_loaded

    def preload_video(self) -> None:
        video_path = self.scenario_config.video_path
        if not video_path.is_file():
            self._video_loaded = False
            self.status_changed.emit(False, f"Not found: {video_path.name}")
            return

        self._prerolling = True
        self.player.setSource(QUrl.fromLocalFile(str(video_path.resolve())))
        self.player.pause()
        self.player.setPosition(0)
        self._video_loaded = True
        self.status_changed.emit(True, f"Loaded (Prerolled): {video_path.name}")

    def show_black_screen(self) -> None:
        self.stimulusStack.setCurrentIndex(0)

    def show_video_screen(self) -> None:
        self.stimulusStack.setCurrentIndex(1)

    def start_stimulus(self, session_dir: Path) -> None:
        csv_path = session_dir / "scenario.csv"
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["timestamp_ms", "frame_index"])
        self._frame_index = 0
        self._recording = True

        if self._video_loaded:
            self._prerolling = False
            self.player.setPosition(0)
            self.show_video_screen()
            self.player.play()
        else:
            self.show_black_screen()

    def stop_stimulus(self) -> None:
        self._recording = False
        self._prerolling = True
        self.player.pause()
        self.player.setPosition(0)
        self.show_black_screen()
        self.playback_state_changed.emit("Stopped")

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def test_play(self) -> None:
        """Play preloaded scenario video from the beginning without recording."""
        if not self._video_loaded:
            self.preload_video()
        if self._video_loaded:
            self._prerolling = False
            self.player.setPosition(0)
            self.show_video_screen()
            self.player.play()

    def test_pause(self) -> None:
        """Pause scenario test playback."""
        self._prerolling = False
        self.player.pause()

    def test_resume(self) -> None:
        """Resume paused scenario test playback."""
        if self._video_loaded:
            self._prerolling = False
            self.show_video_screen()
            self.player.play()

    def test_stop(self) -> None:
        """Stop scenario test playback, return to black screen, and re-preroll at 0ms."""
        self._prerolling = True
        self.player.pause()
        self.player.setPosition(0)
        self.show_black_screen()
        self.playback_state_changed.emit("Stopped")

    def _on_video_frame_changed(self, frame) -> None:
        if not self._recording or self._csv_writer is None:
            return
        if frame is None or not frame.isValid():
            return
        ts = now_ms()
        self._csv_writer.writerow([ts, self._frame_index])
        self._frame_index += 1

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.show_black_screen()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if self._prerolling:
            self.playback_state_changed.emit("Stopped")
            return
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.playback_state_changed.emit("Playing")
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.playback_state_changed.emit("Paused")
        else:
            self.playback_state_changed.emit("Stopped")

    def _on_player_error(self, error, error_string: str) -> None:
        self._video_loaded = False
        self.status_changed.emit(False, f"Playback error: {error_string}")
        self.show_black_screen()

    def closeEvent(self, event) -> None:
        self.stop_stimulus()
        super().closeEvent(event)
