"""Subject stimulus window (1920x1080 fullscreen on left monitor) powered by libVLC."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import vlc
from PyQt6 import uic
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QWidget

from config import ScenarioConfig


def now_ms() -> int:
    """Return current Unix epoch timestamp in milliseconds."""
    return time.time_ns() // 1_000_000


class SubjectWindow(QWidget):
    """Displays black background when idle and plays preloaded stimulus video via libVLC."""

    status_changed = pyqtSignal(bool, str)  # loaded, message
    playback_state_changed = pyqtSignal(str)  # "Playing", "Paused", "Stopped"
    video_ended = pyqtSignal()  # emitted when stimulus video finishes naturally

    def __init__(self, ui_path: Path, scenario_config: ScenarioConfig) -> None:
        super().__init__()
        uic.loadUi(str(ui_path), self)

        self.scenario_config = scenario_config
        self._video_loaded = False
        self._recording = False
        self._csv_file = None
        self._csv_writer = None
        self._stop_logged = False
        self._last_emitted_state = "Stopped"

        self.vlc_instance = vlc.Instance(
            "--no-video-title-show",
            "--quiet",
            "--avcodec-hw=none",
        )
        self.vlc_player = self.vlc_instance.media_player_new()
        self._vlc_media = None

        # Poll VLC state and frame progress for UI state & scenario.csv logging
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(15)
        self._poll_timer.timeout.connect(self._on_poll_tick)
        self._poll_timer.start()

        self.show_black_screen()
        self.preload_video()

    def _attach_video_output(self) -> None:
        win_id = int(self.videoWidget.winId())
        self.vlc_player.set_xwindow(win_id)

    def _apply_audio_device(self) -> None:
        preferred = (self.scenario_config.audio_device or "").strip().lower()
        if preferred and preferred != "default":
            mods = self.vlc_player.audio_output_device_enum()
            if mods:
                cur = mods
                while cur:
                    dev = cur.contents
                    dev_id = (dev.device or b"").decode("utf-8", errors="ignore")
                    desc = (dev.description or b"").decode("utf-8", errors="ignore")
                    if preferred in desc.lower() or preferred in dev_id.lower():
                        self.vlc_player.audio_output_device_set(None, dev_id)
                        break
                    cur = dev.next
                vlc.libvlc_audio_output_device_list_release(mods)
        self.vlc_player.audio_set_mute(False)
        self.vlc_player.audio_set_volume(100)

    @property
    def is_video_loaded(self) -> bool:
        return self._video_loaded

    def preload_video(self) -> None:
        video_path = self.scenario_config.video_path
        if not video_path.is_file():
            self._video_loaded = False
            self.status_changed.emit(False, f"Not found: {video_path.name}")
            return

        self._vlc_media = self.vlc_instance.media_new(str(video_path.resolve()))
        self.vlc_player.set_media(self._vlc_media)
        self._video_loaded = True
        self.status_changed.emit(True, f"Loaded (VLC): {video_path.name}")

    def show_black_screen(self) -> None:
        self.stimulusStack.setCurrentIndex(0)

    def show_video_screen(self) -> None:
        self.stimulusStack.setCurrentIndex(1)

    def _start_playback(self) -> None:
        if not self._video_loaded:
            self.preload_video()
        if not self._video_loaded:
            self.show_black_screen()
            return

        self.show_video_screen()
        self._attach_video_output()
        self.vlc_player.stop()
        if self._vlc_media is not None:
            self.vlc_player.set_media(self._vlc_media)
        self.vlc_player.play()
        QTimer.singleShot(100, self._apply_audio_device)

    def _write_scenario_row(self, event: str) -> None:
        if self._csv_writer is None or self._csv_file is None:
            return
        self._csv_writer.writerow([
            event,
            datetime.now().astimezone().isoformat(timespec="milliseconds"),
            now_ms(),
        ])
        self._csv_file.flush()

    def start_stimulus(self, session_dir: Path) -> None:
        csv_path = session_dir / "scenario.csv"
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["event", "datetime", "timestamp_ms"])
        self._stop_logged = False
        self._recording = True
        self._start_playback()
        self._write_scenario_row("record_start")

    def stop_stimulus(self) -> None:
        was_recording = self._recording
        self._recording = False
        self.vlc_player.stop()
        self.show_black_screen()
        self._emit_state("Stopped")

        if self._csv_file is not None:
            if was_recording and not self._stop_logged:
                self._write_scenario_row("record_stop")
                self._stop_logged = True
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def test_play(self) -> None:
        """Play preloaded scenario video from the beginning without recording."""
        self._start_playback()

    def test_pause(self) -> None:
        """Pause scenario test playback."""
        self.vlc_player.set_pause(1)

    def test_resume(self) -> None:
        """Resume paused scenario test playback."""
        if self._video_loaded:
            self.show_video_screen()
            self.vlc_player.set_pause(0)

    def test_stop(self) -> None:
        """Stop scenario test playback and return to black screen."""
        self.vlc_player.stop()
        self.show_black_screen()
        self._emit_state("Stopped")

    def _emit_state(self, state: str) -> None:
        if state != self._last_emitted_state:
            self._last_emitted_state = state
            self.playback_state_changed.emit(state)

    def _on_poll_tick(self) -> None:
        state = self.vlc_player.get_state()
        if state == vlc.State.Playing:
            self._emit_state("Playing")
        elif state == vlc.State.Paused:
            self._emit_state("Paused")
        elif state == vlc.State.Ended:
            was_recording = self._recording
            if self._recording and not self._stop_logged:
                self._write_scenario_row("record_stop")
                self._stop_logged = True
            self.show_black_screen()
            self._emit_state("Stopped")
            if was_recording:
                self.video_ended.emit()
        elif state in (vlc.State.Stopped, vlc.State.NothingSpecial, vlc.State.Error):
            self._emit_state("Stopped")

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        self.stop_stimulus()
        self.vlc_player.release()
        self.vlc_instance.release()
        super().closeEvent(event)
