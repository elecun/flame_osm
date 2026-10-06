"""
Sound Feedback Player using PyQt6.QtMultimedia.
Preloads audio into memory on startup for zero-latency playback.
Instantly cancels preceding sound on rapid presses to prevent overlapping/mixing.
"""

import os
from PyQt6.QtCore import QObject, QUrl
from PyQt6.QtMultimedia import QSoundEffect


class SoundPlayer(QObject):
    """
    Handles instantaneous audio playback for switch presses.
    """

    def __init__(self, sound_file_path: str, enabled: bool = True, parent=None):
        super().__init__(parent)
        self.sound_file_path = sound_file_path
        self.enabled = enabled
        self._sound_effect = None

        self._load_sound()

    def _load_sound(self):
        """
        Preloads sound into memory.
        """
        if not self.sound_file_path:
            return

        abs_path = os.path.abspath(self.sound_file_path)
        if not os.path.exists(abs_path):
            print(f"[SoundPlayer] Sound file not found: {abs_path}")
            return

        try:
            self._sound_effect = QSoundEffect(self)
            self._sound_effect.setSource(QUrl.fromLocalFile(abs_path))
            self._sound_effect.setVolume(1.0)
            print(f"[SoundPlayer] Preloaded sound effect: {abs_path}")
        except Exception as e:
            print(f"[SoundPlayer] Failed to initialize QSoundEffect: {e}")

    def play(self):
        """
        Plays sound immediately.
        If a previous sound is currently playing, stops it and restarts from the beginning.
        """
        if not self.enabled or self._sound_effect is None:
            return

        try:
            # Prevent overlapping/mixing by stopping previous playback immediately
            if self._sound_effect.isPlaying():
                self._sound_effect.stop()
            self._sound_effect.play()
        except Exception as e:
            print(f"[SoundPlayer] Error during playback: {e}")

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def reload(self, sound_file_path: str):
        self.sound_file_path = sound_file_path
        self._load_sound()
