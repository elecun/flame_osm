"""
Configuration manager for simulator monitor.
Loads and validates user settings from .cfg file.
"""

import configparser
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class CameraConfig:
    camera_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    fps: int = 30
    width: int = 640
    height: int = 480
    codec: str = "XVID"
    mock_if_missing: bool = True


@dataclass
class SubjectDisplayConfig:
    width: int = 1920
    height: int = 1080
    fullscreen: bool = False
    scenario_dir: str = "./scenario"
    scenario_file: str = "default.scenario"
    bg_color_bgr: tuple = (0, 0, 0)


@dataclass
class DAQConfig:
    device_description: str = "USB-4750,BID#0"
    profile_path: str = ""
    port: int = 0
    channel: int = 0
    poll_interval_ms: int = 10
    mock_mode: str = "auto"  # 'auto', 'true', 'false'


@dataclass
class SoundConfig:
    sound_file: str = "assets/beep.wav"
    enabled: bool = True


@dataclass
class EyeTrackerConfig:
    auto_discover: bool = True
    device_address: str = ""
    device_port: int = 8080
    search_timeout_sec: float = 3.0


@dataclass
class AppConfig:
    output_root_dir: str = "./records"
    camera: CameraConfig = field(default_factory=CameraConfig)
    subject_display: SubjectDisplayConfig = field(default_factory=SubjectDisplayConfig)
    daq: DAQConfig = field(default_factory=DAQConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)
    eyetracker: EyeTrackerConfig = field(default_factory=EyeTrackerConfig)


def load_config(config_path: str) -> AppConfig:
    """
    Parses configuration from the specified .cfg file.
    Falls back to default values if keys/sections are missing.
    """
    config = AppConfig()
    parser = configparser.ConfigParser()

    if not os.path.exists(config_path):
        print(f"[Config] Warning: Configuration file '{config_path}' not found. Using defaults.")
        return config

    parser.read(config_path, encoding="utf-8")

    # SYSTEM
    if parser.has_section("SYSTEM"):
        config.output_root_dir = parser.get("SYSTEM", "output_root_dir", fallback=config.output_root_dir)

    # CAMERAS
    if parser.has_section("CAMERAS"):
        cam_ids_str = parser.get("CAMERAS", "camera_ids", fallback="0, 1, 2, 3")
        try:
            config.camera.camera_ids = [int(x.strip()) for x in cam_ids_str.split(",") if x.strip()]
        except ValueError:
            config.camera.camera_ids = [0, 1, 2, 3]

        config.camera.fps = parser.getint("CAMERAS", "fps", fallback=config.camera.fps)
        config.camera.width = parser.getint("CAMERAS", "width", fallback=config.camera.width)
        config.camera.height = parser.getint("CAMERAS", "height", fallback=config.camera.height)
        config.camera.codec = parser.get("CAMERAS", "codec", fallback=config.camera.codec)
        config.camera.mock_if_missing = parser.getboolean(
            "CAMERAS", "mock_if_missing", fallback=config.camera.mock_if_missing
        )

    # SUBJECT_DISPLAY
    if parser.has_section("SUBJECT_DISPLAY"):
        config.subject_display.width = parser.getint(
            "SUBJECT_DISPLAY", "width", fallback=config.subject_display.width
        )
        config.subject_display.height = parser.getint(
            "SUBJECT_DISPLAY", "height", fallback=config.subject_display.height
        )
        config.subject_display.fullscreen = parser.getboolean(
            "SUBJECT_DISPLAY", "fullscreen", fallback=config.subject_display.fullscreen
        )
        config.subject_display.scenario_dir = parser.get(
            "SUBJECT_DISPLAY", "scenario_dir", fallback=config.subject_display.scenario_dir
        )
        config.subject_display.scenario_file = parser.get(
            "SUBJECT_DISPLAY", "scenario_file", fallback=config.subject_display.scenario_file
        )

    # DAQ
    if parser.has_section("DAQ"):
        config.daq.device_description = parser.get(
            "DAQ", "device_description", fallback=config.daq.device_description
        )
        config.daq.profile_path = parser.get("DAQ", "profile_path", fallback=config.daq.profile_path)
        config.daq.port = parser.getint("DAQ", "port", fallback=config.daq.port)
        config.daq.channel = parser.getint("DAQ", "channel", fallback=config.daq.channel)
        config.daq.poll_interval_ms = parser.getint(
            "DAQ", "poll_interval_ms", fallback=config.daq.poll_interval_ms
        )
        config.daq.mock_mode = parser.get("DAQ", "mock_mode", fallback=config.daq.mock_mode).strip().lower()

    # SOUND
    if parser.has_section("SOUND"):
        config.sound.sound_file = parser.get("SOUND", "sound_file", fallback=config.sound.sound_file)
        config.sound.enabled = parser.getboolean("SOUND", "sound_feedback", fallback=config.sound.enabled)

    # EYETRACKER
    if parser.has_section("EYETRACKER"):
        config.eyetracker.auto_discover = parser.getboolean(
            "EYETRACKER", "auto_discover", fallback=config.eyetracker.auto_discover
        )
        config.eyetracker.device_address = parser.get(
            "EYETRACKER", "device_address", fallback=config.eyetracker.device_address
        )
        config.eyetracker.device_port = parser.getint(
            "EYETRACKER", "device_port", fallback=config.eyetracker.device_port
        )
        config.eyetracker.search_timeout_sec = parser.getfloat(
            "EYETRACKER", "search_timeout_sec", fallback=config.eyetracker.search_timeout_sec
        )

    return config
