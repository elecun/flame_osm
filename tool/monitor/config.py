"""Configuration loader for default.cfg."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DisplayConfig:
    main_monitor: str = "auto"
    subject_monitor: str = "auto"
    subject_width: int = 1920
    subject_height: int = 1080
    subject_fullscreen: bool = True


@dataclass
class CameraConfig:
    max_cameras: int = 4
    device_indices: Optional[list[int]] = None  # None means auto-probe
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    preview_scale: float = 0.3


@dataclass
class SwitchConfig:
    device: str = "USB-4750,BID#0"
    channel: int = 0
    poll_interval_ms: int = 5
    active_low: bool = True


@dataclass
class EyeTrackerConfig:
    ip: str = "192.168.18"
    port: int = 8080
    discover_timeout_sec: float = 5.0


@dataclass
class ScenarioConfig:
    directory: Path = Path("scenario")
    video_file: str = "sample-10s.mp4"

    @property
    def video_path(self) -> Path:
        return self.directory / self.video_file


@dataclass
class RecordingConfig:
    output_dir: Path = Path("record")
    default_subject: str = "subject_01"


@dataclass
class AppConfig:
    base_dir: Path
    display: DisplayConfig
    cameras: CameraConfig
    switch: SwitchConfig
    eyetracker: EyeTrackerConfig
    scenario: ScenarioConfig
    recording: RecordingConfig


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value.strip())
    return path if path.is_absolute() else (base_dir / path)


def load_config(config_path: Path) -> AppConfig:
    """Load application configuration from default.cfg."""
    base_dir = config_path.resolve().parent
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if config_path.is_file():
        parser.read(config_path, encoding="utf-8")

    # [display]
    display = DisplayConfig(
        main_monitor=parser.get("display", "main_monitor", fallback="auto").strip(),
        subject_monitor=parser.get("display", "subject_monitor", fallback="auto").strip(),
        subject_width=parser.getint("display", "subject_width", fallback=1920),
        subject_height=parser.getint("display", "subject_height", fallback=1080),
        subject_fullscreen=parser.getboolean("display", "subject_fullscreen", fallback=True),
    )

    # [cameras]
    raw_indices = parser.get("cameras", "device_indices", fallback="auto").strip()
    if not raw_indices or raw_indices.lower() == "auto":
        indices = None
    else:
        cleaned = raw_indices.strip("[]() ")
        indices = [int(tok.strip()) for tok in cleaned.split(",") if tok.strip()]

    cameras = CameraConfig(
        max_cameras=max(1, min(4, parser.getint("cameras", "max_cameras", fallback=4))),
        device_indices=indices,
        width=parser.getint("cameras", "width", fallback=1280),
        height=parser.getint("cameras", "height", fallback=720),
        fps=parser.getint("cameras", "fps", fallback=30),
        fourcc=parser.get("cameras", "fourcc", fallback="MJPG").strip(),
        preview_scale=max(0.1, min(1.0, parser.getfloat("cameras", "preview_scale", fallback=0.3))),
    )

    # [switch]
    switch = SwitchConfig(
        device=parser.get("switch", "device", fallback="USB-4750,BID#0").strip(),
        channel=parser.getint("switch", "channel", fallback=0),
        poll_interval_ms=max(1, parser.getint("switch", "poll_interval_ms", fallback=5)),
        active_low=parser.getboolean("switch", "active_low", fallback=True),
    )

    # [eyetracker]
    eyetracker = EyeTrackerConfig(
        ip=parser.get("eyetracker", "ip", fallback="192.168.18").strip(),
        port=parser.getint("eyetracker", "port", fallback=8080),
        discover_timeout_sec=parser.getfloat("eyetracker", "discover_timeout_sec", fallback=5.0),
    )

    # [scenario]
    scenario_dir = _resolve_path(
        base_dir, parser.get("scenario", "directory", fallback="scenario")
    )
    scenario = ScenarioConfig(
        directory=scenario_dir,
        video_file=parser.get("scenario", "video_file", fallback="sample-10s.mp4").strip(),
    )

    # [recording]
    output_dir = _resolve_path(
        base_dir, parser.get("recording", "output_dir", fallback="record")
    )
    recording = RecordingConfig(
        output_dir=output_dir,
        default_subject=parser.get("recording", "default_subject", fallback="subject_01").strip(),
    )

    return AppConfig(
        base_dir=base_dir,
        display=display,
        cameras=cameras,
        switch=switch,
        eyetracker=eyetracker,
        scenario=scenario,
        recording=recording,
    )
