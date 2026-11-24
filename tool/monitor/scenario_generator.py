#!/usr/bin/env python3
"""Generate OpenCV visual stimulus scenario videos from scenario.cfg."""

from __future__ import annotations

import argparse
import configparser
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass
class VideoSpec:
    width: int
    height: int
    fps: int
    fmt: str
    output_dir: Path
    background_bgr: tuple[int, int, int]


def _parse_bgr(raw: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        return default
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _load_video_spec(parser: configparser.ConfigParser, base_dir: Path) -> VideoSpec:
    out_raw = parser.get("video", "output_dir", fallback="takes").strip()
    out_path = Path(out_raw)
    if not out_path.is_absolute():
        out_path = base_dir / out_path

    fmt = parser.get("video", "format", fallback="avi").strip().lower().lstrip(".")
    if fmt != "avi":
        raise ValueError(f"Unsupported video format '{fmt}'. Only 'avi' is supported.")

    return VideoSpec(
        width=parser.getint("video", "width", fallback=1920),
        height=parser.getint("video", "height", fallback=1080),
        fps=parser.getint("video", "fps", fallback=30),
        fmt=fmt,
        output_dir=out_path,
        background_bgr=_parse_bgr(
            parser.get("video", "background_bgr", fallback="0, 0, 0"), (0, 0, 0)
        ),
    )


def _open_writer(output_path: Path, spec: VideoSpec) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(spec.fps), (spec.width, spec.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
    return writer


def _make_background(spec: VideoSpec) -> np.ndarray:
    frame = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    frame[:] = spec.background_bgr
    return frame


def _blend_color(
    fg: tuple[int, int, int], bg: tuple[int, int, int], alpha: float
) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return (
        int(round(fg[0] * a + bg[0] * (1.0 - a))),
        int(round(fg[1] * a + bg[1] * (1.0 - a))),
        int(round(fg[2] * a + bg[2] * (1.0 - a))),
    )


def _write_blank(
    writer: cv2.VideoWriter,
    spec: VideoSpec,
    duration_sec: float,
) -> int:
    frames = max(0, int(round(duration_sec * spec.fps)))
    if frames == 0:
        return 0
    frame = _make_background(spec)
    for _ in range(frames):
        writer.write(frame)
    return frames


def _write_hold(
    writer: cv2.VideoWriter,
    spec: VideoSpec,
    pos: tuple[float, float],
    radius: int,
    color_bgr: tuple[int, int, int],
    duration_sec: float,
) -> int:
    frames = max(0, int(round(duration_sec * spec.fps)))
    if frames == 0:
        return 0
    frame = _make_background(spec)
    cv2.circle(
        frame,
        (int(round(pos[0])), int(round(pos[1]))),
        radius,
        color_bgr,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    for _ in range(frames):
        writer.write(frame)
    return frames


def _write_move(
    writer: cv2.VideoWriter,
    spec: VideoSpec,
    start_pos: tuple[float, float],
    end_pos: tuple[float, float],
    radius: int,
    color_bgr: tuple[int, int, int],
    duration_sec: float,
) -> int:
    frames = max(0, int(round(duration_sec * spec.fps)))
    if frames == 0:
        return 0
    bg = _make_background(spec)
    denom = max(1, frames - 1) if frames > 1 else 1
    for i in range(frames):
        t = (i + 1) / frames if frames > 0 else 1.0
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * t
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * t
        frame = bg.copy()
        cv2.circle(
            frame,
            (int(round(x)), int(round(y))),
            radius,
            color_bgr,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        writer.write(frame)
    return frames


def _write_fade_out(
    writer: cv2.VideoWriter,
    spec: VideoSpec,
    pos: tuple[float, float],
    radius: int,
    color_bgr: tuple[int, int, int],
    duration_sec: float,
) -> int:
    frames = max(0, int(round(duration_sec * spec.fps)))
    if frames == 0:
        return 0
    bg = _make_background(spec)
    center = (int(round(pos[0])), int(round(pos[1])))
    for i in range(frames):
        alpha = 1.0 - ((i + 1) / frames)
        frame = bg.copy()
        if alpha > 0.0:
            blended = _blend_color(color_bgr, spec.background_bgr, alpha)
            cv2.circle(frame, center, radius, blended, thickness=-1, lineType=cv2.LINE_AA)
        writer.write(frame)
    return frames


def generate_take_1(
    section: configparser.SectionProxy, spec: VideoSpec, output_path: Path
) -> int:
    """Take 1: Green circle moves center -> top-left -> bottom-left -> bottom-right -> top-right -> center -> disappear."""
    radius = max(1, section.getint("radius", fallback=100))
    # Clamp radius so circle always fits inside the frame
    max_r = min(spec.width, spec.height) // 2
    radius = min(radius, max_r)
    color_bgr = _parse_bgr(section.get("color_bgr", fallback="0, 255, 0"), (0, 255, 0))

    # All positions keep the full circle strictly inside [0, width] x [0, height]
    center = (spec.width / 2.0, spec.height / 2.0)
    top_left = (float(radius), float(radius))
    bottom_left = (float(radius), float(spec.height - radius))
    bottom_right = (float(spec.width - radius), float(spec.height - radius))
    top_right = (float(spec.width - radius), float(radius))

    wait_center_1 = section.getfloat("wait_center_1_sec", fallback=1.0)
    move_to_tl = section.getfloat("move_to_top_left_sec", fallback=2.0)
    wait_tl = section.getfloat("wait_top_left_sec", fallback=1.0)
    move_to_bl = section.getfloat("move_to_bottom_left_sec", fallback=2.0)
    wait_bl = section.getfloat("wait_bottom_left_sec", fallback=1.0)
    move_to_br = section.getfloat("move_to_bottom_right_sec", fallback=2.0)
    wait_br = section.getfloat("wait_bottom_right_sec", fallback=1.0)
    move_to_tr = section.getfloat("move_to_top_right_sec", fallback=2.0)
    wait_tr = section.getfloat("wait_top_right_sec", fallback=1.0)
    move_to_center = section.getfloat("move_to_center_sec", fallback=2.0)
    wait_center_2 = section.getfloat("wait_center_2_sec", fallback=1.0)

    writer = _open_writer(output_path, spec)
    total_frames = 0
    try:
        total_frames += _write_hold(writer, spec, center, radius, color_bgr, wait_center_1)
        total_frames += _write_move(writer, spec, center, top_left, radius, color_bgr, move_to_tl)
        total_frames += _write_hold(writer, spec, top_left, radius, color_bgr, wait_tl)
        total_frames += _write_move(writer, spec, top_left, bottom_left, radius, color_bgr, move_to_bl)
        total_frames += _write_hold(writer, spec, bottom_left, radius, color_bgr, wait_bl)
        total_frames += _write_move(writer, spec, bottom_left, bottom_right, radius, color_bgr, move_to_br)
        total_frames += _write_hold(writer, spec, bottom_right, radius, color_bgr, wait_br)
        total_frames += _write_move(writer, spec, bottom_right, top_right, radius, color_bgr, move_to_tr)
        total_frames += _write_hold(writer, spec, top_right, radius, color_bgr, wait_tr)
        total_frames += _write_move(writer, spec, top_right, center, radius, color_bgr, move_to_center)
        total_frames += _write_hold(writer, spec, center, radius, color_bgr, wait_center_2)
        # Circle disappears immediately after holding at center
        writer.write(_make_background(spec))
        total_frames += 1
    finally:
        writer.release()

    return total_frames


def generate_take_2(
    section: configparser.SectionProxy, spec: VideoSpec, output_path: Path
) -> int:
    """Take 2: Green circle flashes at random in-bounds positions at fixed intervals."""
    radius = max(1, section.getint("radius", fallback=60))
    max_r = min(spec.width, spec.height) // 2
    radius = min(radius, max_r)
    color_bgr = _parse_bgr(section.get("color_bgr", fallback="0, 255, 0"), (0, 255, 0))

    interval_sec = max(0.0, section.getfloat("interval_sec", fallback=2.0))
    display_sec = max(0.0, section.getfloat("display_sec", fallback=0.5))
    repeat_count = max(1, section.getint("repeat_count", fallback=6))

    seed_raw = section.get("random_seed", fallback="").strip()
    rng = random.Random(int(seed_raw)) if seed_raw else random.Random()

    writer = _open_writer(output_path, spec)
    total_frames = 0
    try:
        for _ in range(repeat_count):
            total_frames += _write_blank(writer, spec, interval_sec)
            x = rng.randint(radius, spec.width - radius)
            y = rng.randint(radius, spec.height - radius)
            total_frames += _write_hold(
                writer, spec, (float(x), float(y)), radius, color_bgr, display_sec
            )
        # Ensure the last circle disappears cleanly
        writer.write(_make_background(spec))
        total_frames += 1
    finally:
        writer.release()

    return total_frames


def generate_take_3(
    section: configparser.SectionProxy, spec: VideoSpec, output_path: Path
) -> int:
    """Take 3: Green or yellow circle (uniform random) flashes at uniform random in-bounds positions."""
    radius = max(1, section.getint("radius", fallback=60))
    max_r = min(spec.width, spec.height) // 2
    radius = min(radius, max_r)
    green_bgr = _parse_bgr(section.get("green_bgr", fallback="0, 255, 0"), (0, 255, 0))
    yellow_bgr = _parse_bgr(section.get("yellow_bgr", fallback="0, 255, 255"), (0, 255, 255))
    colors = (green_bgr, yellow_bgr)

    interval_sec = max(0.0, section.getfloat("interval_sec", fallback=2.0))
    display_sec = max(0.0, section.getfloat("display_sec", fallback=0.5))
    repeat_count = max(1, section.getint("repeat_count", fallback=10))

    seed_raw = section.get("random_seed", fallback="").strip()
    rng = random.Random(int(seed_raw)) if seed_raw else random.Random()

    writer = _open_writer(output_path, spec)
    total_frames = 0
    try:
        for _ in range(repeat_count):
            total_frames += _write_blank(writer, spec, interval_sec)
            x = rng.uniform(radius, spec.width - radius)
            y = rng.uniform(radius, spec.height - radius)
            color_bgr = rng.choice(colors)
            total_frames += _write_hold(
                writer, spec, (x, y), radius, color_bgr, display_sec
            )
        # Ensure the last circle disappears cleanly
        writer.write(_make_background(spec))
        total_frames += 1
    finally:
        writer.release()

    return total_frames


TAKE_GENERATORS: dict[str, Callable[[configparser.SectionProxy, VideoSpec, Path], int]] = {
    "take_1": generate_take_1,
    "take_2": generate_take_2,
    "take_3": generate_take_3,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate scenario videos from scenario.cfg")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "scenario.cfg",
        help="Path to scenario.cfg",
    )
    ap.add_argument(
        "--take",
        action="append",
        help="Specific take(s) to generate (e.g. --take take_1). Defaults to all enabled takes.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(cfg_path, encoding="utf-8")

    spec = _load_video_spec(parser, cfg_path.parent)
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    target_takes = args.take if args.take else list(TAKE_GENERATORS.keys())
    for take_name in target_takes:
        if take_name not in TAKE_GENERATORS:
            raise ValueError(
                f"Unknown take '{take_name}'. Available: {', '.join(TAKE_GENERATORS)}"
            )
        if not parser.has_section(take_name):
            print(f"[SKIP] Section [{take_name}] not found in {cfg_path.name}")
            continue
        section = parser[take_name]
        if not args.take and not section.getboolean("enabled", fallback=True):
            print(f"[SKIP] {take_name} is disabled")
            continue

        output_path = spec.output_dir / f"{take_name}.{spec.fmt}"
        frames = TAKE_GENERATORS[take_name](section, spec, output_path)
        duration = frames / spec.fps if spec.fps > 0 else 0.0
        print(
            f"[OK] Generated {output_path} "
            f"({spec.width}x{spec.height}, {spec.fps}fps, {frames} frames, {duration:.2f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
