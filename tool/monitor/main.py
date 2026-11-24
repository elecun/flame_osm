#!/usr/bin/env python3
"""Entry point for the Multi-Modal Experiment Monitor application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtGui import QGuiApplication, QScreen
from PyQt6.QtWidgets import QApplication

from config import DisplayConfig, load_config
from monitor_window import MonitorWindow
from subject_window import SubjectWindow


def _pick_screens(
    screens: list[QScreen], display_cfg: DisplayConfig
) -> tuple[Optional[QScreen], Optional[QScreen]]:
    """Return (main_screen, subject_screen) based on config or dual-monitor layout."""
    if not screens:
        return None, None
    if len(screens) == 1:
        return screens[0], screens[0]

    def _by_index(spec: str) -> Optional[QScreen]:
        if spec.lower() == "auto":
            return None
        try:
            idx = int(spec)
            if 0 <= idx < len(screens):
                return screens[idx]
        except ValueError:
            pass
        return None

    main_screen = _by_index(display_cfg.main_monitor)
    subject_screen = _by_index(display_cfg.subject_monitor)

    # Auto-detect: left 1920x1080 = subject, right/primary 2560x1440 = main
    sorted_by_x = sorted(screens, key=lambda s: s.geometry().x())
    if subject_screen is None:
        for s in sorted_by_x:
            g = s.geometry()
            if (
                g.width() == display_cfg.subject_width
                and g.height() == display_cfg.subject_height
            ):
                subject_screen = s
                break
        if subject_screen is None:
            subject_screen = sorted_by_x[0]

    if main_screen is None:
        primary = QGuiApplication.primaryScreen()
        if primary is not None and primary is not subject_screen:
            main_screen = primary
        else:
            candidates = [s for s in sorted_by_x if s is not subject_screen]
            main_screen = candidates[-1] if candidates else screens[0]

    return main_screen, subject_screen


def _place_windows(
    monitor_win: MonitorWindow,
    subject_win: SubjectWindow,
    display_cfg: DisplayConfig,
) -> None:
    screens = QGuiApplication.screens()
    main_screen, subject_screen = _pick_screens(screens, display_cfg)

    if main_screen is not None:
        geom = main_screen.availableGeometry()
        monitor_win.adjustSize()
        x = geom.x() + max(0, (geom.width() - monitor_win.width()) // 2)
        y = geom.y() + max(0, (geom.height() - monitor_win.height()) // 2)
        monitor_win.move(x, y)
        monitor_win.show()
    else:
        monitor_win.adjustSize()
        monitor_win.show()

    if subject_screen is not None:
        geom = subject_screen.geometry()
        subject_win.setGeometry(geom)
        if display_cfg.subject_fullscreen and len(screens) > 1:
            subject_win.showFullScreen()
        elif display_cfg.subject_fullscreen:
            subject_win.showFullScreen()
        else:
            subject_win.show()
    else:
        subject_win.show()


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir / "default.cfg")

    app = QApplication(sys.argv)
    app.setApplicationName("Multi-Modal Experiment Monitor")

    subject_ui = base_dir / "ui" / "subject_window.ui"
    monitor_ui = base_dir / "ui" / "monitor_window.ui"

    subject_window = SubjectWindow(subject_ui, config.scenario)
    monitor_window = MonitorWindow(monitor_ui, config, subject_window)

    _place_windows(monitor_window, subject_window, config.display)
    monitor_window.activateWindow()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
