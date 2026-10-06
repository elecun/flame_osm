#!/usr/bin/env python3
"""
Simulator Monitor Application Entry Point.
Runs operator control interface and subject visual stimulus window.
Usage:
    python run.py --config exp.cfg
"""

import sys
import os
import signal
import argparse

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from src.config import load_config
from src.gui.operator_window import OperatorWindow
from src.gui.subject_window import SubjectWindow


def parse_args():
    parser = argparse.ArgumentParser(description="Simulator Monitor & Visual Stimulus System")
    parser.add_argument(
        "--config",
        type=str,
        default="exp.cfg",
        help="Path to configuration file (.cfg). Default: exp.cfg",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine config path relative to script directory if not absolute
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    print(f"[Main] Loading configuration from: {config_path}")
    app_config = load_config(config_path)

    # Initialize Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("Simulator Monitor")

    # Enable graceful termination on terminal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Subject Presentation Window
    subject_win = SubjectWindow(app_config.subject_display)

    # Operator Monitoring & Control Window
    operator_win = OperatorWindow(app_config, subject_win)
    operator_win.show()

    # Bring operator window to front initially if not in full screen mode
    if not app_config.subject_display.fullscreen:
        operator_win.raise_()
        operator_win.activateWindow()

    # Heartbeat timer for Python signal handling in Qt event loop
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(500)

    # Run Qt Event Loop
    exit_code = app.exec()
    print(f"[Main] Application terminated with code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
