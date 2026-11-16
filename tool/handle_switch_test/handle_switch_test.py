#!/usr/bin/env python3
"""Monitor IDI0 through IDI15 on an Advantech USB-4750 in Ubuntu.

Install the Advantech DAQNavi Linux driver before running this program. Its
Python API is installed below /opt/advantech/examples/Python by default.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence


CHANNEL_COUNT = 16
PORT_COUNT = 2  # USB-4750 exposes the first 16 isolated inputs as two ports.
DAQNAVI_PYTHON_PATH = Path("/opt/advantech/examples/Python")


def states_from_ports(port_values: Sequence[int]) -> tuple[int, ...]:
    """Return IDI0..IDI15 as bits, with IDI0 being bit 0 of port 0."""
    if len(port_values) < PORT_COUNT:
        raise ValueError(f"expected {PORT_COUNT} input ports, got {len(port_values)}")
    return tuple((int(port_values[channel // 8]) >> (channel % 8)) & 1
                 for channel in range(CHANNEL_COUNT))


def format_change(timestamp: datetime, states: Sequence[int]) -> str:
    """Log active-low inputs as 1 when active and 0 when inactive."""
    values = " ".join(f"IDI{channel}={1 - state}"
                      for channel, state in enumerate(states))
    return f"{timestamp.isoformat(timespec='milliseconds')} {values}"


def make_reader(device_description: str):
    """Open DAQNavi's Linux instantaneous-DI controller for the USB-4750."""
    if not DAQNAVI_PYTHON_PATH.is_dir():
        raise RuntimeError(
            "DAQNavi Linux Python API was not found at "
            f"{DAQNAVI_PYTHON_PATH}. Install the USB-4750 DAQNavi driver; "
            "see README.md."
        )

    sys.path.insert(0, str(DAQNAVI_PYTHON_PATH))
    try:
        from Automation.BDaq.BDaqApi import BioFailed  # type: ignore[import-not-found]
        from Automation.BDaq.InstantDiCtrl import InstantDiCtrl  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Cannot load the DAQNavi Python API or libbiodaq.so. "
            "Check the DAQNavi installation and run 'sudo ldconfig'."
        ) from exc

    try:
        controller = InstantDiCtrl(device_description)
    except (TypeError, ValueError, OSError) as exc:
        raise RuntimeError(f"cannot open {device_description}: {exc}") from exc

    def read_states() -> tuple[int, ...]:
        # Read ports 0 and 1: their bits map respectively to IDI0..7 and IDI8..15.
        result, port_values = controller.readAny(0, PORT_COUNT)
        if BioFailed(result):
            code = getattr(result, "value", result)
            raise RuntimeError(f"digital-input read failed (DAQNavi error 0x{int(code):X})")
        return states_from_ports(port_values)

    return controller, read_states


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("usb4750-input")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log USB-4750 IDI0~IDI15 states whenever an input changes."
    )
    parser.add_argument("--device", default="USB-4750,BID#0",
                        help="DAQNavi device description (default: USB-4750,BID#0)")
    parser.add_argument("--interval", type=float, default=0.05,
                        help="polling period in seconds (default: 0.05)")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    logger = configure_logging()
    controller = None
    try:
        controller, read_states = make_reader(args.device)
        previous = read_states()  # Establish a baseline; only subsequent changes are logged.
        logger.info("Monitoring %s IDI0~IDI15 (Ctrl+C to stop).", args.device)

        while True:
            current = read_states()
            if current != previous:
                logger.info(format_change(datetime.now().astimezone(), current))
                previous = current
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
        return 0
    except RuntimeError as exc:
        logger.error("ERROR: %s", exc)
        return 1
    finally:
        if controller is not None:
            controller.dispose()


if __name__ == "__main__":
    sys.exit(main())
