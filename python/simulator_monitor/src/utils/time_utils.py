"""
Time and timestamp utilities for simulator monitor.
Ensures uniform millisecond precision across all logs and directories.
"""

from datetime import datetime
import time


def get_formatted_timestamp_ms(dt: datetime | None = None) -> str:
    """
    Returns timestamp string formatted as 'YYYY-MM-DD HH:MM:SS.mmm'.
    Example: 2026-10-06 23:15:30.123
    """
    if dt is None:
        dt = datetime.now()
    # Microsecond truncated to 3 digits (milliseconds)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{dt.microsecond // 1000:03d}"


def get_directory_timestamp(dt: datetime | None = None) -> str:
    """
    Returns timestamp string for directory creation in 'yyyymmddHHMMSS' format.
    Example: 20261006231530
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d%H%M%S")


def get_current_epoch_ms() -> int:
    """
    Returns current epoch time in milliseconds.
    """
    return int(time.time() * 1000)
