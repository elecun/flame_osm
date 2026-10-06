"""
DAQ Handle Switch CSV logger.
Records timestamps with millisecond precision whenever switch input is received.
"""

import os
import csv
from typing import Optional
from src.utils.time_utils import get_formatted_timestamp_ms


class DAQLogger:
    """
    Handles writing switch trigger events to handle_switch.csv.
    """

    def __init__(self):
        self._file = None
        self._writer = None
        self._is_active = False
        self._csv_path: Optional[str] = None

    def start_logging(self, output_dir: str, file_name: str = "handle_switch.csv") -> str:
        """
        Creates and opens handle_switch.csv in the specified directory.
        """
        self.stop_logging()

        os.makedirs(output_dir, exist_ok=True)
        self._csv_path = os.path.join(output_dir, file_name)

        # Open file with utf-8 encoding and newline='' for standard CSV
        self._file = open(self._csv_path, mode="w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        # First column MUST be timestamp with millisecond precision
        self._writer.writerow(["timestamp", "channel", "event"])
        self._file.flush()

        self._is_active = True
        return self._csv_path

    def log_event(self, channel: int = 0, event: str = "pressed", timestamp_str: Optional[str] = None):
        """
        Logs a switch input event to handle_switch.csv immediately.
        """
        if not self._is_active or self._file is None or self._writer is None:
            return

        if timestamp_str is None:
            timestamp_str = get_formatted_timestamp_ms()

        try:
            self._writer.writerow([timestamp_str, channel, event])
            self._file.flush()
        except Exception as e:
            print(f"[DAQLogger] Error writing event: {e}")

    def stop_logging(self):
        """
        Flushes and closes the CSV file safely.
        """
        self._is_active = False
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception as e:
                print(f"[DAQLogger] Error closing file: {e}")
            finally:
                self._file = None
                self._writer = None

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def csv_path(self) -> Optional[str]:
        return self._csv_path
