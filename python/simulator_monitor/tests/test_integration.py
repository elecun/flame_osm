"""
End-to-end integration test for Operator & Subject Windows.
"""

import os
import shutil
import time
import tempfile
import unittest
from PyQt6.QtWidgets import QApplication

from src.config import load_config
from src.gui.subject_window import SubjectWindow
from src.gui.operator_window import OperatorWindow


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication.instance() is None:
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sim_integ_")
        self.operator_win = None
        self.subject_win = None

    def tearDown(self):
        if self.operator_win is not None:
            self.operator_win.close()
        if self.subject_win is not None:
            self.subject_win.close()
        self.app.processEvents()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _process_qt_events(self, duration_sec: float = 0.5):
        start = time.time()
        while time.time() - start < duration_sec:
            self.app.processEvents()
            time.sleep(0.02)

    def test_full_experiment_session(self):
        config = load_config("exp.cfg")
        config.output_root_dir = self.test_dir
        config.camera.camera_ids = [0, 1, 2, 3]
        config.daq.mock_mode = "true"
        config.eyetracker.auto_discover = False  # Avoid network scan delay in test

        self.subject_win = SubjectWindow(config.subject_display)
        self.operator_win = OperatorWindow(config, self.subject_win)

        # Verify initial UI states
        self.assertTrue(self.operator_win.btn_start.isEnabled())
        self.assertFalse(self.operator_win.btn_stop.isEnabled())
        self.assertEqual(self.operator_win.label_rec_state.text(), "IDLE")
        self.assertGreater(self.operator_win.combo_scenario.count(), 0)

        # Set subject name
        self.operator_win.edit_subject_name.setText("test_volunteer")

        # 1. Start Experiment
        self.operator_win.start_experiment()
        self.assertTrue(self.operator_win.is_recording)
        # Check mutual exclusion of Start/Stop buttons and RECORDING status
        self.assertFalse(self.operator_win.btn_start.isEnabled())
        self.assertTrue(self.operator_win.btn_stop.isEnabled())
        self.assertEqual(self.operator_win.label_rec_state.text(), "RECORDING")

        session_dir = self.operator_win.current_session_dir
        self.assertTrue(os.path.exists(session_dir))
        self.assertIn("test_volunteer", session_dir)

        # 2. Trigger Switch Presses with event processing (also tests sound trigger)
        self._process_qt_events(0.1)
        self.operator_win.simulate_switch_press()
        self._process_qt_events(0.3)
        self.operator_win.simulate_switch_press()
        self._process_qt_events(0.4)

        # 3. Stop Experiment
        self.operator_win.stop_experiment()
        self.assertFalse(self.operator_win.is_recording)
        self.assertTrue(self.operator_win.btn_start.isEnabled())
        self.assertFalse(self.operator_win.btn_stop.isEnabled())
        self.assertEqual(self.operator_win.label_rec_state.text(), "IDLE")
        self._process_qt_events(0.1)

        # 4. Verify Files in Session Directory
        csv_path = os.path.join(session_dir, "handle_switch.csv")
        self.assertTrue(os.path.exists(csv_path), "handle_switch.csv must exist")

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split(",") for line in f.readlines()]

        self.assertGreaterEqual(len(lines), 2, "CSV should contain header + at least 1 switch event")
        self.assertEqual(lines[0], ["timestamp", "channel", "event"])
        # First column must match millisecond timestamp format
        self.assertRegex(lines[1][0], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$")

        # Verify cam_0.avi, cam_1.avi, cam_2.avi, cam_3.avi exist
        for cam_id in [0, 1, 2, 3]:
            cam_file = os.path.join(session_dir, f"cam_{cam_id}.avi")
            self.assertTrue(os.path.exists(cam_file), f"{cam_file} must exist")
            self.assertGreater(os.path.getsize(cam_file), 0, f"{cam_file} must not be empty")


if __name__ == "__main__":
    unittest.main()
