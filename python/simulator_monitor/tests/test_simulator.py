"""
Unit and integration tests for simulator monitor system.
"""

import os
import shutil
import time
import tempfile
import unittest
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication

from src.config import load_config, AppConfig
from src.utils.time_utils import (
    get_formatted_timestamp_ms,
    get_directory_timestamp,
    get_current_epoch_ms,
)
from src.daq.daq_logger import DAQLogger
from src.scenario.scenario_player import ScenarioPlayer
from src.camera.camera_thread import CameraWorker
from src.sound.sound_player import SoundPlayer
from src.eyetracker.neon_worker import NeonTrackerWorker


class TestSimulatorMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication.instance() is None:
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sim_test_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_time_utils(self):
        dir_ts = get_directory_timestamp()
        self.assertEqual(len(dir_ts), 14)
        self.assertTrue(dir_ts.isdigit())

        ts_ms = get_formatted_timestamp_ms()
        self.assertRegex(ts_ms, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$")

        epoch = get_current_epoch_ms()
        self.assertGreater(epoch, 1700000000000)

    def test_config_loading(self):
        cfg = load_config("exp.cfg")
        self.assertIsInstance(cfg, AppConfig)
        self.assertIn(0, cfg.camera.camera_ids)
        self.assertEqual(cfg.subject_display.width, 1920)
        self.assertEqual(cfg.subject_display.height, 1080)
        self.assertEqual(cfg.daq.channel, 0)
        self.assertTrue(cfg.sound.enabled)
        self.assertEqual(cfg.sound.sound_file, "assets/beep.wav")

    def test_daq_logger(self):
        logger = DAQLogger()
        csv_path = logger.start_logging(self.test_dir, "handle_switch.csv")
        self.assertTrue(os.path.exists(csv_path))

        ts1 = get_formatted_timestamp_ms()
        logger.log_event(channel=0, event="pressed", timestamp_str=ts1)
        time.sleep(0.01)
        ts2 = get_formatted_timestamp_ms()
        logger.log_event(channel=0, event="pressed", timestamp_str=ts2)

        logger.stop_logging()

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split(",") for line in f.readlines()]

        self.assertEqual(lines[0], ["timestamp", "channel", "event"])
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1][0], ts1)
        self.assertEqual(lines[1][1], "0")
        self.assertEqual(lines[1][2], "pressed")
        self.assertEqual(lines[2][0], ts2)

    def test_scenario_player(self):
        player = ScenarioPlayer(width=640, height=480)
        loaded = player.load_scenario("scenario/default.scenario")
        self.assertTrue(loaded)
        self.assertGreater(len(player.items), 0)

        # Before start: pure black screen
        frame = player.get_frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(int(frame.sum()), 0)

        # After start: renders stimulus
        player.start()
        frame_started = player.get_frame()
        self.assertIsInstance(frame_started, np.ndarray)

    def test_camera_worker_and_no_camera_frame(self):
        cam = CameraWorker(
            camera_id=99,
            fps=15,
            width=320,
            height=240,
            codec="MJPG",
            mock_if_missing=True,
        )
        cam.init_capture()

        # Disconnected camera generates 'no camera' frame
        no_cam_frame = cam._generate_no_camera_frame()
        self.assertEqual(no_cam_frame.shape, (240, 320, 3))

        # Start recording
        rec_dir = os.path.join(self.test_dir, "rec")
        cam.start_recording(rec_dir)
        self.assertTrue(cam.is_recording)

        # Write 5 raw frames
        for _ in range(5):
            f = cam._generate_no_camera_frame()
            cam._writer.write(f)

        cam.stop_recording()
        self.assertFalse(cam.is_recording)

        video_path = os.path.join(rec_dir, "cam_99.avi")
        self.assertTrue(os.path.exists(video_path))
        self.assertGreater(os.path.getsize(video_path), 0)

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened())
        ret, read_frame = cap.read()
        self.assertTrue(ret)
        self.assertEqual(read_frame.shape, (240, 320, 3))
        cap.release()

    def test_sound_player(self):
        player = SoundPlayer(sound_file_path="assets/beep.wav", enabled=True)
        self.assertTrue(player.enabled)
        # Should not raise exception
        player.play()
        player.set_enabled(False)
        self.assertFalse(player.enabled)
        player.play()

    def test_neon_tracker_worker(self):
        # Test worker with non-existent device (timeout 0.2s)
        worker = NeonTrackerWorker(auto_discover=False, device_address="127.0.0.1", search_timeout_sec=0.2)
        self.assertFalse(worker.is_connected)
        # Safe recording calls when disconnected
        self.assertFalse(worker.start_recording())
        self.assertFalse(worker.stop_recording())


if __name__ == "__main__":
    unittest.main()
