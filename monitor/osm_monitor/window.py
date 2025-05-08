'''
OSM Application Window class
@Author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

import os, sys
import cv2
import pathlib
import threading
import queue
import time
import numpy as np
from datetime import datetime
import pyqtgraph as graph
import random
import zmq
import zmq.asyncio
import json
import cv2
from functools import partial
from concurrent.futures import ThreadPoolExecutor

try:
    # using PyQt5
    from PyQt5.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt5.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QMessageBox
    from PyQt5.QtWidget import QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QCheckBox, QComboBox
    from PyQt5.uic import loadUi
    from PyQt5.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
except ImportError:
    # using PyQt6
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QCheckBox, QComboBox
    from PyQt6.QtWidgets import QMessageBox, QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QVBoxLayout
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
    
from util.logger.console import ConsoleLogger
from subscriber.camera import CameraMonitorSubscriber

class AppWindow(QMainWindow):
    def __init__(self, config:dict):
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger() # logger
        self.__config = config  # copy configuration data

        n_ctx_value = config.get("n_io_context", 10)
        self.__pipeline_context = zmq.Context(n_ctx_value) # zmq context

        # option flags
        self.__show_frame_info = False

        # device/service control interfaces
        self.__camera_image_subscriber_map = {}

        try:            
            if "gui" in config:

                # load UI File
                ui_path = pathlib.Path(config["app_path"]) / config["gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                else:
                    raise Exception(f"Cannot found UI file : {ui_path}")
                
                # ui components event callback def.
                self.chk_option_disable_camera_monitoring_stream.stateChanged.connect(self.on_check_option_disable_camera_monitor_stream)
                self.chk_option_show_frame_info.stateChanged.connect(self.on_check_option_show_frame_info)
                self.chk_option_show_body_keypoints.stateChanged.connect(self.on_check_show_body_keypoints)

                # set options
                if self.__config.get("option_show_frame_info", False):
                    self.chk_option_show_frame_info.setChecked(True)
                    self.on_check_option_show_frame_info()

                # map between camera device and windows
                self.__frame_window_map = {}

                for idx, id in enumerate(config["camera_ids"]):
                    self.__frame_window_map[id] = self.findChild(QLabel, config["camera_windows"][idx])
                    portname = f"image_stream_monitor_source_{id}"
                    self.__camera_image_subscriber_map[id] = CameraMonitorSubscriber(self.__pipeline_context,connection=config[portname],
                                                                                     topic=f"{config['image_stream_monitor_topic_prefix']}{id}")
                    self.__camera_image_subscriber_map[id].frame_update_signal.connect(self.on_update_camera_image)
                    self.__camera_image_subscriber_map[id].start()
                    self.__console.info("** Start Camera #{id} Monitoring Subscriber...")

        except Exception as e:
            self.__console.error(f"{e}")

    def clear_all(self):
        """ clear graphic view """
        try:
            self.__frame_defect_grid_plot.clear()
        except Exception as e:
            self.__console.error(f"{e}")

    def on_update_camera_image(self, image:np.ndarray, tags:dict):

        camera_id = tags["camera_id"]
        fps = round(tags["fps"], 1)

        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.__show_frame_info:
            t = datetime.now()
            cv2.putText(color_image, t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], (5, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.putText(color_image, f"Camera #{camera_id}(fps:{fps})", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1,255,0), 1, cv2.LINE_AA)
        h, w, ch = color_image.shape
        

        qt_image = QImage(color_image.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        try:
            self.__frame_window_map[camera_id].setPixmap(pixmap.scaled(self.__frame_window_map[camera_id].size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.__frame_window_map[camera_id].show()
        except Exception as e:
            self.__console.error(e)
    
                
    def closeEvent(self, event:QCloseEvent) -> None: 
        """ terminate main window """      

        # close camera stream monitoring subscriber
        if len(self.__camera_image_subscriber_map.keys())>0:
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(lambda subscriber: subscriber.close(), self.__camera_image_subscriber_map.values())

        # context termination with linger=0
        self.__pipeline_context.destroy(0)
            
        return super().closeEvent(event)
    
    def on_check_option_disable_camera_monitor_stream(self):
        pass

    def on_check_option_show_frame_info(self):
        self.__show_frame_info = self.chk_option_show_frame_info.isChecked()

    def on_check_show_body_keypoints(self):
        pass
