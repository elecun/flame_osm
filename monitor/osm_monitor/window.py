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
        """ initialization """
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger() # logger
        self.__config = config  # copy configuration data
        self.__pipeline_context = zmq.Context(14) # zmq context

        self.__frame_defect_grid_layout = QVBoxLayout()
        self.__frame_defect_grid_plot = graph.PlotWidget()

        # device/service control interfaces
        self.__camera_image_subscriber_map = {}

        # variables
        self.__total_frames = 0

        try:            
            if "gui" in config:

                # load UI File
                ui_path = pathlib.Path(config["app_path"]) / config["gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                else:
                    raise Exception(f"Cannot found UI file : {ui_path}")
                
                # defect graphic view frame
                self.__frame_defect_grid_frame = self.findChild(QFrame, name="frame_defect_grid_frame")
                self.__frame_defect_grid_layout.addWidget(self.__frame_defect_grid_plot)
                self.__frame_defect_grid_layout.setContentsMargins(0, 0, 0, 0)
                self.__frame_defect_grid_plot.setBackground('w')
                self.__frame_defect_grid_plot.showGrid(x=True, y=True)
                self.__frame_defect_grid_plot.setLimits(xMin=0, xMax=10000, yMin=0, yMax=11)
                self.__frame_defect_grid_plot.setRange(yRange=(0,10), xRange=(0,100))
                self.__frame_defect_grid_plot.setMouseEnabled(x=True, y=False)
                self.__frame_defect_grid_frame.setLayout(self.__frame_defect_grid_layout)

                # grid plot style
                styles = {"color": "#000", "font-size": "15px"}
                self.__frame_defect_grid_plot.setLabel("left", "Camera Channels", **styles)
                self.__frame_defect_grid_plot.setLabel("bottom", "Frame Counts", **styles)
                self.__frame_defect_grid_plot.addLegend()


                # find focus preset files in preset directory
                #preset_path = pathlib.Path(config["app_path"])/pathlib.Path(config["preset_path"])
                self.__config["preset_path"] = pathlib.Path(config["preset_path"]).as_posix()
                #self.__config["preset_path"] = preset_path.as_posix()
                #self.__console.info(f"+ Preset Path : {config["preset_path"]}")
                if os.path.exists(pathlib.Path(self.__config["preset_path"])):
                    preset_files = [f for f in os.listdir(self.__config["preset_path"])]
                    for preset in preset_files:
                        self.combobox_focus_preset.addItem(preset)

                # map between camera device and windows
                self.__frame_window_map = {}

                for idx, id in enumerate(config["camera_ids"]):
                    self.__frame_window_map[id] = self.findChild(QLabel, config["camera_windows"][idx])
                    self.__console.info(f"Ready for camera grabber #{id} monitoring")
                    portname = f"image_stream_monitor_source_{id}"
                    self.__console.info("+ Create Camera #{id} Monitoring Subscriber...")
                    self.__camera_image_subscriber_map[id] = CameraMonitorSubscriber(self.__pipeline_context,connection=config[portname],
                                                                                     topic=f"{config['image_stream_monitor_topic_prefix']}{id}")
                    self.__camera_image_subscriber_map[id].frame_update_signal.connect(self.on_update_camera_image)
                    self.__camera_image_subscriber_map[id].start()

        except Exception as e:
            self.__console.error(f"{e}")

    def clear_all(self):
        """ clear graphic view """
        try:
            self.__frame_defect_grid_plot.clear()
        except Exception as e:
            self.__console.error(f"{e}")


    def on_update_camera_image(self, camera_id:int, image:np.ndarray):
        """ show image on window for each camera id """
        h, w, ch = image.shape
        check = self.findChild(QCheckBox, "chk_show_alignment_line")
        if check and check.isChecked():
            cx = w//2
            cy = h//2
            cv2.line(image, (cx, 0), (cx, h), (0, 255, 0), 1) #(960, 0) (960, 1920)
            cv2.line(image, (0, cy), (w, cy), (0, 255, 0), 1) # 

        qt_image = QImage(image.data, w, h, ch*w, QImage.Format.Format_RGB888)
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
