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
#import pyqtgraph as graph
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
    from PyQt5.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QMessageBox, QTableWidget, QTableWidgetItem, QGroupBox, QCheckBox
    from PyQt5.QtWidgets import QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider
    from PyQt5.uic import loadUi
    from PyQt5.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
except ImportError:
    # using PyQt6
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QCheckBox, QComboBox
    from PyQt6.QtWidgets import QMessageBox, QProgressBar, QFileDialog, QLineEdit, QSlider, QVBoxLayout, QTableWidget, QTableWidgetItem, QGroupBox
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
    
from util.logger.console import ConsoleLogger
from subscriber.camera import CameraMonitorSubscriber
from subscriber.video import VideoImageStreamSubscriber

class AppWindow(QMainWindow):
    def __init__(self, config:dict):
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger() # logger
        self.__config = config  # copy configuration data

        n_ctx_value = config.get("n_io_context", 10)
        self.__pipeline_context = zmq.Context(n_ctx_value) # zmq context

        # option flags
        self.__show_frame_info = False

        # recording state
        self.__is_recording = False
        self.__record_start_time = ""
        self.__video_writers = {}

        # device/service control interfaces
        self.__camera_image_subscriber_map = {}

        # module instance
        self.__video_image_subscriber = None
        self.__can_subscriber = None

        try:            
            if "gui" in config:

                # load UI File
                ui_path = pathlib.Path(config["app_path"]) / config["gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                else:
                    raise Exception(f"Cannot found UI file : {ui_path}")
                
                # Setup CAN monitoring
                self.load_can_signals()
                self.init_can_table()
                
                can_conn = config.get("can_monitor_port", "tcp://192.168.100.91:5101")
                can_topic = config.get("can_monitor_topic", "status")
                can_sim = config.get("can_simulate", True)
                
                from subscriber.can import CANMonitorSubscriber
                self.__can_subscriber = CANMonitorSubscriber(self.__pipeline_context, connection=can_conn, topic=can_topic, simulate=can_sim)
                self.__can_subscriber.can_message_received.connect(self.on_update_can_message)
                self.__can_subscriber.start()
                
                # ui components event callback def.
                if hasattr(self, 'chk_dms_enable'):
                    self.chk_dms_enable.stateChanged.connect(self.on_check_dms_enable)
                if hasattr(self, 'chk_option_show_frame_info'):
                    self.chk_option_show_frame_info.stateChanged.connect(self.on_check_option_show_frame_info)
                if hasattr(self, 'chk_option_show_body_keypoints'):
                    self.chk_option_show_body_keypoints.stateChanged.connect(self.on_check_show_body_keypoints)
                elif hasattr(self, 'chk_show_body_kps'):
                    self.chk_show_body_kps.stateChanged.connect(self.on_check_show_body_keypoints)
                if hasattr(self, 'btn_video_open'):
                    self.btn_video_open.clicked.connect(self.on_btn_video_open)

                if hasattr(self, 'btn_record_start'):
                    self.btn_record_start.clicked.connect(self.on_btn_record_start)
                if hasattr(self, 'btn_record_stop'):
                    self.btn_record_stop.clicked.connect(self.on_btn_record_stop)
                    self.btn_record_stop.setEnabled(False)

                # set options
                if self.__config.get("option_show_frame_info", False):
                    if hasattr(self, 'chk_option_show_frame_info'):
                        self.chk_option_show_frame_info.setChecked(True)
                    self.on_check_option_show_frame_info()

                # map between camera device and windows
                self.__frame_window_map = {}

                for idx, id in enumerate(config["camera_ids"]):
                    self.__frame_window_map[id] = self.findChild(QLabel, config["camera_windows"][idx])
                    portname = f"image_stream_{id}_monitor"
                    self.__camera_image_subscriber_map[id] = CameraMonitorSubscriber(self.__pipeline_context,connection=config[portname],
                                                                                     topic=config[f'image_stream_{id}_monitor_topic'])
                    self.__camera_image_subscriber_map[id].frame_update_signal.connect(self.on_update_camera_image)
                    self.__camera_image_subscriber_map[id].start()

        except Exception as e:
            self.__console.error(f"{e}")

    def clear_all(self):
        """ clear graphic view """
        try:
            pass
        except Exception as e:
            self.__console.error(f"{e}")

    ## component event callbacks
    def on_btn_video_open(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Video File", "", "AVI Files (*.avi);;All Files (*)")
        if video_file:
            widget = self.findChild(QLabel, name=f"label_video_filepath")
            if widget:
                widget.setText(f"{video_file}")
        else:
            self.__console.warning("No Markers file selected.")

    def on_update_camera_image(self, image:np.ndarray, tags:dict):

        id = tags.get("id", tags.get("camera_id", tags.get("cam_id", tags.get("cam_channel", 1))))
        
        # Robust ID lookup: try type conversions to match key types in self.__frame_window_map
        if id not in self.__frame_window_map and self.__frame_window_map:
            try:
                int_id = int(id)
                if int_id in self.__frame_window_map:
                    id = int_id
            except (ValueError, TypeError):
                pass
        if id not in self.__frame_window_map and self.__frame_window_map:
            try:
                str_id = str(id)
                if str_id in self.__frame_window_map:
                    id = str_id
            except (ValueError, TypeError):
                pass
                
        if id not in self.__frame_window_map and self.__frame_window_map:
            id = list(self.__frame_window_map.keys())[0]
        fps = round(tags.get("fps", 0.0), 1)

        print(tags)

        # Rotate received image 90 degrees clockwise
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        color_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
        if self.__show_frame_info:
            t = datetime.now()
            cv2.putText(color_image, t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
        h, w, ch = color_image.shape
        
        # Write to video if recording is active
        if self.__is_recording:
            if id not in self.__video_writers:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer_fps = fps if fps > 0 else 30.0
                filename = f"{self.__record_start_time}_cam{id}.avi"
                self.__video_writers[id] = cv2.VideoWriter(filename, fourcc, writer_fps, (w, h))
                self.__console.info(f"Started video recording for Camera {id} to {filename}")
            
            try:
                self.__video_writers[id].write(rotated_image)
            except Exception as e:
                self.__console.error(f"Error writing frame to Camera {id} video: {e}")

        qt_image = QImage(color_image.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        try:
            self.__frame_window_map[id].setPixmap(pixmap.scaled(self.__frame_window_map[id].size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.__frame_window_map[id].show()
        except Exception as e:
            self.__console.error(e)
    
    def on_update_video_image(self, image:np.ndarray, tags:dict):
        """ video image stream subscribe """
        print(tags)
    
                
    def closeEvent(self, event:QCloseEvent) -> None: 
        """ terminate main window """      

        # close video writers
        if hasattr(self, '__video_writers'):
            for id, writer in self.__video_writers.items():
                writer.release()
            self.__video_writers.clear()

        # close CAN subscriber
        if self.__can_subscriber:
            self.__can_subscriber.close()

        # close camera stream monitoring subscriber
        if len(self.__camera_image_subscriber_map.keys())>0:
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(lambda subscriber: subscriber.close(), self.__camera_image_subscriber_map.values())

        # close video stream subscriber
        if self.__video_image_subscriber:
            self.__video_image_subscriber.close()

        # context termination with linger=0
        self.__pipeline_context.destroy(0)
            
        return super().closeEvent(event)
    
    def on_btn_record_start(self):
        if not self.__is_recording:
            self.__is_recording = True
            self.__record_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.__video_writers = {}
            self.__console.info(f"Video recording started at {self.__record_start_time}")
            if hasattr(self, 'btn_record_start'):
                self.btn_record_start.setEnabled(False)
            if hasattr(self, 'btn_record_stop'):
                self.btn_record_stop.setEnabled(True)
            if hasattr(self, 'label_record_status'):
                self.label_record_status.setText("RECORDING")
                self.label_record_status.setStyleSheet("color: #ff5c5c; font-weight: bold;")

    def on_btn_record_stop(self):
        if self.__is_recording:
            self.__is_recording = False
            for id, writer in self.__video_writers.items():
                writer.release()
            self.__video_writers.clear()
            self.__console.info("Video recording stopped.")
            if hasattr(self, 'btn_record_start'):
                self.btn_record_start.setEnabled(True)
            if hasattr(self, 'btn_record_stop'):
                self.btn_record_stop.setEnabled(False)
            if hasattr(self, 'label_record_status'):
                self.label_record_status.setText("IDLE")
                self.label_record_status.setStyleSheet("color: #8e8e93; font-weight: bold;")

    def on_check_option_disable_camera_monitor_stream(self):
        pass

    def on_check_option_show_frame_info(self):
        if hasattr(self, 'chk_option_show_frame_info'):
            self.__show_frame_info = self.chk_option_show_frame_info.isChecked()
        else:
            self.__show_frame_info = self.__config.get("option_show_frame_info", False)

    def on_check_show_body_keypoints(self):
        pass

    def load_can_signals(self):
        self.__can_signals_isc = []
        self.__can_signals_dms = []
        try:
            import pandas as pd
            excel_path = os.path.join(self.__config.get("root_path", ""), "can_protocol.xlsx")
            if os.path.isfile(excel_path):
                self.__console.info(f"Loading CAN protocol from {excel_path}...")
                xl = pd.ExcelFile(excel_path)
                df = xl.parse("4. CAN Matrix")
                headers = df.iloc[2].tolist()
                clean_df = df.iloc[3:].copy()
                clean_df.columns = headers
                
                # Load ISC_01_10ms
                isc_signals = clean_df[clean_df["Message Name"] == "ISC_01_10ms"]
                for _, row in isc_signals.iterrows():
                    self.__can_signals_isc.append({
                        "signal_name": str(row["Signal Name"]).strip(),
                        "start_bit": str(row["Start Bit"]),
                        "len_bit": str(row["Len(bit)"]),
                        "value_table_enum": str(row["Value Table / Enum"]),
                        "init_value": str(int(float(row["Init Value"]))) if pd.notna(row["Init Value"]) else "0"
                    })
                    
                # Load STS_DMS_1000ms
                dms_signals = clean_df[clean_df["Message Name"] == "STS_DMS_1000ms"]
                for _, row in dms_signals.iterrows():
                    self.__can_signals_dms.append({
                        "signal_name": str(row["Signal Name"]).strip(),
                        "start_bit": str(row["Start Bit"]),
                        "len_bit": str(row["Len(bit)"]),
                        "value_table_enum": str(row["Value Table / Enum"]),
                        "init_value": str(int(float(row["Init Value"]))) if pd.notna(row["Init Value"]) else "0"
                    })
        except Exception as e:
            self.__console.warning(f"Failed to load CAN protocol from Excel: {e}. Using fallback signals.")
            
        if not self.__can_signals_isc:
            # fallback signals from can_protocol.xlsx for ISC
            fallback_isc = [
                {"signal_name": "MasterStatus", "start_bit": "0", "len_bit": "2", "init_value": "0", "value_table_enum": "0x0=Init \n0x1=Normal\n0x2=Error\n0x3=Reserved"},
                {"signal_name": "ISC_OperatingCASE", "start_bit": "2", "len_bit": "4", "init_value": "0", "value_table_enum": "0x0=CASE0\n0x1=CASE1\n0x2=CASE1_1\n0x3=CASE1_2\n0x4=CASE1_3\n0x5=CASE2\n0x6=CASE3\n0x7=CASE4\n0x8 ~ 0xF = Reserved"},
                {"signal_name": "FaultClearReq", "start_bit": "7", "len_bit": "1", "init_value": "1", "value_table_enum": "0x0=Normal\n0x1=Clear (pulse: 1 frame SET → auto CLEAR)"},
                {"signal_name": "ISC_DMS_Enable", "start_bit": "8", "len_bit": "1", "init_value": "1", "value_table_enum": "0x0=Disable\n0x1=Enable"},
                {"signal_name": "ISC_DMS_State", "start_bit": "9", "len_bit": "2", "init_value": "0", "value_table_enum": "0x0=Init\n0x1=Inactive\n0x2=Active\n0x3=Fault"},
                {"signal_name": "ISC_DMS_DriverPresent", "start_bit": "11", "len_bit": "2", "init_value": "0", "value_table_enum": "0x0=Unknown\n0x1=High\n0x2=Moderate\n0x3=Low"},
                {"signal_name": "ISC_Ridar1_Passenger1_Status", "start_bit": "16", "len_bit": "1", "init_value": "0", "value_table_enum": "0x0=Inactive\n0x1=Active"},
                {"signal_name": "ISC_Ridar1_Passenger2_Status", "start_bit": "17", "len_bit": "1", "init_value": "0", "value_table_enum": "0x0=Inactive\n0x1=Active"},
                {"signal_name": "ISC_Ridar1_FaultStatus", "start_bit": "18", "len_bit": "1", "init_value": "0", "value_table_enum": "0x0=Normal\n0x1=Error"},
                {"signal_name": "ISC_Ridar2_Passenger3_Status", "start_bit": "19", "len_bit": "1", "init_value": "0", "value_table_enum": "0x0=Inactive\n0x1=Active"},
                {"signal_name": "ISC_Ridar2_Passenger4_Status", "start_bit": "20", "len_bit": "1", "init_value": "0", "value_table_enum": "0x0=Inactive\n0x1=Active"},
                {"signal_name": "ISC_Ridar2_FaultStatus", "start_bit": "21", "len_bit": "1", "init_value": "0", "value_table_enum": "0x0=Normal\n0x1=Error"},
                {"signal_name": "ISC_STR_FoldState", "start_bit": "24", "len_bit": "3", "init_value": "0", "value_table_enum": "0x0=Folded\n0x1=Unfolded\n0x2=Folding\n0x3=Unfolding\n0x4=Error\n0x5 ~ 0x7=Reserved"},
                {"signal_name": "ISC_Seat_ModeFeedback", "start_bit": "32", "len_bit": "3", "init_value": "0", "value_table_enum": "0x0=Normal\n0x1=Drive\n0x2=InOut\n0x3=Conversation\n0x4=Meet\n0x5=Relax\n0x6=Done\n0x7=Reserved"},
                {"signal_name": "ISC_Seat_MotorMoving", "start_bit": "35", "len_bit": "1", "init_value": "0", "value_table_enum": "0x0=Idle\n0x1=Moving"},
                {"signal_name": "ISC_Seat_FaultFlag", "start_bit": "36", "len_bit": "2", "init_value": "0", "value_table_enum": "0x0=Normal\n0x1=DRV_FAULT\n0x2=PAS_FAULT\n0x3=Reserved"},
                {"signal_name": "ISC_MoodLampIDFeedback", "start_bit": "40", "len_bit": "5", "init_value": "0", "value_table_enum": "0x0=None \n0x1=Cockpit \n0x2=Doortrim\n0x3=Reserved \n0x4=Roof\n0x5=Reserved \n0x6=Reserved \n0x7=Reserved \n0x8=A_Pillar\n0x9~0xF=Reserved\n0x10=Seatback\n0x11~0x1F=Reserved"},
                {"signal_name": "ISC_MoodLampState", "start_bit": "45", "len_bit": "4", "init_value": "0", "value_table_enum": "0x0=None\n0x1=Work\n0x2=Relax\n0x3=Media\n0x4=Meal\n0x5=Conversation\n0x6=Fear_Surprise\n0x7=Sadness\n0x8=Caution\n0x9=Warning\n0xA~0xF=Reserved"},
                {"signal_name": "ISC_TC_DisplayState", "start_bit": "56", "len_bit": "2", "init_value": "0", "value_table_enum": "0x0=Off\n0x1=On\n0x2=Standby\n0x3=Reserved"},
                {"signal_name": "ISC_TC_FaultStatus", "start_bit": "58", "len_bit": "1", "init_value": "0", "value_table_enum": "0x0=Normal\n0x1=Error"}
            ]
            for sig in fallback_isc:
                self.__can_signals_isc.append({
                    "signal_name": sig["signal_name"],
                    "start_bit": sig["start_bit"],
                    "len_bit": sig["len_bit"],
                    "init_value": sig["init_value"],
                    "value_table_enum": sig["value_table_enum"]
                })
                
        if not self.__can_signals_dms:
            # fallback signals for STS_DMS_1000ms
            fallback_dms = [
                {"signal_name": "DMS_State", "start_bit": "0", "len_bit": "2", "init_value": "0", "value_table_enum": "0x0=Init\n0x1=Inactive\n0x2=Active\n0x3=Fault"},
                {"signal_name": "DMS_DriverPresent", "start_bit": "2", "len_bit": "2", "init_value": "0", "value_table_enum": "0x0=Unknown\n0x1=High\n0x2=Moderate\n0x3=Low"}
            ]
            for sig in fallback_dms:
                self.__can_signals_dms.append({
                    "signal_name": sig["signal_name"],
                    "start_bit": sig["start_bit"],
                    "len_bit": sig["len_bit"],
                    "init_value": sig["init_value"],
                    "value_table_enum": sig["value_table_enum"]
                })
                
        # Parse enums for each signal
        for sig in self.__can_signals_isc:
            sig["enum_map"] = self.parse_value_table(sig["value_table_enum"])
        for sig in self.__can_signals_dms:
            sig["enum_map"] = self.parse_value_table(sig["value_table_enum"])

    def parse_value_table(self, enum_str):
        if not isinstance(enum_str, str) or not enum_str.strip():
            return {}
        enum_map = {}
        lines = enum_str.replace('\r', '').split('\n')
        for line in lines:
            line = line.strip()
            if not line or '=' not in line:
                continue
            parts = line.split('=', 1)
            val_part = parts[0].strip()
            name_part = parts[1].strip()
            
            if '~' in val_part or '-' in val_part:
                sep = '~' if '~' in val_part else '-'
                range_parts = val_part.split(sep)
                try:
                    start_val = int(range_parts[0].strip(), 16) if range_parts[0].strip().lower().startswith('0x') else int(range_parts[0].strip())
                    end_val = int(range_parts[1].strip(), 16) if range_parts[1].strip().lower().startswith('0x') else int(range_parts[1].strip())
                    for v in range(start_val, end_val + 1):
                        enum_map[v] = name_part
                except ValueError:
                    pass
            else:
                try:
                    val = int(val_part, 16) if val_part.lower().startswith('0x') else int(val_part)
                    enum_map[val] = name_part
                except ValueError:
                    try:
                        val = int(val_part)
                        enum_map[val] = name_part
                    except ValueError:
                        pass
        return enum_map

    def init_can_table(self):
        # Initialize table 1 (ISC_01_10ms)
        table_isc = self.findChild(QTableWidget, "table_can_signals")
        if table_isc:
            self._populate_table(table_isc, self.__can_signals_isc)
        else:
            self.__console.error("table_can_signals not found in UI layout")
            
        # Initialize table 2 (STS_DMS_1000ms)
        table_dms = self.findChild(QTableWidget, "table_can_signals_dms")
        if table_dms:
            self._populate_table(table_dms, self.__can_signals_dms)
        else:
            self.__console.error("table_can_signals_dms not found in UI layout")

    def _populate_table(self, table, signals):
        table.setRowCount(len(signals))
        editable_flag = Qt.ItemFlag.ItemIsEditable if hasattr(Qt, "ItemFlag") else Qt.ItemIsEditable
        
        for row_idx, signal in enumerate(signals):
            item_name = QTableWidgetItem(signal["signal_name"])
            item_name.setFlags(item_name.flags() & ~editable_flag)
            table.setItem(row_idx, 0, item_name)
            
            init_val_str = signal.get("init_value", "0")
            try:
                init_val = int(float(init_val_str))
            except ValueError:
                init_val = 0
                
            enum_map = signal.get("enum_map", {})
            enum_str = enum_map.get(init_val, f"Unknown ({init_val})")
            if not enum_map:
                enum_str = str(init_val)
                
            item_val = QTableWidgetItem(enum_str)
            item_val.setFlags(item_val.flags() & ~editable_flag)
            
            lower_enum = enum_str.lower()
            if "error" in lower_enum or "fault" in lower_enum:
                item_val.setForeground(Qt.GlobalColor.red if hasattr(Qt, "GlobalColor") else Qt.red)
            elif any(word in lower_enum for word in ["normal", "active", "enable", "unfolded"]):
                item_val.setForeground(Qt.GlobalColor.green if hasattr(Qt, "GlobalColor") else Qt.green)
            elif any(word in lower_enum for word in ["init", "moving", "folding", "unfolding"]):
                item_val.setForeground(Qt.GlobalColor.yellow if hasattr(Qt, "GlobalColor") else Qt.yellow)
            else:
                item_val.setForeground(Qt.GlobalColor.white if hasattr(Qt, "GlobalColor") else Qt.white)
                
            table.setItem(row_idx, 1, item_val)
            
            length = int(signal["len_bit"])
            raw_str = f"0x{init_val:X}" if length > 1 else str(init_val)
            item_raw = QTableWidgetItem(raw_str)
            item_raw.setFlags(item_raw.flags() & ~editable_flag)
            table.setItem(row_idx, 2, item_raw)

    def on_update_can_message(self, msg):
        msg_id = msg.get("id")
        data_bytes = msg.get("data", [])
        if not data_bytes:
            return
            
        if msg_id == 0x100:
            table = self.findChild(QTableWidget, "table_can_signals")
            signals = self.__can_signals_isc
        elif msg_id == 0x220:
            table = self.findChild(QTableWidget, "table_can_signals_dms")
            signals = self.__can_signals_dms
        else:
            return
            
        if not table:
            return
            
        for row_idx, signal in enumerate(signals):
            start = int(signal["start_bit"])
            length = int(signal["len_bit"])
            
            val = 0
            for i in range(length):
                bit_pos = start + i
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                if byte_idx < len(data_bytes):
                    bit = (data_bytes[byte_idx] >> bit_idx) & 1
                    val |= (bit << i)
                    
            enum_map = signal.get("enum_map", {})
            enum_str = enum_map.get(val, f"Unknown ({val})")
            if not enum_map:
                enum_str = str(val)
                
            raw_str = f"0x{val:X}" if length > 1 else str(val)
            
            item_enum = table.item(row_idx, 1)
            if item_enum:
                item_enum.setText(enum_str)
                lower_enum = enum_str.lower()
                if "error" in lower_enum or "fault" in lower_enum:
                    item_enum.setForeground(Qt.GlobalColor.red if hasattr(Qt, "GlobalColor") else Qt.red)
                elif any(word in lower_enum for word in ["normal", "active", "enable", "unfolded"]):
                    item_enum.setForeground(Qt.GlobalColor.green if hasattr(Qt, "GlobalColor") else Qt.green)
                elif any(word in lower_enum for word in ["init", "moving", "folding", "unfolding"]):
                    item_enum.setForeground(Qt.GlobalColor.yellow if hasattr(Qt, "GlobalColor") else Qt.yellow)
                else:
                    item_enum.setForeground(Qt.GlobalColor.white if hasattr(Qt, "GlobalColor") else Qt.white)
                    
            item_raw = table.item(row_idx, 2)
            if item_raw:
                item_raw.setText(raw_str)

    def on_check_dms_enable(self):
        if hasattr(self, 'chk_dms_enable'):
            enabled = self.chk_dms_enable.isChecked()
            self.__console.info(f"DMS Enable state changed: {enabled}")
