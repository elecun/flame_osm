"""
Camera Image Monitor subscriber
@author Byunghun Hwang <bh.hwang@iae.re.kr>
"""


try:
    # using PyQt5
    from PyQt5.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
    from PyQt6.QtGui import QImage
except ImportError:
    # using PyQt6
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
    from PyQt6.QtGui import QImage



import cv2
from datetime import datetime
import platform
from util.logger.console import ConsoleLogger
import numpy as np
from typing import Tuple
import csv
import pathlib
import zmq
import zmq.utils.monitor as zmq_monitor
from util.logger.console import ConsoleLogger
import json
import threading
import time
from typing import Any, Dict
from multiprocessing import Process, Queue

# connection event message parsing
EVENT_MAP = {}
for name in dir(zmq):
    if name.startswith('EVENT_'):
        value = getattr(zmq, name)
        EVENT_MAP[value] = name


class CameraMonitorSubscriber(QThread):
    
    frame_update_signal = pyqtSignal(np.ndarray, dict) # camera images update signal : np.ndarray, tag
    status_msg_update_signal = pyqtSignal(str) # connection status message update signal

    def __init__(self, context:zmq.Context, connection:str, topic:str):
        super().__init__()

        self.__console = ConsoleLogger.get_logger()   # console logger
        self.__console.info(f"Camera Monitor Connection : {connection} (topic:{topic})")

        # store parameters
        self.__connection = connection
        self.__topic = topic

        # initialize zmq
        self.__socket = context.socket(zmq.SUB)
        self.__socket.setsockopt(zmq.RCVBUF .RCVHWM, 100)
        self.__socket.setsockopt(zmq.RCVTIMEO, 500)
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__socket.connect(connection)
        self.__socket.subscribe(topic)

        self.__poller = zmq.Poller()
        self.__poller.register(self.__socket, zmq.POLLIN) # POLLIN, POLLOUT, POLLERR

        self.__console.info(f"* Start Camera Monitor Subscriber ({topic})")

        self.start() # start thread

    def get_connection_info(self) -> str: # return connection address
        return self.__connection
    
    def get_topic(self) -> str: # return subscriber topic
        return self.__topic

    def run(self):
        """ Run subscriber thread """

        while not self.isInterruptionRequested():
            try:
                events = dict(self.__poller.poll(1000)) # wait 1 sec
                if self.__socket in events:
                    #topic, id, image_data, fps = self.__socket.recv_multipart()
                    topic, tags, image_data = self.__socket.recv_multipart()
                    dict_tags = json.loads(tags)

                    if topic.decode() == self.__topic:
                        nparr = np.frombuffer(image_data, np.uint8)
                        decoded_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        if decoded_image is not None:
                            # color_image = decoded_image #cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB) # color conversion if needed
                            h, w = decoded_image.shape[:2]  # height와 width
                            ch = decoded_image.shape[2] if len(decoded_image.shape) > 2 else 1
                            self.frame_update_signal.emit(decoded_image, dict_tags)

            except json.JSONDecodeError as e:
                self.__console.critical(f"<Camera Monitor> {e}")
                continue
            except zmq.ZMQError as e:
                self.__console.critical(f"<Camera Monitor> {e}")
                break
            except Exception as e:
                self.__console.critical(f"<Camera Monitor> {e}")
                break

    def socket_monitor(self, socket:zmq.SyncSocket):
        """ socket monitoring """
        try:
            monitor = socket.get_monitor_socket()
            while not self._monitor_thread_stop_event.is_set():
                if not monitor.poll(timeout=1000):  # 1sec timeout
                    continue

                event: Dict[str, any] = {}
                monitor_event = zmq_monitor.recv_monitor_message(monitor)
                event.update(monitor_event)
                event["description"] = EVENT_MAP[event["event"]]
                event_msg = event["description"].replace("EVENT_", "")
                endpoint = event["endpoint"].decode('utf-8')

                msg = f"[{endpoint}] {event_msg}" # message format
                self.status_msg_update_signal.emit(msg) # emit message to signal
                
            monitor.close()
        except  zmq.error.ZMQError as e:
            self.__console.error(f"{e}")

    def close(self) -> None:
        """ Close the socket and context """

        # self._monitor_thread_stop_event.set()
        # self._monitor_thread.join()

        self.requestInterruption()
        self.quit()
        self.wait()

        try:
            self.__socket.setsockopt(zmq.LINGER, 0)
            self.__poller.unregister(self.__socket)
            self.__socket.close()
        except zmq.ZMQError as e:
            self.__console.error(f"<Camera Monitor> {e}")
        
        self.__console.info(f"Close Camera Monitor subscriber")

        
