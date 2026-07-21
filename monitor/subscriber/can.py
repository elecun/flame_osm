"""
CAN Monitor subscriber
@author Byunghun Hwang <bh.hwang@iae.re.kr>
"""

try:
    # using PyQt5
    from PyQt5.QtCore import QThread, pyqtSignal
except ImportError:
    # using PyQt6
    from PyQt6.QtCore import QThread, pyqtSignal

import zmq
import json
from util.logger.console import ConsoleLogger

class CANMonitorSubscriber(QThread):
    can_message_received = pyqtSignal(dict)

    def __init__(self, context: zmq.Context, connection: str, topic: str):
        super().__init__()
        self.__console = ConsoleLogger.get_logger()
        self.__console.info(f"CAN Monitor Connection : {connection} (topic:{topic})")

        self.__connection = connection
        self.__topic = topic

        self.__socket = context.socket(zmq.SUB)
        self.__socket.setsockopt(zmq.RCVHWM, 100)
        self.__socket.setsockopt(zmq.RCVTIMEO, 500)
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__socket.connect(connection)
        self.__socket.subscribe(topic)

        self.__poller = zmq.Poller()
        self.__poller.register(self.__socket, zmq.POLLIN)

        self.__console.info(f"* Start CAN Monitor Subscriber ({topic})")

    def run(self):
        while not self.isInterruptionRequested():
            try:
                events = dict(self.__poller.poll(1000))
                if self.__socket in events:
                    multipart = self.__socket.recv_multipart()
                    self.__console.info(f"<CAN Monitor> Raw multipart frames count={len(multipart)}: {multipart}")
                    if len(multipart) >= 2:
                        topic = multipart[0].decode()
                        payload = multipart[-1].decode()
                        if topic == self.__topic:
                            msg = json.loads(payload)
                            self.__console.info(f"<CAN Monitor> Received message [{topic}]: {msg}")
                            self.can_message_received.emit(msg)
            except zmq.ZMQError as e:
                self.__console.error(f"<CAN Monitor> ZMQ Error: {e}")
                break
            except json.JSONDecodeError as e:
                self.__console.error(f"<CAN Monitor> JSON Decode Error: {e}")
                continue
            except Exception as e:
                self.__console.error(f"<CAN Monitor> Exception: {e}")
                break

    def close(self) -> None:
        self.requestInterruption()
        self.quit()
        self.wait()
        try:
            self.__socket.setsockopt(zmq.LINGER, 0)
            self.__poller.unregister(self.__socket)
            self.__socket.close()
        except zmq.ZMQError as e:
            self.__console.error(f"<CAN Monitor> Close Error: {e}")
        self.__console.info("Close CAN Monitor subscriber")
