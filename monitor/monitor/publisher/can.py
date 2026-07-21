"""
CAN Control Publisher
"""

try:
    from PyQt5.QtCore import QObject
except ImportError:
    from PyQt6.QtCore import QObject

import zmq
import json
from util.logger.console import ConsoleLogger

class CANControlPublisher(QObject):
    def __init__(self, context: zmq.Context, connection: str, topic: str):
        super().__init__()
        self.__console = ConsoleLogger.get_logger()
        self.__topic = topic
        self.__connection = connection
        # Convert tcp://IP:PORT to tcp://*:PORT for binding
        bind_address = connection
        if "://" in connection:
            parts = connection.split("://")
            protocol = parts[0]
            addr = parts[1]
            if ":" in addr:
                host, port = addr.split(":")
                bind_address = f"{protocol}://*:{port}"

        self.__socket = context.socket(zmq.PUB)
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__socket.bind(bind_address)
        self.__console.info(f"CAN Control Publisher initialized and bound to {bind_address} (original connection info: {connection}, topic: {topic})")

    def publish_control(self, msg_dict: dict):
        try:
            payload_str = json.dumps(msg_dict)
            self.__socket.send_multipart([self.__topic.encode('utf-8'), payload_str.encode('utf-8')])
            self.__console.info(f"[CAN Control Publisher Sent] topic: '{self.__topic}', payload: {payload_str}")
        except Exception as e:
            self.__console.error(f"Failed to publish CAN control message: {e}")

    def close(self):
        try:
            self.__socket.close()
            self.__console.info("Close CAN Control Publisher")
        except Exception as e:
            self.__console.error(f"Failed to close CAN Control socket: {e}")
