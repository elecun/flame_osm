import json
import time
from collections import deque
from util.logger.console import ConsoleLogger
from common.zpipe import AsyncZSocket, ZPipe
from common.zapi import ZAPIBase

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except ImportError:
    print("PyQt6 is required to run this application.")

class ZAPI(QObject, ZAPIBase):
    """ZMQ communication manager for SimTool.
    Uses DEALER socket to communicate with Viewer (Router).
    """
    
    # Signal for message reception: function_name, json_kwargs
    signal_message_received = pyqtSignal(str, str)

    def __init__(self, zpipe: ZPipe = None, transport: str = "ipc", channel: str = "/tmp/viewervedo"):
        QObject.__init__(self)
        ZAPIBase.__init__(self)
        
        self.__console = ConsoleLogger.get_logger()
        self._zpipe = zpipe
        self._running = False
        self._transport = transport
        self._channel = channel
        
        # Dealer Socket
        self.__dealer_socket = AsyncZSocket("ZAPI_SIMTOOL", "dealer")
        if not self.__dealer_socket.create(pipeline=zpipe):
            self.__console.error("Failed to create Simtool Socket(Dealer)")

    def run(self):
        """Start the ZAPI communication."""
        if self._running:
            self.__console.warning("[ZAPI_SIMTOOL] Already running")
            return
        self._running = True
        self.__console.debug("Starting Simtool ZAPI...")
        
        # Connect immediately (ZeroMQ handles reconnection)
        self.__dealer_socket.set_message_callback(self._on_message_received)
        if self.__dealer_socket.join(self._transport, self._channel):
            self.__console.info(f"[ZAPI SIMTOOL] Connected to {self._transport}://{self._channel}")
        else:
            self.__console.error(f"[ZAPI SIMTOOL] Failed to connect to {self._transport}://{self._channel}")

    def stop(self):
        """Stop and cleanup."""
        self._running = False
        
        if self.__dealer_socket:
            self.__dealer_socket.destroy_socket()
        
        self.__console.debug("[ZAPI] Stopped")

    def _on_message_received(self, multipart_data):
        """Callback from sockets."""
        try:
            if len(multipart_data) < 3:
                return

            socket_name = multipart_data[0]
            function_name = multipart_data[1]
            json_kwargs = multipart_data[2]
            
            # Decode components
            socket_str = socket_name.decode('utf-8') if isinstance(socket_name, bytes) else socket_name
            function_str = function_name.decode('utf-8') if isinstance(function_name, bytes) else function_name
            kwargs_str = json_kwargs.decode('utf-8') if isinstance(json_kwargs, bytes) else json_kwargs
            
            # Emit Signal to Main Thread
            self.signal_message_received.emit(function_str, kwargs_str)

        except Exception as e:
            self.__console.error(f"[ZAPI] Error receiving message: {e}")

    # ----------------------------------------------------------------
    # Supporting ZAPIs
    # ----------------------------------------------------------------
    def _ZAPI_request_load_spool(self, file_path: str):
        """Sends command to load spool."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "path": file_path
            }
            # Use ZAPIBase.call
            self.call(self.__dealer_socket, "zapi_load_spool", kwargs)
            self.__console.info(f"[ZAPI] Sent load_spool request: {file_path}")
        else:
            self.__console.warning("[ZAPI] Cannot send load_spool: Socket not connected")

    def _ZAPI_request_load_test_weld_point(self, file_path: str):
        """Sends command to load test weld points."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "path": file_path
            }
            # Use ZAPIBase.call
            self.call(self.__dealer_socket, "zapi_load_test_weld_point", kwargs)
            self.__console.info(f"[ZAPI] Sent load_test_weld_point request: {file_path}")
        else:
            self.__console.warning("[ZAPI] Cannot send load_test_weld_point: Socket not connected")

    def _ZAPI_request_set_mode(self, mode: str):
        """Sends command to set execution mode (simulation or real)."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "mode": mode
            }
            self.call(self.__dealer_socket, "zapi_set_mode", kwargs)
            self.__console.info(f"[ZAPI] Sent set_mode request: {mode}")
        else:
            self.__console.warning("[ZAPI] Cannot send set_mode: Socket not connected")
