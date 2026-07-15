"""
Stream Receiver - Camera Monitor Image Stream Subscriber (Pure ZMQ)

Subscribes to image_stream_1_monitor and image_stream_2_monitor dataports
from the camera.monitor component and converts received data into
OpenCV-compatible numpy arrays.

No flame framework dependency — uses only zmq, numpy, and cv2.

@author auto-generated
"""

import zmq
import numpy as np
import cv2
import json
import time
import signal
import sys
import threading
from typing import Optional, Dict, Tuple, Callable

# ==============================================================================
# User-configurable parameters
# ==============================================================================

# ZMQ connection addresses for each monitor stream
# Format: "tcp://<host>:<port>"
STREAM_1_ADDRESS = "tcp://192.168.100.91:5201"
STREAM_2_ADDRESS = "tcp://192.168.100.91:5202"

# ZMQ subscription topics (must match the dataport topic on the publisher side)
STREAM_1_TOPIC = "image_stream_1_monitor"
STREAM_2_TOPIC = "image_stream_2_monitor"

# ZMQ socket options
ZMQ_RECV_HWM = 100       # Receive high-water mark (max queued messages)
ZMQ_RECV_TIMEOUT = 500    # Receive timeout in milliseconds
ZMQ_LINGER = 0            # Linger period on socket close (ms), 0 = discard immediately
ZMQ_POLL_TIMEOUT = 1000   # Poller timeout in milliseconds

# OpenCV decode flags
# cv2.IMREAD_UNCHANGED  — return image as-is (including alpha channel if present)
# cv2.IMREAD_COLOR      — always convert to 3-channel BGR
# cv2.IMREAD_GRAYSCALE  — always convert to single-channel grayscale
CV_DECODE_FLAG = cv2.IMREAD_UNCHANGED

# Enable/disable individual streams
ENABLE_STREAM_1 = True
ENABLE_STREAM_2 = True

# ==============================================================================


class StreamReceiver:
    """
    Subscribes to a single ZMQ PUB stream from camera.monitor component,
    receives multipart messages [topic, tag_json, image_data], and decodes
    the image data into an OpenCV-compatible numpy ndarray.
    """

    def __init__(
        self,
        context: zmq.Context,
        address: str,
        topic: str,
        on_frame: Optional[Callable[[np.ndarray, dict, str], None]] = None,
    ):
        """
        Args:
            context: Shared ZMQ context.
            address: ZMQ connection string, e.g. "tcp://192.168.100.91:5201".
            topic:   Subscription topic string, e.g. "image_stream_1_monitor".
            on_frame: Optional callback invoked as on_frame(image, tags, topic)
                      each time a valid frame is received.
        """
        self._address = address
        self._topic = topic
        self._on_frame = on_frame
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # --- ZMQ socket setup (mirrors monitor/subscriber/camera.py) ---
        self._socket = context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, ZMQ_RECV_HWM)
        self._socket.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT)
        self._socket.setsockopt(zmq.LINGER, ZMQ_LINGER)
        self._socket.connect(address)
        self._socket.subscribe(topic)

        # Poller for non-blocking receive with timeout
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

        print(f"[StreamReceiver] Subscribed to {address} (topic: {topic})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the receiver thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the receiver thread and clean up the socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        try:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._poller.unregister(self._socket)
            self._socket.close()
        except zmq.ZMQError as e:
            print(f"[StreamReceiver] Error closing socket ({self._topic}): {e}")

        print(f"[StreamReceiver] Stopped ({self._topic})")

    @property
    def address(self) -> str:
        return self._address

    @property
    def topic(self) -> str:
        return self._topic

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        """
        Main receive loop.

        Wire format from camera.monitor dispatch (ZMQ multipart):
            Frame 0: topic       (string, prepended by flame dispatch)
            Frame 1: tag_data    (JSON-encoded metadata, added via addmem)
            Frame 2: image_data  (encoded image bytes, added via addmem)
        """
        while self._running:
            try:
                events = dict(self._poller.poll(ZMQ_POLL_TIMEOUT))
                if self._socket not in events:
                    continue

                # Receive 3-frame multipart message: [topic, tags, image_data]
                parts = self._socket.recv_multipart()
                if len(parts) < 3:
                    print(
                        f"[StreamReceiver] ({self._topic}) "
                        f"Unexpected frame count: {len(parts)}, expected >= 3. Skipping."
                    )
                    continue

                raw_topic = parts[0]
                raw_tags = parts[1]
                raw_image = parts[2]

                # Verify topic matches
                decoded_topic = raw_topic.decode("utf-8", errors="replace")
                if decoded_topic != self._topic:
                    continue

                # Parse JSON tag metadata
                try:
                    tags: dict = json.loads(raw_tags)
                except json.JSONDecodeError as e:
                    print(f"[StreamReceiver] ({self._topic}) JSON decode error: {e}")
                    continue

                # Decode image bytes into OpenCV numpy array
                image = self._decode_image(raw_image)
                if image is None:
                    print(f"[StreamReceiver] ({self._topic}) Failed to decode image")
                    continue

                # Deliver to callback
                if self._on_frame is not None:
                    self._on_frame(image, tags, decoded_topic)

            except zmq.ZMQError as e:
                if self._running:
                    print(f"[StreamReceiver] ({self._topic}) ZMQ error: {e}")
                break
            except Exception as e:
                if self._running:
                    print(f"[StreamReceiver] ({self._topic}) Error: {e}")
                break

    @staticmethod
    def _decode_image(raw_data: bytes) -> Optional[np.ndarray]:
        """
        Decode raw image bytes into an OpenCV-compatible numpy ndarray.

        The camera.monitor component sends encoded image data (e.g. JPEG/PNG).
        This uses cv2.imdecode to convert the byte buffer into a BGR numpy array
        that can be used directly with any OpenCV function.

        Returns:
            numpy.ndarray (H, W, C) in BGR format, or None on failure.
        """
        buf = np.frombuffer(raw_data, dtype=np.uint8)
        image = cv2.imdecode(buf, CV_DECODE_FLAG)
        return image


# ==============================================================================
# Default frame callback — prints info and shows image via OpenCV window
# ==============================================================================

def _default_frame_handler(image: np.ndarray, tags: dict, topic: str) -> None:
    """
    Default callback for received frames.
    Prints frame metadata and displays the image using OpenCV highgui.
    """
    h, w = image.shape[:2]
    ch = image.shape[2] if image.ndim > 2 else 1
    print(
        f"[{topic}] Frame received — "
        f"size: {w}x{h}, channels: {ch}, dtype: {image.dtype}, tags: {tags}"
    )

    cv2.imshow(topic, image)
    cv2.waitKey(1)


# ==============================================================================
# Main entry point
# ==============================================================================

def main() -> None:
    # Shared ZMQ context (single context for all sockets)
    context = zmq.Context()

    receivers: list[StreamReceiver] = []

    # --- Create receivers based on user configuration ---
    if ENABLE_STREAM_1:
        receivers.append(
            StreamReceiver(
                context=context,
                address=STREAM_1_ADDRESS,
                topic=STREAM_1_TOPIC,
                on_frame=_default_frame_handler,
            )
        )

    if ENABLE_STREAM_2:
        receivers.append(
            StreamReceiver(
                context=context,
                address=STREAM_2_ADDRESS,
                topic=STREAM_2_TOPIC,
                on_frame=_default_frame_handler,
            )
        )

    if not receivers:
        print("[Main] No streams enabled. Exiting.")
        return

    # --- Graceful shutdown on Ctrl+C / SIGTERM ---
    shutdown_event = threading.Event()

    def _signal_handler(sig, frame):
        print("\n[Main] Shutdown signal received.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # --- Start all receivers ---
    for r in receivers:
        r.start()

    print(f"[Main] {len(receivers)} stream receiver(s) running. Press Ctrl+C to stop.")

    # Block until shutdown signal
    shutdown_event.wait()

    # --- Clean up ---
    for r in receivers:
        r.stop()

    cv2.destroyAllWindows()
    context.term()
    print("[Main] Done.")


if __name__ == "__main__":
    main()
