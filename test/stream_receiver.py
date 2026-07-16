import zmq
import numpy as np
import cv2
import json

# ==============================================================================
# User-configurable parameters
# ==============================================================================
CONNECT_ADDRESS = "tcp://localhost:5201"
TOPIC = "image_stream_1_monitor"
# ==============================================================================

def main():
    # Initialize ZMQ Subscribe socket
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(CONNECT_ADDRESS)
    sub_socket.subscribe(TOPIC)
    print(f"ZMQ Subscriber connected to {CONNECT_ADDRESS} (Topic: {TOPIC})")

    frame_count = 0
    try:
        while True:
            # Receive multipart message: [topic, meta, payload]
            parts = sub_socket.recv_multipart()
            if len(parts) < 3:
                print(f"[Receiver] Received invalid multipart message with {len(parts)} frames")
                continue

            topic = parts[0].decode("utf-8")
            meta_str = parts[1].decode("utf-8")
            image_bytes = parts[2]

            try:
                meta = json.loads(meta_str)
            except json.JSONDecodeError:
                print("[Receiver] Failed to parse metadata JSON")
                continue

            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            # Decode JPEG image
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if image is not None:
                frame_count += 1
                h, w = image.shape[:2]
                print(f"[Receiver] Success: Received frame #{frame_count} | Size: {w}x{h} | Meta: {meta}")
            else:
                print("[Receiver] Error: Failed to decode received image.")

    except KeyboardInterrupt:
        print("Receiver interrupted by user.")
    finally:
        sub_socket.close()
        context.term()
        print("Receiver closed.")

if __name__ == "__main__":
    main()
