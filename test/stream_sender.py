import zmq
import time
import json
import cv2
import os

# ==============================================================================
# User-configurable parameters
# ==============================================================================
BIND_ADDRESS = "tcp://*:5201"
TOPIC = "image_stream_1_monitor"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample.jpg")
TARGET_WIDTH = 450
TARGET_HEIGHT = 800
FPS = 30
CAM_CHANNEL = 1
# ==============================================================================

def main():
    print(f"Loading image from: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found.")
        return

    # Load original image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Error: Failed to load image.")
        return

    # Resize image to target resolution (width 450, height 800)
    resized_image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

    # Encode image to JPEG format
    success, encoded_image = cv2.imencode(".jpg", resized_image)
    if not success:
        print("Error: JPEG encoding failed.")
        return

    image_bytes = encoded_image.tobytes()

    # Initialize ZMQ Publish socket
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(BIND_ADDRESS)
    print(f"ZMQ Publisher bound to {BIND_ADDRESS}")

    # Wait a bit for potential subscribers to connect
    time.sleep(1.0)

    # Frame interval in seconds
    interval = 1.0 / FPS
    print(f"Start publishing {TOPIC} at {FPS} FPS...")

    frame_count = 0
    try:
        while True:
            start_time = time.time()

            # Construct JSON metadata
            meta = {
                "fps": float(FPS),
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "type": int(resized_image.dtype.num if hasattr(resized_image.dtype, "num") else 16),
                "timestamp": int(time.time() * 1000),
                "cam_channel": CAM_CHANNEL
            }
            meta_str = json.dumps(meta)

            # Send multipart message: [topic, meta, payload]
            pub_socket.send_multipart([
                TOPIC.encode("utf-8"),
                meta_str.encode("utf-8"),
                image_bytes
            ])

            frame_count += 1
            if frame_count % FPS == 0:
                print(f"[Sender] Published frame count: {frame_count}")

            # Sleep to maintain FPS
            elapsed = time.time() - start_time
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Publisher interrupted by user.")
    finally:
        pub_socket.close()
        context.term()
        print("Publisher closed.")

if __name__ == "__main__":
    main()
