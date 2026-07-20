import zmq
import time
import json
import cv2
import os

BIND_ADDRESS = "tcp://*:5301"
TOPIC = "image_stream_1"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample.jpg")
FPS = 10
CAM_CHANNEL = 1

def main():
    print(f"Loading image from: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found.")
        return

    # Load image (BGR)
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Error: Failed to load image.")
        return

    height, width, channels = image.shape
    print(f"Loaded image size: {width}x{height}, channels: {channels}")

    # Convert to raw BGR bytes
    image_bytes = image.tobytes()

    # OpenCV type CV_8UC3 is 16
    cv_type = 16

    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(BIND_ADDRESS)
    print(f"ZMQ Raw BGR Publisher bound to {BIND_ADDRESS}")

    # Wait a bit for subscriber to connect
    time.sleep(1.0)

    interval = 1.0 / FPS
    print(f"Start publishing {TOPIC} as raw BGR at {FPS} FPS...")

    frame_count = 0
    try:
        while True:
            start_time = time.time()

            meta = {
                "fps": float(FPS),
                "width": width,
                "height": height,
                "type": cv_type,
                "timestamp": int(time.time() * 1000),
                "cam_channel": CAM_CHANNEL
            }
            meta_str = json.dumps(meta)

            # Send multipart message: [topic, topic, meta, raw_payload]
            pub_socket.send_multipart([
                TOPIC.encode("utf-8"),
                TOPIC.encode("utf-8"),
                meta_str.encode("utf-8"),
                image_bytes
            ])

            frame_count += 1
            if frame_count % FPS == 0:
                print(f"[Sender] Published raw frame count: {frame_count}")

            # Maintain frame rate
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
