import argparse
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.sensor_data import ImageData, ImageDataRecord

from common import quit_keypress


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="profile8",
        help="Streaming profile name (default: profile9).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6768,
        help="Local port the stream receiver listens on (default: 6768).",
    )
    return parser.parse_args()


class FrameBuffer:
    """Thread-safe handoff between the receiver's callback thread and the main render loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def set(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            frame, self._frame = self._frame, None
            return frame


def main():
    args = parse_args()

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    frame_buffer = FrameBuffer()

    def rgb_callback(image_data: ImageData, image_record: ImageDataRecord):
        frame = image_data.to_numpy_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.set(frame)

    print("Connecting to device...")
    device_client = sdk_gen2.DeviceClient()
    device_client.set_client_config(sdk_gen2.DeviceClientConfig())
    device = device_client.connect()

    print(f"Starting streaming (profile={args.profile})...")
    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = args.profile
    device.set_streaming_config(streaming_config)
    device.start_streaming()

    print("Starting stream receiver...")
    server_config = sdk_gen2.HttpServerConfig()
    server_config.address = "0.0.0.0"
    server_config.port = args.port

    stream_receiver = receiver.StreamReceiver()
    stream_receiver.set_server_config(server_config)
    stream_receiver.register_rgb_callback(rgb_callback)
    stream_receiver.start_server()

    window_name = "Aria Gen 2 YOLO Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while not quit_keypress():
            frame = frame_buffer.get()
            if frame is None:
                time.sleep(0.001)
                continue

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            cv2.imshow(window_name, annotated_frame)
            cv2.waitKey(1)
    finally:
        print("Stopping stream...")
        device.stop_streaming()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()