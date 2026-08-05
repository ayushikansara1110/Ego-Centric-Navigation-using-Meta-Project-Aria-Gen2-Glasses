import time
import threading
import queue
import asyncio
import edge_tts
import pygame
import tempfile
import os

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.sensor_data import FrontendOutput

import localize
import perception
import navigate_videp
from replay_test import _vio_pose_to_T_device_from_odometry, _vio_status_is_valid

PRIO_COLLISION, PRIO_OFF_PATH, PRIO_NAV = 0, 1, 2
COOLDOWN_S = {PRIO_COLLISION: 1.0, PRIO_OFF_PATH: 4.0, PRIO_NAV: 0.0}


class LiveCache:
    def __init__(self):
        self.latest_vio = None
        self.latest_slam_frame = None
        self.latest_rgb_frame = None
        self.device_calib = None
        self._lock = threading.Lock()

    def on_vio(self, vio_data):
        with self._lock:
            self.latest_vio = vio_data

    def on_slam(self, image, record):
        with self._lock:
            self.latest_slam_frame = image

    def on_rgb(self, image, record):
        with self._lock:
            self.latest_rgb_frame = image

    def on_device_calib(self, calib_data):
        with self._lock:
            self.device_calib = calib_data

    def get_vio(self):
        with self._lock:
            return self.latest_vio

    def get_slam(self):
        with self._lock:
            return self.latest_slam_frame

    def get_rgb(self):
        with self._lock:
            return self.latest_rgb_frame

    def get_calib(self):
        with self._lock:
            return self.device_calib

def bootstrap(tracker, cache, timeout_s=15.0):
    """Waits for one valid VIO sample + one SLAM frame, runs PnP once."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        vio_data = cache.get_vio()
        frame = cache.get_slam()
        if vio_data is not None and frame is not None and _vio_status_is_valid(vio_data):
            def get_frame_fn(): return frame
            def get_session_pose_fn(): return _vio_pose_to_T_device_from_odometry(vio_data)
            if tracker.bootstrap(get_frame_fn, get_session_pose_fn, timeout_s=1.0):
                return True
        time.sleep(0.2)
    return False


def run():
    cache = LiveCache()

    device_client = sdk_gen2.DeviceClient()
    device = device_client.connect()
    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = "mp_streaming_demo"
    device.set_streaming_config(streaming_config)
    device.start_streaming()

    server_config = sdk_gen2.HttpServerConfig()
    server_config.address = "0.0.0.0"
    server_config.port = 6768
    stream_receiver = receiver.StreamReceiver()
    stream_receiver.set_server_config(server_config)
    stream_receiver.register_vio_callback(cache.on_vio)
    stream_receiver.register_slam_callback(cache.on_slam)  # once name confirmed
    stream_receiver.register_rgb_callback(cache.on_rgb) 
    stream_receiver.register_device_calib_callback(cache.on_device_calib)
    stream_receiver.start_server()

    try:
        labels = navigate_videp.load_labels("/home/ayushi/aria_gen2/scripts/labels.json")
        index = localize.load_index("/home/ayushi/aria_gen2/scripts/reloc_index.pkl")
        graph = navigate_videp.load_graph("/home/ayushi/aria_gen2/graphs/graph.npz")

        camera_matrix, dist_coeffs, cam_calib, T_device_from_cam = localize.get_camera_calibration_placeholder()  # wire to real device calib source
        tracker = localize.Tracker(index, camera_matrix, dist_coeffs, cam_calib=cam_calib, T_device_from_cam=T_device_from_cam)  # wire to real device calib source

        print("Getting bearings...")
        if not bootstrap(tracker, cache):
            print("Could not localize -- check SLAM frame callback is wired.")
            return

        # ... rest mirrors run_test.py: nearest labelled node, ask destination,
        # plan_path, loop using tracker.live_map_pose(get_session_pose_fn())
        # each tick instead of tracker.localize(frame) fresh each time.

    finally:
        stream_receiver.stop_server()
        device.stop_streaming()
        device_client.disconnect(device)