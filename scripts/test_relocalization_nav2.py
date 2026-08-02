import time
import queue
import threading
import asyncio
import edge_tts
import pygame
import tempfile
import os
import numpy as np
from rapidfuzz import process, fuzz

import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core import calibration

import localize
import perception
import navigate

PRIO_COLLISION = 0
PRIO_OFF_PATH = 1
PRIO_NAV_INSTRUCTION = 2
COOLDOWN_S = {PRIO_COLLISION: 1.0, PRIO_OFF_PATH: 4.0, PRIO_NAV_INSTRUCTION: 0.0}

LOCALIZE_INTERVAL_S = 0.5  # how often to run ORB+PnP -- real cost per call now, tune to your CPU
POSE_STALENESS_TIMEOUT_S = 5.0  # if no fresh fix in this long, warn instead of navigating on stale pose


class AriaStreamObserver:
    def __init__(self):
        self.latest_slam_frame = None
        self.latest_slam_ts = None
        self.latest_rgb_frame = None
        self.latest_rgb_ts = None
        self._lock = threading.Lock()

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        with self._lock:
            if record.camera_id == aria.CameraId.Slam1:  # confirmed against installed SDK
                self.latest_slam_frame = image
                self.latest_slam_ts = record.capture_timestamp_ns
            elif record.camera_id == aria.CameraId.Rgb:
                self.latest_rgb_frame = image
                self.latest_rgb_ts = record.capture_timestamp_ns

    def get_slam(self):
        with self._lock:
            return self.latest_slam_frame, self.latest_slam_ts

    def get_rgb(self):
        with self._lock:
            return self.latest_rgb_frame, self.latest_rgb_ts


class AriaStream:
    def __init__(self):
        self.device_client = None
        self.device = None
        self.streaming_manager = None
        self.streaming_client = None
        self.observer = AriaStreamObserver()

    def init(self):
        self.device_client = aria.DeviceClient()
        self.device = self.device_client.connect()

        self.streaming_manager = self.device.streaming_manager
        self.streaming_client = self.streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = "profile8"
        self.streaming_manager.streaming_config = streaming_config
        self.streaming_manager.start_streaming()

        sub_config = self.streaming_client.subscription_config
        sub_config.subscriber_data_type = aria.StreamingDataType.Slam | aria.StreamingDataType.Rgb
        self.streaming_client.subscription_config = sub_config

        self.streaming_client.set_streaming_client_observer(self.observer)
        self.streaming_client.subscribe()

    def load_camera_calibration(self):
        sensors_calib_json = self.streaming_manager.sensors_calibration()
        device_calib = calibration.device_calibration_from_json_string(sensors_calib_json)
        slam_cam_calib = device_calib.get_camera_calib("camera-slam-left")
        return slam_cam_calib.intrinsics_matrix, slam_cam_calib.distortion_coeffs

    def shutdown(self):
        try:
            if self.streaming_client is not None:
                self.streaming_client.unsubscribe()
        except Exception as e:
            print(f"[shutdown] unsubscribe failed: {e}")
        try:
            if self.streaming_manager is not None:
                self.streaming_manager.stop_streaming()
        except Exception as e:
            print(f"[shutdown] stop_streaming failed: {e}")
        try:
            if self.device_client is not None and self.device is not None:
                self.device_client.disconnect(self.device)
        except Exception as e:
            print(f"[shutdown] disconnect failed: {e}")


class VoiceArbiter:
    def __init__(self):
        self.voice = "en-GB-SoniaNeural"
        self._q = queue.PriorityQueue()
        self._last_spoken = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        pygame.mixer.init()

    async def _generate(self, text, filename):
        communicate = edge_tts.Communicate(text=text, voice=self.voice, rate="-10%", pitch="-2Hz")
        await communicate.save(filename)

    def _speak(self, text):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            filename = f.name
        try:
            asyncio.run(self._generate(text, filename))
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
        finally:
            try:
                os.remove(filename)
            except Exception:
                pass

    def say(self, priority, text):
        now = time.time()
        if now - self._last_spoken.get(text, 0) < COOLDOWN_S[priority]:
            return
        self._q.put((priority, now, text))
        if priority == PRIO_COLLISION:
            pygame.mixer.music.stop()

    def _worker(self):
        while not self._stop.is_set():
            try:
                priority, ts, text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._last_spoken[text] = time.time()
            print(f"[SPEAKING] {text}")
            self._speak(text)

    def close(self):
        self._stop.set()
        pygame.mixer.music.stop()
        pygame.mixer.quit()


def ask_destination(labels, voice):
    label_names = list(labels.keys())
    while True:
        raw = input("Where do you want to go? ").strip()
        match, score, _ = process.extractOne(raw, label_names, scorer=fuzz.WRatio)
        if score >= 70:
            voice.say(PRIO_NAV_INSTRUCTION, f"Heading to {match}. Let's go.")
            return match
        print(f"Not sure that's a match (closest: '{match}', confidence {score}). Try again.")


def run():
    stream = AriaStream()
    voice = VoiceArbiter()

    try:
        stream.init()

        graph = navigate.load_graph("maps/graph.pkl")
        labels = navigate.load_labels("maps/labels.json")
        index = localize.load_index("maps/reloc_index.pkl")
        camera_matrix, dist_coeffs = stream.load_camera_calibration()

        tracker = localize.Tracker(index, camera_matrix, dist_coeffs)
        perceiver = perception.Perception(focal_length_px=camera_matrix[0, 0])

        voice.say(PRIO_NAV_INSTRUCTION, "Hold still, getting my bearings.")
        T_map = None
        while T_map is None:
            frame, _ = stream.observer.get_slam()
            if frame is not None:
                T_map = tracker.localize(frame)
            if T_map is None:
                voice.say(PRIO_NAV_INSTRUCTION, "Still trying to localize, please stay still.")
                time.sleep(1.0)

        start_pos = T_map[:2, 3]
        start_node = navigate.nearest_node(start_pos, graph)
        print(f"Localized near node: {start_node}")

        goal_label = ask_destination(labels, voice)
        goal_node = navigate.node_for_label(goal_label, labels)
        path = navigate.plan_path(graph, start_node, goal_node)
        edge_idx = 0

        last_rgb_ts = None
        last_localize_t = 0.0
        last_T_map = T_map
        last_fresh_t = time.time()
        warned_lost = False

        while edge_idx < len(path) - 1:
            now = time.time()

            if now - last_localize_t >= LOCALIZE_INTERVAL_S:
                frame, _ = stream.observer.get_slam()
                if frame is not None:
                    T_new, was_fresh = tracker.localize_or_last_known(frame)
                    if T_new is not None:
                        last_T_map = T_new
                        if was_fresh:
                            last_fresh_t = now
                            warned_lost = False
                last_localize_t = now

            if (now - last_fresh_t) > POSE_STALENESS_TIMEOUT_S and not warned_lost:
                voice.say(PRIO_OFF_PATH, "Having trouble tracking your position, please slow down")
                warned_lost = True

            pos_xy = last_T_map[:2, 3]
            heading_deg = navigate.heading_from_matrix(last_T_map)

            frame_rgb, rgb_ts = stream.observer.get_rgb()
            if frame_rgb is not None and rgb_ts != last_rgb_ts:
                last_rgb_ts = rgb_ts
                _, people = perceiver.process_frame(frame_rgb)
                for person in people:
                    if person.is_collision_risk:
                        voice.say(PRIO_COLLISION, "Person ahead, stop or step aside")
                    elif person.distance_m and person.distance_m < 2.5:
                        voice.say(PRIO_COLLISION, f"Person approaching on your {person.side}")

            seg_start, seg_end = path[edge_idx], path[edge_idx + 1]
            correction = localize.check_path_adherence(pos_xy, heading_deg, seg_start.xy, seg_end.xy)
            if correction:
                voice.say(PRIO_OFF_PATH, correction)

            if navigate.reached_waypoint(pos_xy, seg_end.xy):
                edge_idx += 1
                if edge_idx < len(path) - 1:
                    voice.say(PRIO_NAV_INSTRUCTION, navigate.instruction_for_edge(path, edge_idx))
                else:
                    voice.say(PRIO_NAV_INSTRUCTION, f"You've arrived at {goal_label}")

            time.sleep(0.05)

    finally:
        voice.close()
        stream.shutdown()


if __name__ == "__main__":
    run()