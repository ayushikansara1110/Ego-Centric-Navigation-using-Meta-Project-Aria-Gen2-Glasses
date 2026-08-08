import time
import threading
import asyncio
import edge_tts
import pygame
import tempfile
import os
import numpy as np
from rapidfuzz import fuzz, process
import re
import queue
import wave
from faster_whisper import WhisperModel

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver

import localize
import navigate_videp
from replay_test import _vio_pose_to_T_device_from_odometry, _vio_status_is_valid

PRIO_COLLISION, PRIO_OFF_PATH, PRIO_NAV = 0, 1, 2
COOLDOWN_S = {PRIO_COLLISION: 1.0, PRIO_OFF_PATH: 4.0, PRIO_NAV: 0.0}
VOICE = "en-GB-SoniaNeural"
FRONT_LEFT_CAMERA_ID = 1
# ── Speech input ──────────────────────────────────────────────────────────────

STT_MODEL_SIZE = "small.en"
STT_SAMPLE_RATE = 48000

_audio_queue = queue.Queue()


# ── TTS ──────────────────────────────────────────────────────────────────────

pygame.mixer.init()

def _speak_blocking(text):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        asyncio.run(edge_tts.Communicate(text, VOICE).save(tmp))
        pygame.mixer.music.load(tmp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

_tts_queue = []
_tts_lock = threading.Lock()
_last_spoken = {}   # prio -> last spoken time

def speak(text, prio=PRIO_NAV):
    now = time.time()
    with _tts_lock:
        if now - _last_spoken.get(prio, 0) < COOLDOWN_S[prio]:
            return
        _last_spoken[prio] = now
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


# ── LiveCache ─────────────────────────────────────────────────────────────────

class LiveCache:
    def __init__(self):
        self.latest_vio = None
        self.latest_slam_frame = None
        self.latest_rgb_frame = None
        self.device_calib = None
        self._lock = threading.Lock()

    # diagnostic *args versions — replace with typed signatures after desk test
    def on_vio(self, vio_data):
        with self._lock:
            self.latest_vio = vio_data

    def on_slam(self, image, record):
    # mp_streaming_demo sends all 4 SLAM cameras.
    # We only want slam-front-left, which must match the camera
    # used to build reloc_index.pkl and for offline localization.

    # TEMP diagnostic: camera_id tells us which physical SLAM camera this is.
        if not hasattr(self, "_seen_slam_ids"):
            self._seen_slam_ids = set()

        if record.camera_id not in self._seen_slam_ids:
            self._seen_slam_ids.add(record.camera_id)
            print(
                f"[SLAM CAMERA] camera_id={record.camera_id} "
                f"frame={record.frame_number}"
            )

        # IMPORTANT:
        # Replace FRONT_LEFT_CAMERA_ID after the first run once we see the IDs.
        if record.camera_id != FRONT_LEFT_CAMERA_ID:
            return

        frame = image.to_numpy_array()

        if frame is None:
            return

        with self._lock:
            self.latest_slam_frame = frame

    def on_rgb(self, *args):
        print("[RGB] ", len(args), [type(a).__name__ for a in args])

    def on_device_calib(self, *args):
        print("[CALIB]", len(args), [type(a).__name__ for a in args])
        with self._lock:
            self.device_calib = args[0]   # almost certainly the first arg; confirm

    def get_vio(self):
        with self._lock:
            return self.latest_vio

    def get_slam(self):
        with self._lock:
            return self.latest_slam_frame

    def get_calib(self):
        with self._lock:
            return self.device_calib

    def on_audio(self, *args):
        """
        Temporary diagnostic callback.

        First run: print the callback signature so we can see exactly what
        Aria Gen 2's Python receiver gives us for audio.
        """
        if not hasattr(self, "_seen_audio_callback"):
            self._seen_audio_callback = True

            print("\n=== AUDIO CALLBACK ===")
            print("number of args:", len(args))

            for i, arg in enumerate(args):
                print(
                    f"arg[{i}] type={type(arg)} "
                    f"attrs={[x for x in dir(arg) if not x.startswith('_')][:30]}"
                )

            print("======================\n")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap(tracker, cache, timeout_s=90.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        vio_data = cache.get_vio()
        frame = cache.get_slam()
        if vio_data is not None and frame is not None and _vio_status_is_valid(vio_data):
            if tracker.bootstrap(
                cache.get_slam,
                lambda: _vio_pose_to_T_device_from_odometry(cache.get_vio()),
                timeout_s=1.0,
                retry_interval_s=0.2,
            ):
                return True
        time.sleep(0.2)
    return False


def node_for_label(label, labels, threshold=60):
    if label in labels:
        return labels[label]

    # Room numbers get priority over fuzzy matching
    digits = re.sub(r"\D", "", label)

    if digits:
        exact_digit_matches = [
            name for name in labels
            if digits in re.sub(r"\D", "", name)
        ]

        if len(exact_digit_matches) == 1:
            match = exact_digit_matches[0]
            print(f"(matched '{label}' -> '{match}' by room number)")
            return labels[match]

    match, score, _ = process.extractOne(
        label,
        list(labels.keys()),
        scorer=fuzz.WRatio
    )

    if score >= threshold:
        print(
            f"(matched '{label}' -> '{match}', "
            f"score {score:.0f})"
        )
        return labels[match]

    raise KeyError(
        f"'{label}' not found "
        f"(closest: '{match}', score {score:.0f})"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    cache = LiveCache()

    device_client = sdk_gen2.DeviceClient()
    device = device_client.connect()
    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = "mp_streaming_demo"
    device.set_streaming_config(streaming_config)

    server_config = sdk_gen2.HttpServerConfig()
    server_config.address = "0.0.0.0"
    server_config.port = 6768
    stream_receiver = receiver.StreamReceiver()
    stream_receiver.set_server_config(server_config)

    print("\n=== AVAILABLE CALLBACKS ===")
    print([x for x in dir(stream_receiver) if x.startswith("register_")])
    print("===========================\n")

    stream_receiver.register_vio_callback(cache.on_vio)
    stream_receiver.register_slam_callback(cache.on_slam)
    stream_receiver.register_audio_callback(cache.on_audio)
    #stream_receiver.register_rgb_callback(cache.on_rgb)
    stream_receiver.register_device_calib_callback(cache.on_device_calib)
    print("Starting device streaming...")
    device.start_streaming()
    print("Device streaming started.")

    time.sleep(2.0)

    print("Starting receiver...")
    stream_receiver.start_server()
    print("Receiver started.")

    try:
        labels = navigate_videp.load_labels("/home/ayushi/aria_gen2/scripts/labels2.json")
        index = localize.load_index("/home/ayushi/aria_gen2/scripts/reloc_index2.pkl")
        graph = navigate_videp.load_graph("/home/ayushi/aria_gen2/graph2/new/graph.npz")

        print("Loading calibration from VRS (same device)...")
        from replay_test import get_camera_calibration, data_provider
        _provider = data_provider.create_vrs_data_provider(
            "/home/ayushi/aria_gen2/vrs_files/test1.vrs")
        camera_matrix, _, cam_calib, T_device_from_cam = get_camera_calibration(
            _provider, "slam-front-left")
        print("Calibration loaded.")

        tracker = localize.Tracker(index, camera_matrix, None,
                                   cam_calib=cam_calib, T_device_from_cam=T_device_from_cam)

        print("Getting bearings (bootstrapping)...")
        if not bootstrap(tracker, cache):
            print("Bootstrap failed -- no valid map localization found.")
            return
        print("Bootstrap OK.")

        # get first map position
        start_pos = None
        deadline = time.time() + 5.0
        while start_pos is None and time.time() < deadline:
            vio_data = cache.get_vio()
            if vio_data is not None and _vio_status_is_valid(vio_data):
                T_map = tracker.live_map_pose(_vio_pose_to_T_device_from_odometry(vio_data))
                if T_map is not None:
                    start_pos = T_map[:2, 3]
            time.sleep(0.1)
        if start_pos is None:
            print("Could not get initial map pose.")
            return
        start_node = navigate_videp.nearest_node(start_pos, graph)

        # Find nearest labelled location
        labelled_node_ids = list(labels.values())
        labelled_positions = graph.waypoints[labelled_node_ids]

        dists = np.linalg.norm(
            labelled_positions - np.asarray(start_pos),
            axis=1
        )

        nearest_idx = int(np.argmin(dists))
        nearest_node_id = labelled_node_ids[nearest_idx]
        nearest_dist = float(dists[nearest_idx])

        id_to_label = {node_id: name for name, node_id in labels.items()}
        nearest_label = id_to_label[nearest_node_id]

        location_msg = (
            f"You are approximately {nearest_dist:.1f} metres from {nearest_label}"
        )

        print(f"\n📍 {location_msg}")
        speak(location_msg, PRIO_NAV)
        print("\nAvailable destinations:")
        for name in labels:
            print(" -", name)
        goal_label = input("\nWhere do you want to go? ").strip()
        goal_node = node_for_label(goal_label, labels)
        path = navigate_videp.plan_path(graph, start_node, goal_node)

        if len(path) == 1:
            speak(f"You are already at {goal_label}")
            print(f"Already at {goal_label}.")
            return

        # edge_idx = 0
        # first_instruction = navigate_videp.instruction_for_edge(path, 0)
        # if first_instruction:
        #     print(first_instruction)
        #     speak(first_instruction, PRIO_NAV)

        edge_idx = 0

        # Total planned walking distance
        initial_distance = sum(
            float(np.linalg.norm(path[i + 1].xy - path[i].xy))
            for i in range(len(path) - 1)
        )

        first_instruction = (
            f"Navigation started. Continue straight. "
            f"Your destination is approximately "
            f"{initial_distance:.0f} metres away."
        )

        print(first_instruction)
        speak(first_instruction, PRIO_NAV)

        # ── Speech/navigation state ────────────────────────────────────────
        last_progress_announcement = time.time()

        PROGRESS_INTERVAL_S = 12.0

        # Turn guidance
        TURN_WARNING_DISTANCE_M = 6.0
        TURN_REANNOUNCE_STEP_M = 2.0

        # Destination guidance
        DESTINATION_NEAR_M = 5.0
        ARRIVAL_DISTANCE_M = 1.5

        said_destination_near = False
        announced_turn = None
        last_announced_turn_dist = None

        # background refresh disabled for test #1
        # tracker.start_background_refresh(cache.get_slam, lambda: _vio_pose_to_T_device_from_odometry(cache.get_vio()))

        while edge_idx < len(path) - 1:
            vio_data = cache.get_vio()

            if vio_data is None or not _vio_status_is_valid(vio_data):
                time.sleep(0.05)
                continue

            T_map = tracker.live_map_pose(
                _vio_pose_to_T_device_from_odometry(vio_data)
            )

            if T_map is None:
                time.sleep(0.05)
                continue

            pos_xy = T_map[:2, 3]

            goal_xy = graph.waypoints[goal_node]
            goal_dist = float(np.linalg.norm(pos_xy - goal_xy))

            # ── ARRIVAL ────────────────────────────────────────────────────
            # User does NOT need to stand exactly on the graph node.
            if goal_dist <= ARRIVAL_DISTANCE_M:
                instr = f"You have arrived at {goal_label}"
                print(f"\n📍 {instr}")
                speak(instr, PRIO_NAV)
                break

            heading_deg = navigate_videp.heading_from_matrix(T_map)
            # ── Automatically synchronize route progress ─────────────────────────
            new_edge_idx = navigate_videp.closest_route_edge(
                pos_xy,
                path,
                current_edge_idx=edge_idx,
                lookahead=6,
            )

            if new_edge_idx > edge_idx:
                print(
                    f"\n[ROUTE SYNC] edge {edge_idx} -> {new_edge_idx}"
                )
                edge_idx = new_edge_idx
            seg_start, seg_end = path[edge_idx], path[edge_idx + 1]

            # How much route remains?
            remaining = navigate_videp.remaining_path_distance(
                pos_xy, path, edge_idx
            )

            # Where is the next meaningful turn?
            turn_dist, turn_direction, turn_idx = navigate_videp.distance_to_next_turn(pos_xy, path, edge_idx)

            print(f"\r[LIVE] x={pos_xy[0]:7.2f} y={pos_xy[1]:7.2f} "
                  f"heading={heading_deg:7.1f}° edge={edge_idx+1}/{len(path)-1}",
                  end="", flush=True)

            # ── DESTINATION APPROACHING ────────────────────────────────────
            if (
                goal_dist <= DESTINATION_NEAR_M
                and not said_destination_near
            ):
                instr = (
                    f"Your destination is about "
                    f"{max(1, round(goal_dist))} metres ahead."
                )

                print(f"\n===> {instr}")
                speak(instr, PRIO_NAV)
                said_destination_near = True


            # ── ADVANCE TURN WARNING ───────────────────────────────────────
            if (
                turn_dist is not None
                and turn_dist <= TURN_WARNING_DISTANCE_M
            ):
                turn_key = (turn_idx, turn_direction)

                # New turn: reset the distance announcement history
                if announced_turn != turn_key:
                    last_announced_turn_dist = None

                should_announce = (
                    announced_turn != turn_key
                    or last_announced_turn_dist is None
                    or (last_announced_turn_dist - turn_dist) >= TURN_REANNOUNCE_STEP_M
                )

                if should_announce:
                    instr = (
                        f"In about {max(1, round(turn_dist))} metres, "
                        f"turn {turn_direction}."
                    )

                    print(f"\n===> {instr}")
                    speak(instr, PRIO_NAV)

                    announced_turn = turn_key
                    last_announced_turn_dist = turn_dist    

            # ── OCCASIONAL PROGRESS UPDATE ─────────────────────────────────
            now = time.time()

            if (
                now - last_progress_announcement >= PROGRESS_INTERVAL_S
                and goal_dist > DESTINATION_NEAR_M
            ):
                if turn_dist is not None and turn_dist < 10.0:
                    instr = (
                        f"Continue straight. "
                        f"Turn {turn_direction} in approximately "
                        f"{max(1, round(turn_dist))} metres."
                    )
                else:
                    instr = (
                        f"Continue straight. "
                        f"Approximately {max(1, round(remaining))} "
                        f"metres remaining."
                    )

                print(f"\n===> {instr}")
                speak(instr, PRIO_NAV)

                last_progress_announcement = now
            # correction = localize.check_path_adherence(pos_xy, heading_deg, seg_start.xy, seg_end.xy)
            # if correction:
            #     speak(correction, PRIO_OFF_PATH)

            if navigate_videp.reached_waypoint(pos_xy, seg_end.xy):
                edge_idx += 1

                if edge_idx < len(path) - 1:
                    instr = navigate_videp.instruction_for_edge(
                        path, edge_idx
                    )

                    # Only speak meaningful turns.
                    # Do NOT announce every tiny graph segment.
                    if instr and (
                        "turn" in instr.lower()
                        or "bear" in instr.lower()
                    ):
                        print(f"\n===> {instr}")
                        speak(instr, PRIO_NAV)

                # Do NOT announce arrival here.
                # Arrival is handled using actual distance to the goal above.

            time.sleep(0.1)
            
        tracker.stop()

    finally:
        stream_receiver.stop_server()
        device.stop_streaming()
        device_client.disconnect(device)


if __name__ == "__main__":
    run()