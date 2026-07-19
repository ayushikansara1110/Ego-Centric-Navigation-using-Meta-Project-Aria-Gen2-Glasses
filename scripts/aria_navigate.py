"""
aria_navigate.py
Main real-time loop. Run this instead of navigate.py directly.

Ties together:
  - localize.py    -> where am I, right now, in map coordinates
  - perception.py  -> doors and people in view, right now
  - navigate.py    -> path planning + turn-by-turn logic (existing, refactored
                       into importable functions — see checklist at bottom)

Voice output is arbitrated by priority so a collision warning always
interrupts a routine nav instruction instead of queuing behind it.
"""

import time
import queue
import threading
import pyttsx3

import localize
import perception
import navigate  # your existing file, refactored — see checklist below

# --- priority levels: lower number = more urgent, always jumps the queue ---
PRIO_COLLISION = 0
PRIO_OFF_PATH = 1
PRIO_DOOR = 2
PRIO_NAV_INSTRUCTION = 3

COOLDOWN_S = {
    PRIO_COLLISION: 1.0,     # can repeat quickly, it's safety-critical
    PRIO_OFF_PATH: 4.0,
    PRIO_DOOR: 3.0,
    PRIO_NAV_INSTRUCTION: 0.0,
}


class VoiceArbiter:
    """Speaks the highest-priority pending message; drops lower-priority
    messages that are stale by the time their turn comes."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self._q = queue.PriorityQueue()
        self._last_spoken = {}
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, priority, text):
        now = time.time()
        if now - self._last_spoken.get(text, 0) < COOLDOWN_S[priority]:
            return  # don't repeat the same line too often
        self._q.put((priority, now, text))

    def _worker(self):
        while True:
            priority, ts, text = self._q.get()
            self._last_spoken[text] = time.time()
            self.engine.say(text)
            self.engine.runAndWait()


def run(goal_label, maps_dir="maps/"):
    voice = VoiceArbiter()

    graph = navigate.load_graph(maps_dir + "graph.pkl")
    labels = navigate.load_labels(maps_dir + "labels.json")

    index = localize.load_index(maps_dir + "reloc_index.pkl")
    camera_matrix, dist_coeffs = load_camera_calibration()  # from Aria MPS calibration

    tracker = localize.Tracker(index, camera_matrix, dist_coeffs)
    perceiver = perception.Perception(focal_length_px=camera_matrix[0, 0])

    voice.say(PRIO_NAV_INSTRUCTION, "Hold still for a moment while I get my bearings")
    ok = tracker.bootstrap(get_frame_fn=get_live_frame, get_session_pose_fn=get_live_session_pose)
    if not ok:
        voice.say(PRIO_NAV_INSTRUCTION, "Couldn't localize yet, please walk forward a few steps")
        while not ok:
            ok = tracker.bootstrap(get_frame_fn=get_live_frame, get_session_pose_fn=get_live_session_pose, timeout_s=3.0)

    tracker.start_background_refresh(get_live_frame, get_live_session_pose)

    start_pos = tracker.live_map_pose(get_live_session_pose())[:2, 3]
    start_node = navigate.nearest_node(start_pos, graph)
    goal_node = navigate.node_for_label(goal_label, labels)
    path = navigate.plan_path(graph, start_node, goal_node)
    edge_idx = 0

    voice.say(PRIO_NAV_INSTRUCTION, f"Heading to {goal_label}. Let's go.")

    while edge_idx < len(path) - 1:
        T_session_from_cam = get_live_session_pose()
        T_map = tracker.live_map_pose(T_session_from_cam)
        pos_xy = T_map[:2, 3]
        heading_deg = navigate.heading_from_matrix(T_map)

        # --- path adherence ---
        seg_start, seg_end = path[edge_idx], path[edge_idx + 1]
        correction = localize.check_path_adherence(pos_xy, heading_deg, seg_start.xy, seg_end.xy)
        if correction:
            voice.say(PRIO_OFF_PATH, correction)

        # --- waypoint progress -> next turn instruction ---
        if navigate.reached_waypoint(pos_xy, seg_end.xy):
            edge_idx += 1
            if edge_idx < len(path) - 1:
                instruction = navigate.instruction_for_edge(path, edge_idx)
                voice.say(PRIO_NAV_INSTRUCTION, instruction)
            else:
                voice.say(PRIO_NAV_INSTRUCTION, f"You've arrived at {goal_label}")

        # --- perception: doors + people ---
        frame_rgb = get_live_rgb_frame()
        doors, people = perceiver.process_frame(frame_rgb)

        nearby_label = navigate.label_near_node(seg_end, labels)
        if nearby_label and doors:
            side = doors[0].side
            voice.say(PRIO_DOOR, f"The office of {nearby_label} is to your {side}")

        for person in people:
            if person.is_collision_risk:
                voice.say(PRIO_COLLISION, "Person ahead, stop or step aside")
            elif person.distance_m and person.distance_m < 2.5:
                voice.say(PRIO_COLLISION, f"Person approaching on your {person.side}")

        time.sleep(0.05)  # ~20Hz loop; tune to your actual frame rate


# --- stubs: wire these to your existing Gen 2 stream_receiver / FrameBuffer code ---
def get_live_frame():
    raise NotImplementedError("Return latest SLAM camera grayscale frame")

def get_live_rgb_frame():
    raise NotImplementedError("Return latest RGB frame for YOLO")

def get_live_session_pose():
    raise NotImplementedError("Return latest 4x4 pose matrix from Aria's on-device VIO/SLAM stream")

def load_camera_calibration():
    raise NotImplementedError("Return (camera_matrix, dist_coeffs) from Aria MPS calibration output")


if __name__ == "__main__":
    run(goal_label="Room 214")
