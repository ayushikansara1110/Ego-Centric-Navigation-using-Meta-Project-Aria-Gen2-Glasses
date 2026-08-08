import time
import localize
import navigate_videp
from rapidfuzz import fuzz, process
from replay_test_new import (data_provider, get_camera_calibration, _stream_frames_by_timestamp,
    replay_localization_only, plot_localization, replay_full_pipeline)
import re
import matplotlib
matplotlib.use("TkAgg")  # avoids the Qt plugin conflict cv2 causes
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["VRS_LOG_LEVEL"] = "ERROR"   # or "WARNING" / "OFF" depending on build


def node_for_label(label, labels, threshold=60):
    if label in labels:
        return labels[label]

    # if the query is purely/mostly digits, prefer an exact substring match
    # over fuzzy scoring -- fuzzy matching treats "3028" and "3038" as nearly
    # identical, which is wrong for room numbers
    digits = re.sub(r"\D", "", label)
    if digits:
        exact_digit_matches = [name for name in labels if digits in re.sub(r"\D", "", name)]
        if len(exact_digit_matches) == 1:
            match = exact_digit_matches[0]
            print(f"(matched '{label}' -> '{match}' by room number)")
            return labels[match]

    match, score, _ = process.extractOne(label, list(labels.keys()), scorer=fuzz.WRatio)
    if score >= threshold:
        print(f"(matched '{label}' -> '{match}', score {score:.0f})")
        return labels[match]
    raise KeyError(f"'{label}' not found (closest: '{match}', score {score:.0f})")

VRS_PATH = "/home/ayushi/aria_gen2/vrs_files/test6.vrs"
SLAM_LABEL = "slam-front-left"

labels = navigate_videp.load_labels("/home/ayushi/aria_gen2/scripts/labels2.json")
id_to_label = {v: k for k, v in labels.items()}

print("=== Loading map data ===")
index = localize.load_index("/home/ayushi/aria_gen2/scripts/reloc_index2.pkl")
graph = navigate_videp.load_graph("/home/ayushi/aria_gen2/graph2/new/graph.npz")

# --- FAST first fix: only scan frames until PnP succeeds once, not the whole file ---
print("\n=== Localizing... ===")
t0 = time.time()
provider = data_provider.create_vrs_data_provider(VRS_PATH)
camera_matrix, dist_coeffs, cam_calib, T_device_from_cam = get_camera_calibration(provider, SLAM_LABEL)
tracker = localize.Tracker(index, camera_matrix, dist_coeffs, cam_calib=cam_calib, T_device_from_cam=T_device_from_cam)
stream_id = provider.get_stream_id_from_label(SLAM_LABEL)

start_pos = None
for ts, frame in _stream_frames_by_timestamp(provider, stream_id):
    T_map = tracker.localize(frame)
    if T_map is not None:
        start_pos = T_map[:2, 3]
        break

if start_pos is None:
    raise RuntimeError("Could not localize on any frame in this recording.")

start_node = navigate_videp.nearest_node(start_pos, graph)

# nearest node with an actual label, not just the nearest raw waypoint --
# most waypoints (like node 108) are unlabelled corridor points
labelled_node_ids = set(labels.values())
labelled_dists = np.linalg.norm(graph.waypoints[list(labelled_node_ids)] - np.array(start_pos), axis=1)
nearest_labelled_id = list(labelled_node_ids)[int(labelled_dists.argmin())]
nearest_labelled_dist = float(labelled_dists.min())
nearest_label_name = id_to_label[nearest_labelled_id]

print(f"Localized in {time.time()-t0:.1f}s")
print(f"You are approximately {nearest_labelled_dist:.1f}m from: {nearest_label_name}")

# --- ask destination, now that position is known ---
print("\nAvailable destinations:")
for name in labels:
    print(" -", name)
goal_label = input("\nWhere do you want to go? ").strip()
goal_node = node_for_label(goal_label, labels)
goal_label = id_to_label[goal_node]

# --- plan + instructions ---
path = navigate_videp.plan_path(graph, start_node, goal_node)
# print(f"\n=== Instructions to {goal_label} ===")
# for i in range(len(path) - 1):
#     print(navigate_videp.instruction_for_edge(path, i))
# print(f"You've arrived at {goal_label}.")
print(f"\n=== Instructions to {goal_label} ===")

instructions = navigate_videp.build_route_instructions(path)

for instr in instructions:
    print(instr)

print(f"You've arrived at {goal_label}.")



# --- full pass for the plot, driven by recorded VIO (PnP only for bootstrap
#     + periodic drift correction inside replay_full_pipeline itself) ---
print("\n=== Building trajectory plot (VIO-driven, PnP only for corrections) ===")
events = replay_full_pipeline(
    VRS_PATH, index, graph, labels, goal_label,
    camera_label=SLAM_LABEL, slam_stream_label=SLAM_LABEL,
    vio_stream_label="vio", pose_source="recorded_vio",
    drift_refresh_interval_s=20,  # periodic PnP correction, not per-frame
)

positions = [
    {"localized": True, "pos_xy": e["pos_xy"]}
    for e in events if e["type"] == "pose"
]

plot_localization(graph, positions, path=path, goal_label=goal_label,
                   save_path="/home/ayushi/aria_gen2/scripts/localization_plot.png")

for e in events:
    if e["type"] in ("off_path", "instruction", "localization_lost", "collision", "drift_corrected"):
        print(e)