import time
import localize
import navigate_videp
from rapidfuzz import fuzz, process
from replay_test import (data_provider, get_camera_calibration, _stream_frames_by_timestamp,
    replay_localization_only, plot_localization)
import re
import matplotlib
matplotlib.use("TkAgg")  # avoids the Qt plugin conflict cv2 causes
import matplotlib.pyplot as plt
import numpy as np


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

VRS_PATH = "/home/ayushi/aria_gen2/vrs_files/test7.vrs"
SLAM_LABEL = "slam-front-left"

labels = navigate_videp.load_labels("/home/ayushi/aria_gen2/scripts/labels.json")
id_to_label = {v: k for k, v in labels.items()}

print("=== Loading map data ===")
index = localize.load_index("/home/ayushi/aria_gen2/scripts/reloc_index.pkl")
graph = navigate_videp.load_graph("/home/ayushi/aria_gen2/graphs/graph.npz")

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

# --- plan + instructions ---
path = navigate_videp.plan_path(graph, start_node, goal_node)
print(f"\n=== Instructions to {goal_label} ===")
for i in range(len(path) - 1):
    print(navigate_videp.instruction_for_edge(path, i))
print(f"You've arrived at {goal_label}.")

# --- full pass for the plot only, strided for speed ---
print("\n=== Building trajectory plot (strided for speed) ===")
results, success_rate = replay_localization_only(VRS_PATH, index, camera_label=SLAM_LABEL,
                                                    slam_stream_label=SLAM_LABEL, stride=5)
plot_localization(graph, results, path=path, goal_label=goal_label,
                   save_path="/home/ayushi/aria_gen2/scripts/localization_plot.png")