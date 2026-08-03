import numpy as np
from projectaria_tools.core import data_provider

import localize
import perception
import navigate_videp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  


def _vio_status_is_valid(vio_data):
    status_str = str(vio_data.status)
    quality_str = str(vio_data.pose_quality)
    bad_markers = ("NOT_INITIALIZED", "INVALID", "BAD", "FAILED", "LOST")
    if any(m in status_str.upper() for m in bad_markers):
        return False
    if any(m in quality_str.upper() for m in bad_markers):
        return False
    return True


def _se3_to_matrix(se3_obj):
    if hasattr(se3_obj, "matrix"):
        return np.array(se3_obj.matrix())
    if hasattr(se3_obj, "to_matrix"):
        return np.array(se3_obj.to_matrix())
    if hasattr(se3_obj, "rotationMatrix") and hasattr(se3_obj, "translation"):
        T = np.eye(4)
        T[:3, :3] = np.array(se3_obj.rotationMatrix())
        T[:3, 3] = np.array(se3_obj.translation()).flatten()
        return T
    if hasattr(se3_obj, "rotation") and hasattr(se3_obj, "translation"):
        T = np.eye(4)
        T[:3, :3] = np.array(se3_obj.rotation().matrix()) if hasattr(se3_obj.rotation(), "matrix") else np.array(se3_obj.rotation())
        T[:3, 3] = np.array(se3_obj.translation()).flatten()
        return T
    raise AttributeError(
        f"{type(se3_obj)} has none of the expected conversion methods -- "
        f"run print(dir({se3_obj})) manually and add the real method to _se3_to_matrix()."
    )


def _vio_pose_to_T_device_from_odometry(vio_data):
    # Try the two-step composed transform first
    if hasattr(vio_data, "transform_odometry_bodyimu") and hasattr(vio_data, "transform_bodyimu_device"):
        T_odom_bodyimu = _se3_to_matrix(vio_data.transform_odometry_bodyimu)
        T_bodyimu_device = _se3_to_matrix(vio_data.transform_bodyimu_device)
        return T_odom_bodyimu @ T_bodyimu_device
    # Fall back to a single direct odometry->device transform, if that's what this build exposes
    if hasattr(vio_data, "transform_odometry_device"):
        return _se3_to_matrix(vio_data.transform_odometry_device)
    raise AttributeError(
        f"FrontendOutput has none of the expected transform attributes -- "
        f"run print([a for a in dir({vio_data}) if 'transform' in a.lower()]) "
        f"and update this function with the real attribute name(s)."
    )


def list_streams(vrs_path):
    provider = data_provider.create_vrs_data_provider(vrs_path)
    for stream_id in provider.get_all_streams():
        label = provider.get_label_from_stream_id(stream_id)
        print(stream_id, label)

    # one-time VIO diagnostic, safe no-op if this recording has no vio stream
    vio_stream_id = provider.get_stream_id_from_label("vio")
    if vio_stream_id is not None and provider.get_num_data(vio_stream_id) > 5:
        sample_vio = provider.get_vio_data_by_index(vio_stream_id, 5)  # single object, no unpack
        print("VIO transform attrs found:", [a for a in dir(sample_vio) if "transform" in a.lower()])
        print("VIO status:", sample_vio.status, "| quality:", sample_vio.pose_quality)


def get_camera_calibration(provider, camera_label="slam-front-left"):
    
    device_calib = provider.get_device_calibration()
    cam_calib = device_calib.get_camera_calib(camera_label)
    if cam_calib is None:
        available = [c for c in device_calib.get_all_labels() if "slam" in c.lower()]
        raise ValueError(
            f"'{camera_label}' not found in calibration -- "
            f"SLAM-related labels available: {available}"
        )
    camera_matrix = localize.build_camera_matrix(cam_calib)
    return camera_matrix, None, cam_calib


def _stream_frames_by_timestamp(provider, stream_id):
    num = provider.get_num_data(stream_id)
    for i in range(num):
        image_data, record = provider.get_image_data_by_index(stream_id, i)
        yield record.capture_timestamp_ns, image_data.to_numpy_array()


def replay_localization_only(vrs_path, index, camera_label="slam-front-left",
                              slam_stream_label="slam-front-left", stride=1):

    provider = data_provider.create_vrs_data_provider(vrs_path)
    camera_matrix, dist_coeffs, cam_calib = get_camera_calibration(provider, camera_label)
    tracker = localize.Tracker(index, camera_matrix, dist_coeffs, cam_calib=cam_calib)

    stream_id = provider.get_stream_id_from_label(slam_stream_label)
    if stream_id is None:
        raise ValueError(f"Stream label '{slam_stream_label}' not found -- run list_streams() first.")

    results = []
    for i, (ts, frame) in enumerate(_stream_frames_by_timestamp(provider, stream_id)):
        if i % stride != 0:
            continue
        T_map = tracker.localize(frame)
        results.append({
            "timestamp_ns": ts,
            "localized": T_map is not None,
            "pos_xy": T_map[:2, 3].tolist() if T_map is not None else None,
        })

    success_rate = sum(r["localized"] for r in results) / max(len(results), 1)
    print(f"{vrs_path}: {len(results)} SLAM frames sampled, {success_rate:.1%} localization success rate")
    return results, success_rate


def replay_full_pipeline(vrs_path, index, graph, labels, goal_label,
                          camera_label="slam-front-left",
                          slam_stream_label="slam-front-left",
                          rgb_stream_label="camera-rgb",
                          vio_stream_label="vio",
                          pose_source="pnp"):

    if pose_source not in ("pnp", "recorded_vio"):
        raise ValueError(f"pose_source must be 'pnp' or 'recorded_vio', got {pose_source!r}")

    provider = data_provider.create_vrs_data_provider(vrs_path)
    camera_matrix, dist_coeffs, cam_calib = get_camera_calibration(provider, camera_label)
    tracker = localize.Tracker(index, camera_matrix, dist_coeffs, cam_calib=cam_calib)
    perceiver = perception.Perception(focal_length_px=camera_matrix[0, 0])

    # T_map_from_session: cached transform from VIO's odometry frame into the
    # map frame, established once via a real PnP solve at the first valid
    # VIO sample. None until that happens.
    T_map_from_session = None

    slam_stream_id = provider.get_stream_id_from_label(slam_stream_label)
    rgb_stream_id = provider.get_stream_id_from_label(rgb_stream_label)
    if slam_stream_id is None or rgb_stream_id is None:
        raise ValueError(
            "SLAM or RGB stream label not found -- run list_streams() on this "
            "file first and update slam_stream_label / rgb_stream_label."
        )

    slam_frames = [(ts, "slam", f) for ts, f in _stream_frames_by_timestamp(provider, slam_stream_id)]
    rgb_frames = [(ts, "rgb", f) for ts, f in _stream_frames_by_timestamp(provider, rgb_stream_id)]

    if pose_source == "recorded_vio":
        vio_stream_id = provider.get_stream_id_from_label(vio_stream_label)
        if vio_stream_id is None:
            raise ValueError(
                f"VIO stream label '{vio_stream_label}' not found -- run "
                f"list_streams() and confirm the label."
            )
        num_vio = provider.get_num_data(vio_stream_id)
        vio_frames = []
        for i in range(num_vio):
            vio_data = provider.get_vio_data_by_index(vio_stream_id, i)
            ts = getattr(vio_data, "capture_timestamp_ns", None)
            if ts is None:
                ts = getattr(vio_data, "tracking_timestamp_ns", None)  # fallback name, check dir() output below
            vio_frames.append((ts, "vio", vio_data))
        timeline = sorted(slam_frames + rgb_frames + vio_frames, key=lambda x: x[0])
    else:
        timeline = sorted(slam_frames + rgb_frames, key=lambda x: x[0])

    events = []
    path = None
    edge_idx = 0
    last_T_map = None
    last_good_ts_s = None
    STALE_TIMEOUT_S = 5.0
    warned_lost = False
    start_node = None

    for ts_ns, kind, frame in timeline:
        ts_s = ts_ns / 1e9

        if kind == "vio":
            vio_data = frame  # already the FrontendOutput object, not raw pixels
            if not _vio_status_is_valid(vio_data):
                continue  # bad/uninitialized sample -- hold last known pose

            T_session_from_device = _vio_pose_to_T_device_from_odometry(vio_data)

            if T_map_from_session is None:
                # Not bootstrapped yet -- need a real PnP solve on the
                # nearest-in-time SLAM frame to establish T_map_from_session.
                # Find the closest SLAM sample at or before this timestamp.
                nearest_slam = None
                for s_ts, s_kind, s_frame in timeline:
                    if s_kind == "slam" and s_ts <= ts_ns:
                        nearest_slam = s_frame
                    elif s_kind == "slam" and s_ts > ts_ns:
                        break
                if nearest_slam is not None:
                    T_map_from_cam = tracker.localize(nearest_slam)
                    if T_map_from_cam is not None:
                        T_map_from_session = T_map_from_cam @ np.linalg.inv(T_session_from_device)
                        events.append({"timestamp_ns": ts_ns, "type": "bootstrapped_vio"})
                # still None if PnP failed on the nearest frame -- try again
                # on the next valid VIO sample
                if T_map_from_session is None:
                    continue

            T_map = T_map_from_session @ T_session_from_device
            last_T_map = T_map
            last_good_ts_s = ts_s
            warned_lost = False

            if start_node is None:
                start_pos = T_map[:2, 3]
                start_node = navigate_videp.nearest_node(start_pos, graph)
                goal_node = navigate_videp.node_for_label(goal_label, labels)
                path = navigate_videp.plan_path(graph, start_node, goal_node)
                events.append({"timestamp_ns": ts_ns, "type": "localized_start", "node": start_node})
                events.append({"timestamp_ns": ts_ns, "type": "instruction",
                                "text": navigate_videp.instruction_for_edge(path, 0)})

        elif kind == "slam" and pose_source == "pnp":
            T_map, was_fresh = tracker.localize_or_last_known(frame)

            if T_map is not None:
                last_T_map = T_map
                if was_fresh:
                    last_good_ts_s = ts_s
                    warned_lost = False

                if start_node is None:
                    # First successful fix in the recording -- treat like
                    # "bootstrap" in the live script.
                    start_pos = T_map[:2, 3]
                    start_node = navigate_videp.nearest_node(start_pos, graph)
                    goal_node = navigate_videp.node_for_label(goal_label, labels)
                    path = navigate_videp.plan_path(graph, start_node, goal_node)
                    events.append({
                        "timestamp_ns": ts_ns, "type": "localized_start",
                        "node": start_node,
                    })
                    events.append({
                        "timestamp_ns": ts_ns, "type": "instruction",
                        "text": navigate_videp.instruction_for_edge(path, 0),
                    })

        # (kind == "slam" and pose_source == "recorded_vio" -> SLAM frames are
        # only used above, on-demand, inside the bootstrap search -- no
        # per-frame action needed here in that mode.)

        if kind in ("slam", "vio"):
            if last_good_ts_s is not None and (ts_s - last_good_ts_s) > STALE_TIMEOUT_S and not warned_lost:
                events.append({"timestamp_ns": ts_ns, "type": "localization_lost"})
                warned_lost = True

            if path is not None and last_T_map is not None and edge_idx < len(path) - 1:
                pos_xy = last_T_map[:2, 3]
                heading_deg = navigate_videp.heading_from_matrix(last_T_map)
                seg_start, seg_end = path[edge_idx], path[edge_idx + 1]

                correction = localize.check_path_adherence(pos_xy, heading_deg, seg_start.xy, seg_end.xy)
                if correction:
                    events.append({"timestamp_ns": ts_ns, "type": "off_path", "text": correction})

                if navigate_videp.reached_waypoint(pos_xy, seg_end.xy):
                    edge_idx += 1
                    if edge_idx < len(path) - 1:
                        text = navigate_videp.instruction_for_edge(path, edge_idx)
                    else:
                        text = f"You've arrived at {goal_label}"
                    events.append({"timestamp_ns": ts_ns, "type": "instruction", "text": text})

        elif kind == "rgb":
            _, people = perceiver.process_frame(frame)
            for person in people:
                if person.is_collision_risk:
                    events.append({"timestamp_ns": ts_ns, "type": "collision", "text": "Person ahead, stop or step aside"})
                elif person.distance_m and person.distance_m < 2.5:
                    events.append({
                        "timestamp_ns": ts_ns, "type": "collision",
                        "text": f"Person approaching on your {person.side}",
                    })

    return events


def compare_pose_sources(vrs_path, index, graph, labels, goal_label):
    pnp_events = replay_full_pipeline(vrs_path, index, graph, labels, goal_label, pose_source="pnp")
    vio_events = replay_full_pipeline(vrs_path, index, graph, labels, goal_label, pose_source="recorded_vio")

    def summarize(events, label):
        counts = {}
        for e in events:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        print(f"[{label}] {counts}")
        return counts

    print(f"--- {vrs_path} ---")
    pnp_counts = summarize(pnp_events, "pnp")
    vio_counts = summarize(vio_events, "recorded_vio")
    return {"pnp": pnp_events, "recorded_vio": vio_events, "pnp_counts": pnp_counts, "recorded_vio_counts": vio_counts}


def compare_recordings(vrs_paths, index_path="maps/reloc_index.pkl"):
    """Localization-only robustness comparison across recordings/conditions."""
    index = localize.load_index(index_path)
    print(index["descriptors"].shape)
    summary = {}
    for path in vrs_paths:
        _, rate = replay_localization_only(path, index)
        summary[path] = rate
    print("\n--- Summary ---")
    for path, rate in summary.items():
        print(f"{path}: {rate:.1%}")
    return summary


def plot_localization(graph, results, path=None, goal_label=None, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(graph.waypoints[:, 0], graph.waypoints[:, 1], c="lightgray", s=15, zorder=1, label="waypoints")

    if path is not None:
        path_xy = np.array([n.xy for n in path])
        ax.plot(path_xy[:, 0], path_xy[:, 1], "b--", linewidth=2, zorder=2, label="planned path")
        ax.scatter(*path_xy[0], c="blue", s=100, marker="^", zorder=4, label="start")
        ax.scatter(*path_xy[-1], c="blue", s=100, marker="*", zorder=4, label="goal")

    localized_xy = np.array([r["pos_xy"] for r in results if r["localized"]])
    lost_count = sum(1 for r in results if not r["localized"])
    if len(localized_xy) > 0:
        ax.plot(localized_xy[:, 0], localized_xy[:, 1], "g-", linewidth=1, alpha=0.6, zorder=3)
        ax.scatter(localized_xy[:, 0], localized_xy[:, 1], c="green", s=8, zorder=3, label="localized position")

    title = f"Localization{' -> ' + goal_label if goal_label else ''}"
    title += f"\n{len(localized_xy)}/{len(results)} frames localized ({lost_count} lost)"
    ax.set_title(title)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.axis("equal"); ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    compare_recordings([
        "/home/ayushi/aria_gen2/vrs_files/test1.vrs",
        "/home/ayushi/aria_gen2/vrs_files/test2.vrs",
        "/home/ayushi/aria_gen2/vrs_files/test3.vrs",
    ], index_path="/home/ayushi/aria_gen2/scripts/reloc_index.pkl")

    # Step 3 -- full pipeline replay (once localization numbers look solid):
    # index = localize.load_index("/home/ayushi/aria_gen2/scripts/reloc_index.pkl")
    # graph = navigate_videp.load_graph("/home/ayushi/aria_gen2/graphs/graph.pkl")
    # labels = navigate_videp.load_labels("/home/ayushi/aria_gen2/scripts/labels.json")
    # events = replay_full_pipeline(
    #     "/home/ayushi/aria_gen2/vrs_files/test1.vrs", index, graph, labels,
    #     goal_label="doctor waqar qureshi's office", pose_source="pnp",
    # )
    # for e in events:
    #     print(e)

    # Step 4 -- compare PnP-only vs recorded on-device VIO path adherence:
    # compare_pose_sources(
    #     "/home/ayushi/aria_gen2/vrs_files/test1.vrs", index, graph, labels,
    #     goal_label="doctor waqar qureshi's office",
    # )