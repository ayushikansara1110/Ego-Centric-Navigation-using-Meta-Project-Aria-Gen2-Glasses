import gzip
import csv
import pickle
import numpy as np
import cv2
from scipy.spatial import cKDTree

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

KEYFRAME_SPACING_M = 0.75   # sample a keyframe every ~0.75m of walked trajectory
MATCH_PIXEL_RADIUS = 4.0    # px radius to link an ORB keypoint to an observed point UID
OBS_WINDOW_NS = 50_000_000  # +/- 50ms window around each keyframe timestamp


def load_trajectory(traj_csv_path):
    """Returns list of (timestamp_ns, x, y, z, qw, qx, qy, qz) sorted by time."""
    rows = []
    with open(traj_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((
                int(r["tracking_timestamp_us"]) * 1000,
                float(r["tx_world_device"]), float(r["ty_world_device"]), float(r["tz_world_device"]),
                float(r["qw_world_device"]), float(r["qx_world_device"]),
                float(r["qy_world_device"]), float(r["qz_world_device"]),
            ))
    return rows


def sample_keyframe_timestamps(trajectory, spacing_m=KEYFRAME_SPACING_M):
    """Walk the trajectory, keep a timestamp every `spacing_m` meters travelled."""
    keep = [trajectory[0]]
    last_xyz = np.array(trajectory[0][1:4])
    for row in trajectory[1:]:
        xyz = np.array(row[1:4])
        if np.linalg.norm(xyz - last_xyz) >= spacing_m:
            keep.append(row)
            last_xyz = xyz
    return keep


def load_points(points_csv_gz_path):
    """uid -> xyz, from semidense_points.csv.gz"""
    points = {}
    with gzip.open(points_csv_gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for r in reader:
            points[int(r["uid"])] = np.array([float(r["px_world"]), float(r["py_world"]), float(r["pz_world"])])
    return points


def load_observations_sorted(observations_csv_gz_path):
    """
    Read the WHOLE observations file exactly once, into flat numpy arrays
    sorted by timestamp. Returns (ts_arr, uid_arr, u_arr, v_arr).
    This replaces the old per-keyframe full-file rescan.
    """
    ts_list, uid_list, u_list, v_list = [], [], [], []
    with gzip.open(observations_csv_gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts_list.append(int(r["frame_tracking_timestamp_us"]) * 1000)
            uid_list.append(int(r["uid"]))
            u_list.append(float(r["u"]))
            v_list.append(float(r["v"]))

    ts_arr = np.array(ts_list, dtype=np.int64)
    uid_arr = np.array(uid_list, dtype=np.int64)
    u_arr = np.array(u_list, dtype=np.float64)
    v_arr = np.array(v_list, dtype=np.float64)

    order = np.argsort(ts_arr, kind="mergesort")  # stable, fine for near-sorted data
    return ts_arr[order], uid_arr[order], u_arr[order], v_arr[order]


def get_observations_window(ts_arr, uid_arr, u_arr, v_arr, target_ts_ns, window_ns=OBS_WINDOW_NS):
    """Binary-search the pre-sorted arrays for the time window (O(log n), no file I/O)."""
    lo = np.searchsorted(ts_arr, target_ts_ns - window_ns, side="left")
    hi = np.searchsorted(ts_arr, target_ts_ns + window_ns, side="right")
    return uid_arr[lo:hi], u_arr[lo:hi], v_arr[lo:hi]


def get_frame_at_timestamp(vrs_reader, slam_stream_id, timestamp_ns):
    """
    Pull the SLAM-camera grayscale frame nearest to timestamp_ns using
    projectaria_tools' data_provider (Gen 2 SDK).
    """
    stream_id = StreamId(slam_stream_id)

    image_data, image_record = vrs_reader.get_image_data_by_time_ns(
        stream_id,
        timestamp_ns,
        TimeDomain.DEVICE_TIME,
        TimeQueryOptions.CLOSEST,
    )

    img = image_data.to_numpy_array()  # HxW grayscale for SLAM cameras
    return img


def match_keypoints_to_uids(kps, window_uid, window_u, window_v, radius=MATCH_PIXEL_RADIUS):
    """
    Vectorized nearest-neighbor match: all keypoints in this frame against all
    observed (u,v) in the time window, via a KD-tree instead of a nested Python loop.
    Returns a list the same length as kps: uid or None.
    """
    if len(window_uid) == 0 or len(kps) == 0:
        return [None] * len(kps)

    tree = cKDTree(np.column_stack([window_u, window_v]))
    kp_pts = np.array([kp.pt for kp in kps], dtype=np.float64)
    dists, idxs = tree.query(kp_pts, k=1, distance_upper_bound=radius)

    n = len(window_uid)
    results = []
    for d, i in zip(dists, idxs):
        if np.isfinite(d) and i < n:
            results.append(int(window_uid[i]))
        else:
            results.append(None)
    return results


def build_index(vrs_path, traj_csv, points_csv_gz, observations_csv_gz, slam_stream_id, out_path):
    orb = cv2.ORB_create(nfeatures=1500)
    trajectory = load_trajectory(traj_csv)
    keyframes = sample_keyframe_timestamps(trajectory)
    points = load_points(points_csv_gz)

    print("Loading observations file once...")
    ts_arr, uid_arr, u_arr, v_arr = load_observations_sorted(observations_csv_gz)
    print(f"Loaded {len(ts_arr)} observations")

    reader = data_provider.create_vrs_data_provider(vrs_path)
    assert reader is not None, f"Failed to open {vrs_path}"

    all_descs, all_xyz = [], []

    for i, (ts, x, y, z, qw, qx, qy, qz) in enumerate(keyframes):
        img = get_frame_at_timestamp(reader, slam_stream_id, ts)
        kps, descs = orb.detectAndCompute(img, None)
        if descs is None:
            continue

        win_uid, win_u, win_v = get_observations_window(ts_arr, uid_arr, u_arr, v_arr, ts)
        matched_uids = match_keypoints_to_uids(kps, win_uid, win_u, win_v)

        for uid, desc in zip(matched_uids, descs):
            if uid is not None and uid in points:
                all_descs.append(desc)
                all_xyz.append(points[uid])

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(keyframes)} keyframes")

    print(f"Indexed {len(all_descs)} descriptor->3D-point pairs from {len(keyframes)} keyframes")

    with open(out_path, "wb") as f:
        pickle.dump({
            "descriptors": np.array(all_descs, dtype=np.uint8),
            "xyz_map_frame": np.array(all_xyz, dtype=np.float64),
        }, f)


if __name__ == "__main__":
    build_index(
        vrs_path="/home/ayushi/aria_gen2/vrs_files/new/Corridor.vrs",
        traj_csv="/home/ayushi/aria_gen2/vrs_files/new/mps_Corridor_vrs/slam/closed_loop_trajectory.csv",
        points_csv_gz="/home/ayushi/aria_gen2/vrs_files/new/mps_Corridor_vrs/slam/semidense_points.csv.gz",
        observations_csv_gz="/home/ayushi/aria_gen2/vrs_files/new/mps_Corridor_vrs/slam/semidense_observations.csv.gz",
        slam_stream_id="1201-1",
        out_path="/home/ayushi/aria_gen2/scripts/reloc_index.pkl",
    )