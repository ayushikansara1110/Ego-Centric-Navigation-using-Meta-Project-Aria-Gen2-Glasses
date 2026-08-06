import gzip
import csv
import pickle
import numpy as np
import cv2
import gc
from scipy.spatial import cKDTree

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

KEYFRAME_SPACING_M = 0.75
MATCH_PIXEL_RADIUS = 4.0
OBS_WINDOW_NS = 50_000_000


def load_trajectory(traj_csv_path):
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
    keep = [trajectory[0]]
    last_xyz = np.array(trajectory[0][1:4])
    for row in trajectory[1:]:
        xyz = np.array(row[1:4])
        if np.linalg.norm(xyz - last_xyz) >= spacing_m:
            keep.append(row)
            last_xyz = xyz
    return keep


def _count_csv_rows_gz(path):
    """One fast pass to count data rows (excludes header) so we can preallocate."""
    n = 0
    with gzip.open(path, "rt") as f:
        next(f)  # header
        for _ in f:
            n += 1
    return n


def load_points_arrays(points_csv_gz_path):
    """
    Returns (sorted_uid_arr int64, xyz_arr float32 [N,3]) instead of a dict.
    Preallocated NumPy arrays -> no per-point Python object overhead.
    """
    n = _count_csv_rows_gz(points_csv_gz_path)
    print(f"  points file: {n} rows, preallocating arrays...")

    uid_arr = np.empty(n, dtype=np.int64)
    xyz_arr = np.empty((n, 3), dtype=np.float32)

    with gzip.open(points_csv_gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            uid_arr[i] = int(r["uid"])
            xyz_arr[i, 0] = float(r["px_world"])
            xyz_arr[i, 1] = float(r["py_world"])
            xyz_arr[i, 2] = float(r["pz_world"])

    order = np.argsort(uid_arr, kind="mergesort")
    uid_sorted = uid_arr[order]
    xyz_sorted = xyz_arr[order]
    del uid_arr, xyz_arr, order
    gc.collect()
    return uid_sorted, xyz_sorted


def lookup_points(uid_sorted, xyz_sorted, uid):
    """O(log n) lookup, dict-free. Returns xyz or None."""
    idx = np.searchsorted(uid_sorted, uid)
    if idx < len(uid_sorted) and uid_sorted[idx] == uid:
        return xyz_sorted[idx]
    return None


def load_observations_sorted(observations_csv_gz_path):
    """
    Preallocated-array version: counts rows first, fills NumPy arrays directly
    (no intermediate Python lists), then sorts by timestamp in place.
    """
    n = _count_csv_rows_gz(observations_csv_gz_path)
    print(f"  observations file: {n} rows, preallocating arrays...")

    ts_arr = np.empty(n, dtype=np.int64)
    uid_arr = np.empty(n, dtype=np.int64)
    u_arr = np.empty(n, dtype=np.float32)
    v_arr = np.empty(n, dtype=np.float32)

    with gzip.open(observations_csv_gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            ts_arr[i] = int(r["frame_tracking_timestamp_us"]) * 1000
            uid_arr[i] = int(r["uid"])
            u_arr[i] = float(r["u"])
            v_arr[i] = float(r["v"])
            if (i + 1) % 20_000_000 == 0:
                print(f"    read {i + 1}/{n} observation rows...")

    print("  sorting observations by timestamp...")
    order = np.argsort(ts_arr, kind="mergesort")
    ts_sorted = ts_arr[order]
    uid_sorted = uid_arr[order]
    u_sorted = u_arr[order]
    v_sorted = v_arr[order]
    del ts_arr, uid_arr, u_arr, v_arr, order
    gc.collect()

    return ts_sorted, uid_sorted, u_sorted, v_sorted


def get_observations_window(ts_arr, uid_arr, u_arr, v_arr, target_ts_ns, window_ns=OBS_WINDOW_NS):
    lo = np.searchsorted(ts_arr, target_ts_ns - window_ns, side="left")
    hi = np.searchsorted(ts_arr, target_ts_ns + window_ns, side="right")
    return uid_arr[lo:hi], u_arr[lo:hi], v_arr[lo:hi]


def get_frame_at_timestamp(vrs_reader, slam_stream_id, timestamp_ns):
    stream_id = StreamId(slam_stream_id)
    image_data, image_record = vrs_reader.get_image_data_by_time_ns(
        stream_id, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST,
    )
    return image_data.to_numpy_array()


def match_keypoints_to_uids(kps, window_uid, window_u, window_v, radius=MATCH_PIXEL_RADIUS):
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


def _rss_gb():
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        return float("nan")


def build_index(vrs_path, traj_csv, points_csv_gz, observations_csv_gz, slam_stream_id, out_path):
    orb = cv2.ORB_create(nfeatures=1500)

    print("Loading trajectory...")
    trajectory = load_trajectory(traj_csv)
    keyframes = sample_keyframe_timestamps(trajectory)
    print(f"Selected {len(keyframes)} candidate keyframes")
    del trajectory
    print(f"  RSS: {_rss_gb():.2f} GB")

    print("Loading semidense points...")
    uid_sorted, xyz_sorted = load_points_arrays(points_csv_gz)
    print(f"  RSS: {_rss_gb():.2f} GB")

    print("Loading observations file once...")
    ts_arr, uid_arr, u_arr, v_arr = load_observations_sorted(observations_csv_gz)
    print(f"Loaded {len(ts_arr)} observations")
    print(f"  RSS: {_rss_gb():.2f} GB")

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
            if uid is None:
                continue
            xyz = lookup_points(uid_sorted, xyz_sorted, uid)
            if xyz is not None:
                all_descs.append(desc)
                all_xyz.append(xyz)

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(keyframes)} keyframes, "
                  f"{len(all_descs)} pairs so far, RSS: {_rss_gb():.2f} GB")

    print(f"Indexed {len(all_descs)} descriptor->3D-point pairs from {len(keyframes)} keyframes")
    print(f"  RSS before save: {_rss_gb():.2f} GB")

    with open(out_path, "wb") as f:
        pickle.dump({
            "descriptors": np.array(all_descs, dtype=np.uint8),
            "xyz_map_frame": np.array(all_xyz, dtype=np.float64),  # kept float64 for localize.py / PnP precision
        }, f)

    print(f"Saved {out_path}")
    print(f"  final RSS: {_rss_gb():.2f} GB")


if __name__ == "__main__":
    build_index(
        vrs_path="/home/ayushi/aria_gen2/vrs_files/Corridor2.vrs",
        traj_csv="/home/ayushi/aria_gen2/vrs_files/mps_Corridor2_vrs/slam/closed_loop_trajectory.csv",
        points_csv_gz="/home/ayushi/aria_gen2/vrs_files/mps_Corridor2_vrs/slam/semidense_points.csv.gz",
        observations_csv_gz="/home/ayushi/aria_gen2/vrs_files/mps_Corridor2_vrs/slam/semidense_observations.csv.gz",
        slam_stream_id="1201-1",
        out_path="/home/ayushi/aria_gen2/scripts/reloc_index2.pkl",
    )