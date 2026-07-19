"""
localize.py
Real-time. Two jobs:

1. Bootstrap (runs ONCE at session start): match the first live frame against
   reloc_index.pkl, solvePnPRansac -> transform from this session's SLAM world
   frame to the map's world frame.

2. Live tracking (runs every frame, cheap): apply that cached transform to
   Aria's own live fused pose (Aria's on-device SLAM already fuses IMU + visual
   odometry for you — you don't hand-fuse raw IMU yourself, you just consume
   its continuous 6DoF pose stream). Also checks whether the person is
   drifting off the planned path.
"""

import pickle
import threading
import time
import numpy as np
import cv2

ORB = cv2.ORB_create(nfeatures=1000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

MIN_INLIERS = 6
LOWE_RATIO = 0.75
OFF_PATH_THRESHOLD_M = 0.6      # perpendicular distance from path before we warn
HEADING_DELTA_THRESHOLD_DEG = 25
DRIFT_REFRESH_INTERVAL_S = 20   # re-run PnP in the background this often


def load_index(path="maps/reloc_index.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def try_pnp(frame, index, camera_matrix, dist_coeffs):
    """Single-frame relocalization attempt. Returns (rvec, tvec) or None."""
    kps, descs = ORB.detectAndCompute(frame, None)
    if descs is None or len(descs) < MIN_INLIERS:
        return None

    matches = BF.knnMatch(descs, index["descriptors"], k=2)
    good = [m for m, n in matches if n is not None and m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None

    image_points = np.array([kps[m.queryIdx].pt for m in good], dtype=np.float64)
    object_points = np.array([index["xyz_map_frame"][m.trainIdx] for m in good], dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, dist_coeffs,
        reprojectionError=4.0, confidence=0.99
    )
    if not ok or inliers is None or len(inliers) < MIN_INLIERS:
        return None
    return rvec, tvec


def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


class Tracker:
    """
    Holds the cached session->map transform and answers "where am I in map
    frame right now" every frame for near-zero cost. Refreshes the transform
    in the background periodically to correct drift.
    """

    def __init__(self, index, camera_matrix, dist_coeffs):
        self.index = index
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.T_map_from_session = None
        self._lock = threading.Lock()
        self._stop = False

    def bootstrap(self, get_frame_fn, get_session_pose_fn, timeout_s=5.0, retry_interval_s=0.3):
        """
        Blocking. Call this once at startup — keep asking for frames (standing
        still is fine, no walking needed) until PnP succeeds or timeout hits.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frame = get_frame_fn()
            result = try_pnp(frame, self.index, self.camera_matrix, self.dist_coeffs)
            if result is not None:
                rvec, tvec = result
                T_map_from_cam = pose_to_matrix(rvec, tvec)
                T_session_from_cam = get_session_pose_fn()  # Aria's own live pose at this instant
                with self._lock:
                    self.T_map_from_session = T_map_from_cam @ np.linalg.inv(T_session_from_cam)
                return True
            time.sleep(retry_interval_s)
        return False  # caller should fall back to a "please walk forward a few steps" prompt

    def live_map_pose(self, T_session_from_cam):
        """Cheap: one matrix multiply, called every frame."""
        with self._lock:
            if self.T_map_from_session is None:
                return None
            return self.T_map_from_session @ T_session_from_cam

    def start_background_refresh(self, get_frame_fn, get_session_pose_fn):
        def loop():
            while not self._stop:
                time.sleep(DRIFT_REFRESH_INTERVAL_S)
                frame = get_frame_fn()
                result = try_pnp(frame, self.index, self.camera_matrix, self.dist_coeffs)
                if result is not None:
                    rvec, tvec = result
                    T_map_from_cam = pose_to_matrix(rvec, tvec)
                    T_session_from_cam = get_session_pose_fn()
                    with self._lock:
                        self.T_map_from_session = T_map_from_cam @ np.linalg.inv(T_session_from_cam)
        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._stop = True


def perpendicular_distance_and_heading_delta(pos_xy, heading_deg, seg_start_xy, seg_end_xy):
    """
    Path-adherence check. pos_xy: current (x,y) in map frame. heading_deg:
    current facing direction. seg_start/end: the graph edge the person should
    currently be walking along (from your existing A* path).
    Returns (perpendicular_distance_m, heading_delta_deg).
    """
    seg = np.array(seg_end_xy) - np.array(seg_start_xy)
    seg_len = np.linalg.norm(seg)
    seg_dir = seg / seg_len
    to_pos = np.array(pos_xy) - np.array(seg_start_xy)
    perp_dist = abs(to_pos[0] * seg_dir[1] - to_pos[1] * seg_dir[0])

    seg_bearing = np.degrees(np.arctan2(seg_dir[1], seg_dir[0]))
    heading_delta = (heading_deg - seg_bearing + 180) % 360 - 180
    return perp_dist, heading_delta


def check_path_adherence(pos_xy, heading_deg, seg_start_xy, seg_end_xy):
    """Returns None if on track, else a short correction string."""
    perp_dist, heading_delta = perpendicular_distance_and_heading_delta(
        pos_xy, heading_deg, seg_start_xy, seg_end_xy
    )
    if perp_dist > OFF_PATH_THRESHOLD_M:
        side = "right" if (heading_delta > 0) else "left"
        return f"You've drifted off path, move slightly {side}"
    if abs(heading_delta) > HEADING_DELTA_THRESHOLD_DEG:
        side = "right" if heading_delta < 0 else "left"
        return f"Turn slightly {side} to stay on course"
    return None
