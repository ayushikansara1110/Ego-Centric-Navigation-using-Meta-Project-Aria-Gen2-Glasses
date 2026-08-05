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
DRIFT_REFRESH_INTERVAL_S = 20   # re-run PnP in the background this often (bootstrap/refresh path)


def load_index(path="/home/ayushi/aria_gen2/scripts/reloc_index.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_camera_matrix(cam_calib):
   
    fx, fy = cam_calib.get_focal_lengths()
    cx, cy = cam_calib.get_principal_point()
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def undistort_points_fisheye624(image_points, cam_calib, camera_matrix):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    undistorted = []
    valid_mask = []
    for pt in image_points:
        ray = cam_calib.unproject(np.array(pt, dtype=np.float64))
        if ray is None:
            valid_mask.append(False)
            continue
        x, y, z = ray
        if z <= 0:
            valid_mask.append(False)
            continue
        undistorted.append([fx * (x / z) + cx, fy * (y / z) + cy])
        valid_mask.append(True)

    return np.array(undistorted, dtype=np.float64), np.array(valid_mask, dtype=bool)


IMGIDX_LIMIT = 262144  # OpenCV BFMatcher's hard per-train-set row cap (1 << 18)

def _chunked_knn_match(bf, query_descs, train_descs, k=2, chunk_size=200000):
    """
    BFMatcher.knnMatch() hard-caps train descriptors at IMGIDX_LIMIT rows.
    Splits train_descs into chunks under that limit, matches against each,
    then merges to keep the true best-k matches per query descriptor across
    chunks (remapping trainIdx back to the original, unchunked index).
    """
    if len(train_descs) <= chunk_size:
        return bf.knnMatch(query_descs, train_descs, k=k)

    all_chunk_matches = []
    offsets = []
    for start in range(0, len(train_descs), chunk_size):
        chunk = train_descs[start:start + chunk_size]
        all_chunk_matches.append(bf.knnMatch(query_descs, chunk, k=k))
        offsets.append(start)

    merged = []
    for q_idx in range(len(query_descs)):
        candidates = []
        for chunk_idx, chunk_matches in enumerate(all_chunk_matches):
            for m in chunk_matches[q_idx]:
                m.trainIdx += offsets[chunk_idx]  # remap to global index space
                candidates.append(m)
        candidates.sort(key=lambda m: m.distance)
        merged.append(candidates[:k])
    return merged


def try_pnp(frame, index, camera_matrix, dist_coeffs, cam_calib=None):

    kps, descs = ORB.detectAndCompute(frame, None)
    if descs is None or len(descs) < MIN_INLIERS:
        return None

    matches = _chunked_knn_match(BF, descs, index["descriptors"], k=2)
    good = [m for m, n in matches if n is not None and m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None

    image_points = np.array([kps[m.queryIdx].pt for m in good], dtype=np.float64)
    object_points = np.array([index["xyz_map_frame"][m.trainIdx] for m in good], dtype=np.float64)

    if cam_calib is not None:
        image_points, valid_mask = undistort_points_fisheye624(image_points, cam_calib, camera_matrix)
        object_points = object_points[valid_mask]
        if len(image_points) < MIN_INLIERS:
            return None
        pnp_dist_coeffs = None
    else:
        pnp_dist_coeffs = dist_coeffs

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, pnp_dist_coeffs,
        reprojectionError=4.0, confidence=0.99
    )
    if not ok or inliers is None or len(inliers) < MIN_INLIERS:
        return None
    return rvec, tvec

def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T_cam_from_map = np.eye(4)
    T_cam_from_map[:3, :3] = R
    T_cam_from_map[:3, 3] = tvec.flatten()
    return np.linalg.inv(T_cam_from_map)  


class Tracker:
   
    def __init__(self, index, camera_matrix, dist_coeffs, cam_calib=None, T_device_from_cam=None):
        self.index = index
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cam_calib = cam_calib
        self.T_map_from_session = None
        self.T_device_from_cam = T_device_from_cam
        self._last_good_pose = None
        self._last_good_ts = 0.0
        self._lock = threading.Lock()
        self._stop = False

    def _try_pnp(self, frame):
        return try_pnp(frame, self.index, self.camera_matrix, self.dist_coeffs, cam_calib=self.cam_calib)

    # --- session-relative mode ---

    def bootstrap(self, get_frame_fn, get_session_pose_fn, timeout_s=5.0, retry_interval_s=0.3):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frame = get_frame_fn()
            result = self._try_pnp(frame)
            if result is not None:
                rvec, tvec = result
                T_map_from_cam = pose_to_matrix(rvec, tvec)
                T_map_from_device = T_map_from_cam @ np.linalg.inv(self.T_device_from_cam)
                T_session_from_device = get_session_pose_fn()
                with self._lock:
                    self.T_map_from_session = T_map_from_device @ np.linalg.inv(T_session_from_device)
                return True
            time.sleep(retry_interval_s)
        return False

    def live_map_pose(self, T_session_from_cam):
        with self._lock:
            if self.T_map_from_session is None:
                return None
            return self.T_map_from_session @ T_session_from_cam

    def start_background_refresh(self, get_frame_fn, get_session_pose_fn):
        def loop():
            while not self._stop:
                time.sleep(DRIFT_REFRESH_INTERVAL_S)
                frame = get_frame_fn()
                result = self._try_pnp(frame)
                if result is not None:
                    rvec, tvec = result
                    T_map_from_cam = pose_to_matrix(rvec, tvec)
                    T_map_from_device = T_map_from_cam @ np.linalg.inv(self.T_device_from_cam)
                    T_session_from_device = get_session_pose_fn()
                    with self._lock:
                        self.T_map_from_session = T_map_from_device @ np.linalg.inv(T_session_from_device)
        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._stop = True

    # --- direct PnP mode (no session pose required) ---

    def localize(self, frame):
        """
        Single-shot PnP against the map. Returns a 4x4 T_map_from_cam matrix
        directly. Returns None on failure -- caller decides fallback behavior.
        """
        result = self._try_pnp(frame)
        if result is None:
            return None
        rvec, tvec = result
        T_map_from_cam = pose_to_matrix(rvec, tvec)
        with self._lock:
            self._last_good_pose = T_map_from_cam
            self._last_good_ts = time.time()
        return T_map_from_cam

    def localize_or_last_known(self, frame, max_staleness_s=5.0):
        T = self.localize(frame)
        if T is not None:
            return T, True
        with self._lock:
            if self._last_good_pose is not None and (time.time() - self._last_good_ts) < max_staleness_s:
                return self._last_good_pose, False
            return None, False


def perpendicular_distance_and_heading_delta(pos_xy, heading_deg, seg_start_xy, seg_end_xy):
    seg = np.array(seg_end_xy) - np.array(seg_start_xy)
    seg_len = np.linalg.norm(seg)
    seg_dir = seg / seg_len
    to_pos = np.array(pos_xy) - np.array(seg_start_xy)
    perp_dist = abs(to_pos[0] * seg_dir[1] - to_pos[1] * seg_dir[0])

    seg_bearing = np.degrees(np.arctan2(seg_dir[1], seg_dir[0]))
    heading_delta = (heading_deg - seg_bearing + 180) % 360 - 180
    return perp_dist, heading_delta


def check_path_adherence(pos_xy, heading_deg, seg_start_xy, seg_end_xy):
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