"""
python build_map.py \
    --points  "/home/ayushi/aria_gen2/vrs_files/new/mps_Corridor_vrs/slam/semidense_points.csv.gz"
    --traj    "/home/ayushi/aria_gen2/vrs_files/new/mps_Corridor_vrs/slam/closed_loop_trajectory.csv"
    --session csb_corridor \
    --outdir  "/home/ayushi/aria_gen2/maps" \
    --cell_size 0.05          # metres per grid cell (default 5 cm)
    --min_quality 0.0         # filter points below this quality (0=keep all)
    --floor_margin 0.1        # metres below trajectory centroid considered floor
    --ceil_margin  2.0        # metres above trajectory centroid considered ceiling
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation



def load_points(path: str, min_quality: float = 0.0) -> np.ndarray:
    """Load semidense point cloud, return (N,3) float32 in world frame."""
    print(f"  loading points from {path} …")
    df = pd.read_csv(path)
    if "inverse_distance_std" in df.columns and min_quality > 0:
        df = df[df["inverse_distance_std"] >= min_quality]
    pts = df[["px_world", "py_world", "pz_world"]].to_numpy(dtype=np.float32)
    print(f"  {len(pts):,} points loaded")
    return pts


def load_trajectory(path: str) -> np.ndarray:
    """Load closed-loop trajectory, return (N,3) float32 in world frame."""
    print(f"  loading trajectory from {path} …")
    df = pd.read_csv(path)
    traj = df[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy(dtype=np.float32)
    print(f"  {len(traj):,} pose samples loaded")
    return traj


def compute_floor_band(traj: np.ndarray,
                       floor_margin: float,
                       ceil_margin: float) -> tuple[float, float]:
    """
    Estimate the vertical (Z) band that corresponds to the walkable floor layer.
    Aria glasses sit ~1.5 m above the floor; trajectory Z is roughly head height.
    We keep points within [traj_z_min - floor_margin, traj_z_min + ceil_margin].
    """
    z_min = float(np.percentile(traj[:, 2], 5))  
    floor_z = z_min - floor_margin
    ceil_z  = z_min + ceil_margin
    print(f"  Z band: {floor_z:.2f} m → {ceil_z:.2f} m")
    return floor_z, ceil_z


def build_occupancy_grid(pts: np.ndarray,
                         traj: np.ndarray,
                         cell_size: float,
                         floor_z: float,
                         ceil_z: float) -> dict:
    mask = (pts[:, 2] >= floor_z) & (pts[:, 2] <= ceil_z)
    pts2d = pts[mask, :2]

    padding = 2.0
    traj_xy = traj[:, :2]
    x_min = traj_xy[:, 0].min() - padding
    x_max = traj_xy[:, 0].max() + padding
    y_min = traj_xy[:, 1].min() - padding
    y_max = traj_xy[:, 1].max() + padding

    in_bounds = (
        (pts2d[:, 0] >= x_min) & (pts2d[:, 0] <= x_max) &
        (pts2d[:, 1] >= y_min) & (pts2d[:, 1] <= y_max)
    )
    pts2d = pts2d[in_bounds]
    print(f"  {in_bounds.sum():,} / {len(in_bounds):,} points within trajectory bounds")

    cols = int(np.ceil((x_max - x_min) / cell_size))
    rows = int(np.ceil((y_max - y_min) / cell_size))
    print(f"  grid size: {rows} rows × {cols} cols  ({rows*cell_size:.1f} m × {cols*cell_size:.1f} m)")

    grid = np.zeros((rows, cols), dtype=np.uint8)

    xi = np.clip(((pts2d[:, 0] - x_min) / cell_size).astype(int), 0, cols - 1)
    yi = np.clip(((pts2d[:, 1] - y_min) / cell_size).astype(int), 0, rows - 1)
    grid[yi, xi] = 1

    grid = (binary_dilation(grid, iterations=1).astype(np.uint8))

    tx = np.clip(((traj[:, 0] - x_min) / cell_size).astype(int), 0, cols - 1)
    ty = np.clip(((traj[:, 1] - y_min) / cell_size).astype(int), 0, rows - 1)
    grid[ty, tx] = 2

    return {
        "grid":      grid,
        "x_min":     x_min,
        "y_min":     y_min,
        "cell_size": cell_size,
        "rows":      rows,
        "cols":      cols,
    }

def main():
    parser = argparse.ArgumentParser(description="Build 2-D occupancy map from Aria MPS outputs")
    parser.add_argument("--points",       required=True,  help="path to semidense_points.csv.gz")
    parser.add_argument("--traj",         required=True,  help="path to closed_loop_trajectory.csv")
    parser.add_argument("--session",      required=True,  help="session name tag, e.g. walk_0")
    parser.add_argument("--outdir",       default="./maps")
    parser.add_argument("--cell_size",    type=float, default=0.5) #50 cm 
    parser.add_argument("--min_quality",  type=float, default=0.0)
    parser.add_argument("--floor_margin", type=float, default=0.1)
    parser.add_argument("--ceil_margin",  type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n=== Building map for session: {args.session} ===")

    pts  = load_points(args.points, args.min_quality)
    traj = load_trajectory(args.traj)

    floor_z, ceil_z = compute_floor_band(traj, args.floor_margin, args.ceil_margin)
    map_data = build_occupancy_grid(pts, traj, args.cell_size, floor_z, ceil_z)

    map_path  = os.path.join(args.outdir, f"{args.session}_map.npz")
    traj_path = os.path.join(args.outdir, f"{args.session}_traj.npy")

    np.savez_compressed(map_path,
                        grid=map_data["grid"],
                        x_min=np.array(map_data["x_min"]),
                        y_min=np.array(map_data["y_min"]),
                        cell_size=np.array(map_data["cell_size"]),
                        session=np.array(args.session))
    np.save(traj_path, traj)

    print(f"\n  saved: {map_path}")
    print(f"  saved: {traj_path}")
    print(f"\nDone. Grid has {int((map_data['grid'] == 1).sum()):,} obstacle cells "
          f"and {int((map_data['grid'] == 2).sum()):,} trajectory cells.")


if __name__ == "__main__":
    main()
