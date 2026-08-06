import argparse
import json
import os
import numpy as np
from collections import defaultdict


def load_map(path):
    data = np.load(path, allow_pickle=True)
    return {
        "grid":      data["grid"],
        "x_min":     float(data["x_min"]),
        "y_min":     float(data["y_min"]),
        "cell_size": float(data["cell_size"]),
    }


def cell_to_world(ri, ci, x_min, y_min, cs):
    return np.array([x_min + (ci + 0.5) * cs, y_min + (ri + 0.5) * cs])


def world_to_cell(wx, wy, x_min, y_min, cs, rows, cols):
    ci = int((wx - x_min) / cs)
    ri = int((wy - y_min) / cs)
    return (max(0, min(rows - 1, ri)), max(0, min(cols - 1, ci)))


def bresenham_clear(grid, r0, c0, r1, c1):
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0
    err = dr - dc
    while True:
        if grid[r, c] == 1:
            return False
        if r == r1 and c == c1:
            return True
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 < dr:  err += dr; c += sc


def extract_waypoints(m, spacing):
    grid = m["grid"]
    cs   = m["cell_size"]
    traj_cells = np.argwhere(grid == 2)
    if len(traj_cells) == 0:
        raise ValueError("No trajectory cells (value=2) found in grid.")
    world = np.array([cell_to_world(r, c, m["x_min"], m["y_min"], cs)
                      for r, c in traj_cells])
    step_sq = spacing ** 2
    kept = [world[0]]
    for pt in world[1:]:
        if np.sum((pt - kept[-1]) ** 2) >= step_sq:
            kept.append(pt)
    waypoints = np.array(kept)
    print(f"  {len(traj_cells):,} trajectory cells → {len(waypoints)} waypoints at {spacing} m spacing")
    return waypoints


def build_edges(waypoints, m, max_edge_m):
    grid = m["grid"]
    cs   = m["cell_size"]
    rows, cols = grid.shape
    edges = []
    for i in range(len(waypoints)):
        for j in range(i + 1, len(waypoints)):
            dist = float(np.linalg.norm(waypoints[i] - waypoints[j]))
            if dist > max_edge_m:
                continue
            r0, c0 = world_to_cell(*waypoints[i], m["x_min"], m["y_min"], cs, rows, cols)
            r1, c1 = world_to_cell(*waypoints[j], m["x_min"], m["y_min"], cs, rows, cols)
            if bresenham_clear(grid, r0, c0, r1, c1):
                edges.append((i, j, dist))
    print(f"  {len(edges)} edges (max_edge={max_edge_m} m)")
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map",      required=True)
    parser.add_argument("--outdir",   default="./maps")
    parser.add_argument("--spacing",  type=float, default=0.5)
    parser.add_argument("--max_edge", type=float, default=2.0)
    parser.add_argument("--labels",   default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n=== Building topological graph ===")

    m         = load_map(args.map)
    waypoints = extract_waypoints(m, args.spacing)
    edges     = build_edges(waypoints, m, args.max_edge)

    labels = {}
    if args.labels and os.path.exists(args.labels):
        with open(args.labels) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if isinstance(v, dict):
                lbl = v.get("label", "").strip()
            else:
                lbl = str(v).strip()
            if lbl:
                labels[int(k)] = lbl
        print(f"  {len(labels)} semantic labels loaded")

    label_array = np.array([labels.get(i, "") for i in range(len(waypoints))], dtype=object)

    adj = defaultdict(list)
    for i, j, d in edges:
        adj[i].append((j, d))
        adj[j].append((i, d))

    adj_obj = np.empty(len(waypoints), dtype=object)
    for i in range(len(waypoints)):
        adj_obj[i] = adj[i]

    edge_arr = np.array([(i, j, d) for i, j, d in edges],
                        dtype=[("i", np.int32), ("j", np.int32), ("dist", np.float32)])

    out_path = os.path.join(args.outdir, "graph.npz")
    np.savez_compressed(out_path,
                        waypoints=waypoints,
                        edges=edge_arr,
                        labels=label_array,
                        adj=adj_obj)

    summary_path = os.path.join(args.outdir, "graph_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "num_nodes": len(waypoints),
            "num_edges": len(edges),
            "labeled_nodes": {str(k): v for k, v in labels.items()},
            "bbox_world": {
                "x_min": float(waypoints[:, 0].min()),
                "x_max": float(waypoints[:, 0].max()),
                "y_min": float(waypoints[:, 1].min()),
                "y_max": float(waypoints[:, 1].max()),
            }
        }, f, indent=2)

    print(f"\n  saved: {out_path}")
    print(f"  saved: {summary_path}")
    print(f"  {len(waypoints)} nodes, {len(edges)} edges")


if __name__ == "__main__":
    main()
