"""
# just map + graph
python visualize_graph.py \
    --map    "/home/ayushi/aria_gen2/maps/csb_corridor_map.npz" \
    --graph  "/home/ayushi/aria_gen2/graphs/graph.npz" \
    --out    "./maps/graph_view.png"

# map + graph + navigation path
python visualize_graph.py \
    --map    "/home/ayushi/aria_gen2/maps/merged_map.npz" \
    --graph  "/home/ayushi/aria_gen2/graphs/graph.npz" \
    --nav    "/home/ayushi/aria_gen2/maps/navigation_result.json" \
    --out    "./maps/nav_view.png"

# show interactively 
python visualize_graph.py --map "/home/ayushi/aria_gen2/maps/merged_map.npz" --graph "/home/ayushi/aria_gen2/graphs/graph.npz" --show
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless by default; overridden by --show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def load_map(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "grid":      data["grid"],
        "x_min":     float(data["x_min"]),
        "y_min":     float(data["y_min"]),
        "cell_size": float(data["cell_size"]),
    }


def load_graph(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "waypoints": data["waypoints"],
        "labels":    data["labels"],
        "edges":     data["edges"],
    }


def grid_extent(m: dict) -> list[float]:
    """Return [x_min, x_max, y_min, y_max] for imshow extent."""
    r, c = m["grid"].shape
    cs   = m["cell_size"]
    return [m["x_min"], m["x_min"] + c * cs,
            m["y_min"], m["y_min"] + r * cs]


def plot(m: dict, graph: dict,
         nav_path: list[int] | None,
         out_path: str,
         show: bool) -> None:

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    cmap = ListedColormap(["#16213e",   # 0 free
                            "#e0e0e0",  # 1 obstacle
                            "#0f3460"])  # 2 trajectory
    ext = grid_extent(m)
    ax.imshow(m["grid"],
              origin="lower",
              extent=ext,
              cmap=cmap,
              vmin=0, vmax=2,
              interpolation="nearest",
              zorder=1)

    wp = graph["waypoints"]
    for edge in graph["edges"]:
        i, j = int(edge["i"]), int(edge["j"])
        ax.plot([wp[i, 0], wp[j, 0]],
                [wp[i, 1], wp[j, 1]],
                color="#533483", linewidth=0.6, alpha=0.5, zorder=2)

    ax.scatter(wp[:, 0], wp[:, 1],
               s=8, color="#e94560", zorder=3, alpha=0.8, label="Waypoints")

    labels = graph["labels"]
    for i, label in enumerate(labels):
        if str(label) and str(label) != "":
            ax.annotate(str(label),
                        (wp[i, 0], wp[i, 1]),
                        fontsize=6, color="white",
                        xytext=(4, 4), textcoords="offset points",
                        zorder=5)

    if nav_path and len(nav_path) >= 2:
        px = [wp[n, 0] for n in nav_path]
        py = [wp[n, 1] for n in nav_path]
        ax.plot(px, py, color="#f5a623", linewidth=2.5, zorder=4,
                solid_capstyle="round", label="Navigation path")
        ax.scatter([px[0]],  [py[0]],  s=120, color="#2ecc71", zorder=6,
                   marker="*", label="Start")
        ax.scatter([px[-1]], [py[-1]], s=120, color="#e74c3c", zorder=6,
                   marker="X", label="Goal")

    ax.set_xlabel("World X (m)", color="white")
    ax.set_ylabel("World Y (m)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_title("Aria Indoor Navigation — Occupancy Grid + Waypoint Graph",
                 color="white", fontsize=12)

    legend_patches = [
        mpatches.Patch(color="#e0e0e0", label="Obstacle"),
        mpatches.Patch(color="#0f3460", label="Trajectory band"),
        mpatches.Patch(color="#e94560", label="Waypoints"),
    ]
    if nav_path:
        legend_patches += [
            mpatches.Patch(color="#f5a623", label="Path"),
            mpatches.Patch(color="#2ecc71", label="Start"),
            mpatches.Patch(color="#e74c3c", label="Goal"),
        ]
    ax.legend(handles=legend_patches, loc="upper right",
              facecolor="#16213e", edgecolor="#444", labelcolor="white",
              fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  saved: {out_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize occupancy grid + waypoint graph")
    parser.add_argument("--map",   required=True,  help="merged_map.npz or <session>_map.npz")
    parser.add_argument("--graph", required=True,  help="graph.npz")
    parser.add_argument("--nav",   default=None,   help="navigation_result.json (optional)")
    parser.add_argument("--out",   default=None,   help="output PNG path (default: <map_dir>/graph_view.png)")
    parser.add_argument("--show",  action="store_true", help="display interactively")
    args = parser.parse_args()

    m     = load_map(args.map)
    graph = load_graph(args.graph)

    nav_path = None
    if args.nav and os.path.exists(args.nav):
        with open(args.nav) as f:
            nav_data = json.load(f)
        nav_path = nav_data.get("path_nodes")

    out = args.out or os.path.join(os.path.dirname(args.map), "graph_view.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"\n=== Visualizing ===")
    print(f"  grid:  {m['grid'].shape}  |  {len(graph['waypoints'])} nodes  |  {len(graph['edges'])} edges")
    plot(m, graph, nav_path, out, args.show)


if __name__ == "__main__":
    main()
