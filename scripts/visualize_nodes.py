"""
python visualize_nodes.py \
    --map   /home/ayushi/aria_gen2/maps/csb_corridor_map.npz \
    --graph  /home/ayushi/aria_gen2/graphs/graph.npz \
    --out   /home/ayushi/aria_gen2/graphs/nodes_numbered.png 
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map",   required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out",   required=True)
    args = parser.parse_args()

    # load
    m     = np.load(args.map,   allow_pickle=True)
    g     = np.load(args.graph, allow_pickle=True)
    grid  = m["grid"]
    wp    = g["waypoints"]
    x_min = float(m["x_min"])
    y_min = float(m["y_min"])
    cs    = float(m["cell_size"])

    r, c  = grid.shape
    extent = [x_min, x_min + c * cs, y_min, y_min + r * cs]

    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # grid
    cmap = ListedColormap(["#16213e", "#e0e0e0", "#0f3460"])
    ax.imshow(grid, origin="lower", extent=extent,
              cmap=cmap, vmin=0, vmax=2, interpolation="nearest", zorder=1)

    # edges
    edges = g["edges"]
    for edge in edges:
        i, j = int(edge["i"]), int(edge["j"])
        ax.plot([wp[i,0], wp[j,0]], [wp[i,1], wp[j,1]],
                color="#533483", linewidth=0.8, alpha=0.5, zorder=2)

    # nodes with numbers
    ax.scatter(wp[:, 0], wp[:, 1],
               s=40, color="#e94560", zorder=4, alpha=0.9)

    for i in range(len(wp)):
        ax.annotate(str(i),
                    (wp[i, 0], wp[i, 1]),
                    fontsize=5,
                    color="white",
                    fontweight="bold",
                    ha="center", va="bottom",
                    xytext=(0, 4),
                    textcoords="offset points",
                    zorder=5)

    ax.set_xlabel("World X (m)", color="white")
    ax.set_ylabel("World Y (m)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_title(f"Node Map — {len(wp)} waypoints  |  identify node IDs before labelling",
                 color="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"saved: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
