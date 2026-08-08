# plot_graph_report.py

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

GRAPH_PATH = "/home/ayushi/aria_gen2/graph2/new/graph.npz"
LABELS_PATH = "/home/ayushi/aria_gen2/scripts/labels2.json"

FIGURE_PATH = "/home/ayushi/aria_gen2/scripts/navigation_graph_report.png"
LEGEND_PATH = "/home/ayushi/aria_gen2/scripts/navigation_graph_legend.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

graph = np.load(GRAPH_PATH, allow_pickle=True)

print("Graph contents:", graph.files)

waypoints = graph["waypoints"]
edges = graph["edges"]

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# --------------------------------------------------
# FIGURE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(20, 8))

# --------------------------------------------------
# DRAW EDGES
# --------------------------------------------------

for edge in edges:
    i = int(edge[0])
    j = int(edge[1])

    ax.plot(
        [waypoints[i, 0], waypoints[j, 0]],
        [waypoints[i, 1], waypoints[j, 1]],
        color="lightgray",
        linewidth=0.8,
        alpha=0.7,
        zorder=1,
    )

# --------------------------------------------------
# OPTIONAL: draw all graph nodes
# --------------------------------------------------

ax.scatter(
    waypoints[:, 0],
    waypoints[:, 1],
    s=8,
    c="black",
    alpha=0.35,
    zorder=2,
)

# --------------------------------------------------
# DRAW DESTINATIONS
# --------------------------------------------------

legend_rows = []
dest_num = 1

for node_id_str, info in labels.items():

    label = info.get("label", "").strip()

    if not label:
        continue

    node_id = int(node_id_str)

    if node_id >= len(waypoints):
        continue

    x = waypoints[node_id, 0]
    y = waypoints[node_id, 1]

    ax.scatter(
        x,
        y,
        s=300,
        color="red",
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
    )

    ax.text(
        x,
        y,
        str(dest_num),
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=11,
    )

    legend_rows.append(
        {
            "ID": dest_num,
            "Destination": label,
        }
    )

    dest_num += 1

# --------------------------------------------------
# STYLING
# --------------------------------------------------

ax.set_title(
    "Indoor Navigation Graph",
    fontsize=18,
    fontweight="bold",
)

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

ax.grid(True, alpha=0.3)
ax.axis("equal")

plt.tight_layout()

# --------------------------------------------------
# SAVE FIGURE
# --------------------------------------------------

plt.savefig(
    FIGURE_PATH,
    dpi=600,
    bbox_inches="tight",
)

print(f"\nSaved graph figure:\n{FIGURE_PATH}")

# --------------------------------------------------
# SAVE LEGEND
# --------------------------------------------------

legend_df = pd.DataFrame(legend_rows)

legend_df.to_csv(
    LEGEND_PATH,
    index=False,
)

print(f"Saved legend CSV:\n{LEGEND_PATH}")

plt.show()