import pandas as pd
import matplotlib.pyplot as plt

traj = pd.read_csv(
    "/home/ayushi/aria_gen2/projectaria_tools_gen2pilot_data/"
    "walk_0/mps/slam/closed_loop_trajectory.csv"
)

x = traj["tx_world_device"]
y = traj["ty_world_device"]

plt.figure(figsize=(10,10))
plt.plot(x, y)# scripts/map_stats.py

import gzip
import csv
import numpy as np

POINTS_FILE = (
    "/home/ayushi/aria_gen2/projectaria_tools_gen2pilot_data/"
    "walk_0/mps/slam/semidense_points.csv.gz"
)

pts = []

with gzip.open(POINTS_FILE, "rt") as f:
    reader = csv.DictReader(f)

    for i,row in enumerate(reader):

        if i % 10000 != 0:
            continue

        pts.append([
            float(row["px_world"]),
            float(row["py_world"]),
            float(row["pz_world"])
        ])

pts = np.array(pts)

print("x:", np.percentile(pts[:,0],[1,5,50,95,99]))
print("y:", np.percentile(pts[:,1],[1,5,50,95,99]))
print("z:", np.percentile(pts[:,2],[1,5,50,95,99]))
plt.axis("equal")

plt.title("walk_0 MPS trajectory")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.savefig("walk0_trajectory.png", dpi=300)
print("saved walk0_trajectory.png")# scripts/map_stats.py

import gzip
import csv
import numpy as np

POINTS_FILE = (
    "/home/ayushi/aria_gen2/projectaria_tools_gen2pilot_data/"
    "walk_0/mps/slam/semidense_points.csv.gz"
)

pts = []

with gzip.open(POINTS_FILE, "rt") as f:
    reader = csv.DictReader(f)

    for i,row in enumerate(reader):

        if i % 10000 != 0:
            continue

        pts.append([
            float(row["px_world"]),
            float(row["py_world"]),
            float(row["pz_world"])
        ])

pts = np.array(pts)

print("x:", np.percentile(pts[:,0],[1,5,50,95,99]))
print("y:", np.percentile(pts[:,1],[1,5,50,95,99]))
print("z:", np.percentile(pts[:,2],[1,5,50,95,99]))