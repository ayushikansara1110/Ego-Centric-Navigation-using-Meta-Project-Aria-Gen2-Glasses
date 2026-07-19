# export_small_map.py

import pandas as pd

pts = pd.read_csv(
    "/home/ayushi/aria_gen2/vrs_files/new/mps_Corridor_vrs/slam/semidense_points.csv.gz"
)


with open("corridor_map.xyz", "w") as f:
    for _, row in pts.iterrows():
        f.write(
            f"{row.px_world} "
            f"{row.py_world} "
            f"{row.pz_world}\n"
        )

print("saved corridor_map.xyz")