"""
python navigate.py \
    --graph   ./maps/graph.npz \
    --start_xy 1.2 3.4 \
    --goal_xy  8.7 2.1 \
    --outdir  ./maps
"""

import argparse
import heapq
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt

def astar(adj: np.ndarray, waypoints: np.ndarray,
          start: int, goal: int) -> list[int] | None:
    """
    A* on the adjacency graph.
    adj[i] = list of (neighbour_id, dist) tuples.
    Returns list of node ids from start to goal, or None if unreachable.
    """
    def h(n):
        return float(np.linalg.norm(waypoints[n] - waypoints[goal]))

    open_heap = [(h(start), 0.0, start)]
    came_from = {}
    g_score   = {start: 0.0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        if g > g_score.get(current, float("inf")):
            continue

        for neighbour, dist in adj[current]:
            tentative = g_score[current] + dist
            if tentative < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative
                came_from[neighbour] = current
                f = tentative + h(neighbour)
                heapq.heappush(open_heap, (f, tentative, neighbour))

    return None   

def bearing_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Bearing from point a to point b in degrees (0° = +X axis, CCW positive)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def turn_description(delta_deg: float) -> str | None:
    """Convert a heading change in degrees to a human instruction."""
    d = delta_deg % 360
    if d > 180:
        d -= 360
    if abs(d) < 15:
        return None                        # go straight, no instruction
    if d > 120:
        return "turn around"
    if d > 45:
        return "turn left"
    if d > 15:
        return "bear left"
    if d < -120:
        return "turn around"
    if d < -45:
        return "turn right"
    if d < -15:
        return "bear right"
    return None


def generate_instructions(path: list[int],
                           waypoints: np.ndarray,
                           labels: np.ndarray) -> list[str]:
    if len(path) < 2:
        return ["You are already at your destination."]

    instructions = []
    prev_bearing  = None
    segment_start = 0
    segment_dist  = 0.0

    for i in range(1, len(path)):
        a = waypoints[path[i - 1]]
        b = waypoints[path[i]]
        d = float(np.linalg.norm(b - a))
        segment_dist += d
        curr_bearing  = bearing_deg(a, b)

        if prev_bearing is not None:
            delta = curr_bearing - prev_bearing
            turn  = turn_description(delta)
            if turn:
                if segment_dist - d > 0.1:
                    instructions.append(
                        f"Walk straight for {segment_dist - d:.1f} metres."
                    )
                instructions.append(turn.capitalize() + ".")
                segment_dist = d   

        label = str(labels[path[i]])
        if label and label != "":
            instructions.append(f"You are approaching: {label}.")

        prev_bearing = curr_bearing

    if segment_dist > 0.1:
        instructions.append(f"Walk straight for {segment_dist:.1f} metres.")

    goal_label = str(labels[path[-1]])
    if goal_label and goal_label != "":
        instructions.append(f"You have arrived at: {goal_label}.")
    else:
        instructions.append("You have arrived at your destination.")

    return instructions

def load_graph(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "waypoints": data["waypoints"],
        "labels":    data["labels"],
        "adj":       data["adj"],
    }

def nearest_node(waypoints: np.ndarray, xy: np.ndarray) -> int:
    return int(np.linalg.norm(waypoints - xy, axis=1).argmin())

def speak(instructions: list[str]) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        for line in instructions:
            engine.say(line)
        engine.runAndWait()
    except ImportError:
        print("  pyttsx3 not installed — skipping speech output.")
    except Exception as e:
        print(f"  TTS error: {e}")


def visualize_path(waypoints, path, start, goal):

    plt.figure(figsize=(8,8))

    plt.plot(
        waypoints[:,0],
        waypoints[:,1],
        color="lightgray",
        linewidth=2
    )

    p = waypoints[path]

    plt.plot(
        p[:,0],
        p[:,1],
        linewidth=4,
        color="red"
    )

    plt.scatter(
        p[0,0],
        p[0,1],
        s=180,
        c="green",
        label="Start"
    )

    plt.scatter(
        p[-1,0],
        p[-1,1],
        s=180,
        c="blue",
        label="Goal"
    )

    for i,node in enumerate(path):
        plt.text(
            p[i,0],
            p[i,1],
            str(node),
            fontsize=8
        )

    plt.axis("equal")
    plt.legend()
    plt.title("Planned Route")
    plt.savefig("planned_route.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="A* navigation + audio instructions")
    parser.add_argument("--graph",    required=True)
    parser.add_argument("--start",    type=int,   default=None, help="start node id")
    parser.add_argument("--goal",     type=int,   default=None, help="goal node id")
    parser.add_argument("--start_xy", type=float, nargs=2, default=None,
                        help="start world XY in metres (nearest node used)")
    parser.add_argument("--goal_xy",  type=float, nargs=2, default=None,
                        help="goal world XY in metres (nearest node used)")
    parser.add_argument("--outdir",   default="./maps")
    parser.add_argument("--speak",    action="store_true",
                        help="speak instructions aloud via pyttsx3")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    graph     = load_graph(args.graph)
    waypoints = graph["waypoints"]
    labels    = graph["labels"]
    adj       = graph["adj"]

    if args.start_xy:
        start = nearest_node(waypoints, np.array(args.start_xy))
    elif args.start is not None:
        start = args.start
    else:
        raise ValueError("Provide either --start or --start_xy")

    if args.goal_xy:
        goal = nearest_node(waypoints, np.array(args.goal_xy))
    elif args.goal is not None:
        goal = args.goal
    else:
        raise ValueError("Provide either --goal or --goal_xy")

    print(f"\n=== Navigation ===")
    print(f"  start node: {start}  {waypoints[start]}")
    print(f"  goal  node: {goal}   {waypoints[goal]}")

    path = astar(adj, waypoints, start, goal)

    if path is None:
        print("  No path found — graph may be disconnected.")
        return

    total_dist = sum(
        float(np.linalg.norm(waypoints[path[i]] - waypoints[path[i-1]]))
        for i in range(1, len(path))
    )
    print(f"  path: {len(path)} nodes, {total_dist:.1f} m total")

    instructions = generate_instructions(path, waypoints, labels)

    print("\n--- Navigation instructions ---")
    for step, line in enumerate(instructions, 1):
        print(f"  {step:>2}. {line}")

    result = {
        "start_node":     start,
        "goal_node":      goal,
        "path_nodes":     path,
        "total_dist_m":   round(total_dist, 2),
        "instructions":   instructions,
    }
    nav_path = os.path.join(args.outdir, "navigation_result.json")
    with open(nav_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  saved: {nav_path}")

    instr_path = os.path.join(args.outdir, "instructions.txt")
    with open(instr_path, "w") as f:
        f.write("\n".join(instructions) + "\n")
    print(f"  saved: {instr_path}  (ready for TTS)")

    if args.speak:
        print("\n  speaking instructions …")
        speak(instructions)


if __name__ == "__main__":
    main()
