import heapq
import json
import math
import pickle

import numpy as np

REACHED_WAYPOINT_THRESHOLD_M = 1  # how close counts as "arrived" at a waypoint
NEARBY_LABEL_THRESHOLD_M = 3.0       # how close a node must be to a labeled point of interest


class Node:
    
    __slots__ = ("id", "xy")

    def __init__(self, node_id, xy):
        self.id = node_id
        self.xy = np.asarray(xy, dtype=np.float64)

    def __repr__(self):
        return f"Node({self.id}, xy={self.xy.tolist()})"


class Graph:
    def __init__(self, waypoints, labels, adj):
        # waypoints: (N, 2) array of world xy per node id
        # labels: (N,) array of per-node label strings ("" if none)
        # adj: list where adj[i] = list of (neighbour_id, dist) tuples
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=object)
        self.adj = adj

    def node(self, node_id):
        return Node(node_id, self.waypoints[node_id])

    def __len__(self):
        return len(self.waypoints)


def load_graph(path):
    
    path = str(path)
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
    return Graph(data["waypoints"], data["labels"], data["adj"])
    return Graph(data["waypoints"], data["labels"], data["adj"])


def load_labels(path):
    
    with open(path, "r") as f:
        raw = json.load(f)
    return {v["label"]: int(k) for k, v in raw.items() if v.get("label")}


def node_for_label(label, labels):
    if label not in labels:
        raise KeyError(f"'{label}' not found in labels.json -- available: {list(labels.keys())}")
    return labels[label]


def nearest_node(pos_xy, graph):
    pos_xy = np.asarray(pos_xy, dtype=np.float64)
    dists = np.linalg.norm(graph.waypoints - pos_xy, axis=1)
    return int(dists.argmin())

def nearest_route_index(pos_xy, path, current_edge_idx=0, lookahead=2):
    """
    Find the route node nearest to the user's actual position.

    Only searches forward from the current route position so localization
    noise cannot easily send navigation backwards along the route.
    """
    if path is None or len(path) == 0:
        return current_edge_idx

    pos_xy = np.asarray(pos_xy, dtype=np.float64)

    start = max(0, current_edge_idx)
    end = min(len(path), current_edge_idx + lookahead + 1)

    distances = [
        float(np.linalg.norm(path[i].xy - pos_xy))
        for i in range(start, end)
    ]

    nearest_offset = int(np.argmin(distances))
    return start + nearest_offset

def _astar(graph, start_id, goal_id):
    waypoints = graph.waypoints
    adj = graph.adj

    def h(n):
        return float(np.linalg.norm(waypoints[n] - waypoints[goal_id]))

    open_heap = [(h(start_id), 0.0, start_id)]
    came_from = {}
    g_score = {start_id: 0.0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal_id:
            path_ids = []
            while current in came_from:
                path_ids.append(current)
                current = came_from[current]
            path_ids.append(start_id)
            return path_ids[::-1]

        if g > g_score.get(current, float("inf")):
            continue

        for neighbour, dist in adj[current]:
            tentative = g_score[current] + dist
            if tentative < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative
                came_from[neighbour] = current
                f = tentative + h(neighbour)
                heapq.heappush(open_heap, (f, tentative, neighbour))

    return None  # unreachable -- graph disconnected between these nodes


def plan_path(graph, start_id, goal_id):
    """
    Returns a list of Node objects (with .xy) from start to goal, or raises
    ValueError if unreachable -- callers (aria_navigate.py, replay_vrs.py)
    don't currently handle a None path, so failing loudly here is safer
    than returning None and crashing later on path[0].xy.
    """
    path_ids = _astar(graph, start_id, goal_id)
    if path_ids is None:
        raise ValueError(
            f"No path found between node {start_id} and node {goal_id} -- "
            f"graph may be disconnected."
        )
    return [graph.node(i) for i in path_ids]


def heading_from_matrix(T):
    forward_local = np.array([1.0, 0.0, 0.0])
    forward_world = T[:3, :3] @ forward_local
    return math.degrees(math.atan2(forward_world[1], forward_world[0]))


def reached_waypoint(pos_xy, waypoint_xy):
    dist = float(np.linalg.norm(np.asarray(pos_xy) - np.asarray(waypoint_xy)))
    return dist <= REACHED_WAYPOINT_THRESHOLD_M

def closest_route_edge(pos_xy, path, current_edge_idx=0, lookahead=6):
    """
    Find which upcoming route segment the user is actually closest to.

    Only searches forward from the current edge so navigation progress
    cannot jump backwards.
    """
    if path is None or len(path) < 2:
        return current_edge_idx

    pos = np.asarray(pos_xy, dtype=np.float64)

    start = current_edge_idx
    end = min(len(path) - 1, current_edge_idx + lookahead + 1)

    best_edge = current_edge_idx
    best_dist = float("inf")

    for i in range(start, end):
        a = np.asarray(path[i].xy, dtype=np.float64)
        b = np.asarray(path[i + 1].xy, dtype=np.float64)

        ab = b - a
        denom = float(np.dot(ab, ab))

        if denom < 1e-9:
            continue

        # Projection of user onto this route segment
        t = float(np.dot(pos - a, ab) / denom)
        t = max(0.0, min(1.0, t))

        closest = a + t * ab
        dist = float(np.linalg.norm(pos - closest))

        if dist < best_dist:
            best_dist = dist
            best_edge = i

    return best_edge

def _bearing_deg(a_xy, b_xy):
    dx = b_xy[0] - a_xy[0]
    dy = b_xy[1] - a_xy[1]
    return math.degrees(math.atan2(dy, dx))


def _turn_description(delta_deg):
    """Same thresholds as the standalone CLI tool's turn_description()."""
    d = delta_deg % 360
    if d > 180:
        d -= 360
    if abs(d) < 15:
        return None
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


def instruction_for_edge(path, edge_idx):

    if edge_idx == 0:
        return "Continue straight."

    prev_start = path[edge_idx - 1]
    seg_start = path[edge_idx]
    seg_end = path[edge_idx + 1]

    prev_bearing = _bearing_deg(prev_start.xy, seg_start.xy)
    curr_bearing = _bearing_deg(seg_start.xy, seg_end.xy)

    delta = curr_bearing - prev_bearing

    # normalize to [-180, 180]
    delta = (delta + 180) % 360 - 180

    # Ignore small bends in the graph.
    # They are not meaningful pedestrian turns.
    if abs(delta) < 55:
        return None

    if delta >= 135 or delta <= -135:
        return "Turn around."

    if delta > 55:
        return "Turn left."

    if delta < -55:
        return "Turn right."

    return None

def remaining_path_distance(pos_xy, path, edge_idx):
    """
    Approximate remaining walking distance from the user's current position
    to the final destination along the planned route.
    """
    if path is None or edge_idx >= len(path) - 1:
        return 0.0

    pos_xy = np.asarray(pos_xy, dtype=np.float64)

    # Current position -> end of current edge
    total = float(np.linalg.norm(path[edge_idx + 1].xy - pos_xy))

    # Remaining route edges
    for i in range(edge_idx + 1, len(path) - 1):
        total += float(np.linalg.norm(path[i + 1].xy - path[i].xy))

    return total


# def distance_to_next_turn(path, edge_idx, turn_threshold_deg=35.0):
    """
    Returns (distance_metres, turn_text) for the next meaningful turn.
    Returns (None, None) if the remaining route is essentially straight.
    """
    if path is None or edge_idx >= len(path) - 2:
        return None, None, None

    distance = 0.0

    for i in range(edge_idx, len(path) - 2):
        a = path[i].xy
        b = path[i + 1].xy
        c = path[i + 2].xy

        distance += float(np.linalg.norm(b - a))

        bearing1 = _bearing_deg(a, b)
        bearing2 = _bearing_deg(b, c)

        delta = (bearing2 - bearing1 + 180) % 360 - 180

        if abs(delta) >= turn_threshold_deg:
            if delta > 0:
                return distance, "left", i+1
            else:
                return distance, "right", i+1

    return None, None, None

def distance_to_next_turn(pos_xy, path, edge_idx, turn_threshold_deg=55.0):
    """
    Distance from the USER'S CURRENT POSITION to the next meaningful turn.

    Returns:
        (distance_m, direction, turn_node_index)

    or:
        (None, None, None)
    """

    if path is None or edge_idx >= len(path) - 2:
        return None, None, None

    pos_xy = np.asarray(pos_xy, dtype=np.float64)

    # Distance from CURRENT USER POSITION to the end of current edge.
    distance = float(
        np.linalg.norm(path[edge_idx + 1].xy - pos_xy)
    )

    # Look ahead through subsequent route edges.
    for i in range(edge_idx, len(path) - 2):

        a = path[i].xy
        b = path[i + 1].xy
        c = path[i + 2].xy

        bearing1 = _bearing_deg(a, b)
        bearing2 = _bearing_deg(b, c)

        delta = (bearing2 - bearing1 + 180) % 360 - 180

        if abs(delta) >= turn_threshold_deg:

            if delta > 0:
                return distance, "left", i + 1
            else:
                return distance, "right", i + 1

        # No turn at b, so add NEXT edge.
        distance += float(np.linalg.norm(c - b))

    return None, None, None

def label_near_node(node, labels_by_name, graph=None, threshold_m=NEARBY_LABEL_THRESHOLD_M):
    
    if graph is None:
        for name, node_id in labels_by_name.items():
            if node_id == node.id:
                return name
        return None

    for name, node_id in labels_by_name.items():
        other_xy = graph.waypoints[node_id]
        if float(np.linalg.norm(other_xy - node.xy)) <= threshold_m:
            return name
    return None

def build_route_instructions(path):
    """
    Compress graph edges into human-readable navigation instructions.
    """

    if len(path) < 2:
        return []

    instructions = []

    distance_accum = 0.0

    for i in range(len(path) - 1):

        a = path[i].xy
        b = path[i + 1].xy

        distance_accum += float(np.linalg.norm(b - a))

        turn = None

        if i < len(path) - 2:
            turn = instruction_for_edge(path, i + 1)

        if turn is not None:

            instructions.append(
                f"Walk straight for {round(distance_accum)} metres."
            )

            instructions.append(turn)

            distance_accum = 0.0

    if distance_accum > 0:
        instructions.append(
            f"Walk straight for {round(distance_accum)} metres."
        )

    return instructions