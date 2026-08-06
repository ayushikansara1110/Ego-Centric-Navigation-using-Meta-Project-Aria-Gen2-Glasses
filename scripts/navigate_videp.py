import heapq
import json
import math
import pickle

import numpy as np

REACHED_WAYPOINT_THRESHOLD_M = 0.75  # how close counts as "arrived" at a waypoint
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
    seg_start = path[edge_idx]
    seg_end = path[edge_idx + 1]
    dist = float(np.linalg.norm(seg_end.xy - seg_start.xy))
    curr_bearing = _bearing_deg(seg_start.xy, seg_end.xy)

    parts = []
    if edge_idx > 0:
        prev_start = path[edge_idx - 1]
        prev_bearing = _bearing_deg(prev_start.xy, seg_start.xy)
        turn = _turn_description(curr_bearing - prev_bearing)
        if turn:
            parts.append(turn.capitalize())

    if dist > 0.1:
        parts.append(f"Walk straight for {dist:.1f} metres")

    if not parts:
        parts.append("Continue")

    return ". ".join(parts) + "."


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
