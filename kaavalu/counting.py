"""Per-frame counting, optionally inside a zone polygon."""

import json

import numpy as np


def load_zone(path, frame_size=None):
    """Load a zone JSON: {"points": [[x, y], ...], "capacity": 40, "name": "gate"}.

    Points may be in pixels or normalised to 0..1 (all values <= 1). Normalised
    points are scaled when ``frame_size`` (width, height) is given.
    """
    with open(path, "r", encoding="utf-8") as f:
        zone = json.load(f)
    pts = np.asarray(zone["points"], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("zone 'points' must be a list of at least 3 [x, y] pairs")
    zone["normalised"] = bool(pts.max() <= 1.0)
    zone["points"] = pts
    if frame_size is not None:
        zone = scale_zone(zone, frame_size)
    return zone


def scale_zone(zone, frame_size):
    out = dict(zone)
    if zone.get("normalised"):
        w, h = frame_size
        out["points"] = zone["points"] * np.array([w, h], dtype=np.float32)
        out["normalised"] = False
    return out


def points_in_polygon(points, polygon):
    """Ray-casting point-in-polygon for (N, 2) points against a (M, 2) polygon."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_at)
        j = i
    return inside


def anchor_points(boxes):
    """Bottom-centre of each box: where a person stands, so zone membership follows feet."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def count_frame(dets, class_map, zone=None):
    """Count detections per chosen class, and how many of them stand inside the zone.

    class_map: {label_index: name}. Returns dict with per-class counts,
    'total', 'zone_total' (None without a zone) and a boolean 'in_zone' mask.
    """
    labels = dets["labels"]
    keep = np.isin(labels, list(class_map.keys()))
    per_class = {name: int(np.sum(labels[keep] == lab)) for lab, name in class_map.items()}
    in_zone = np.zeros(len(labels), dtype=bool)
    zone_per_class = None
    if zone is not None and len(labels):
        in_zone = points_in_polygon(anchor_points(dets["boxes"]), zone["points"]) & keep
    if zone is not None:
        zone_per_class = {name: int(np.sum((labels == lab) & in_zone)) for lab, name in class_map.items()}
    return {
        "per_class": per_class,
        "total": int(keep.sum()),
        "zone_per_class": zone_per_class,
        "zone_total": int(in_zone.sum()) if zone is not None else None,
        "keep": keep,
        "in_zone": in_zone,
    }
