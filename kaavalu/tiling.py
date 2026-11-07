"""Tiled inference: run the same detector on overlapping crops to recover small objects.

At a 320 x 320 model input, distant people shrink to a few pixels. Running the
same approved model on zoomed-in tiles gives them more pixels. Boxes from all
passes are shifted back to frame coordinates and de-duplicated with NMS.
"""

import numpy as np

from .decode import batched_nms, concat_detections, empty_detections, select


def tile_grid(width, height, rows=2, cols=2, overlap=0.2):
    """Yield (x0, y0, x1, y1) tiles that cover the frame with the given fractional overlap."""
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    tw = width / (cols - (cols - 1) * overlap)
    th = height / (rows - (rows - 1) * overlap)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * tw * (1 - overlap)))
            y0 = int(round(r * th * (1 - overlap)))
            x1 = width if c == cols - 1 else int(round(min(width, x0 + tw)))
            y1 = height if r == rows - 1 else int(round(min(height, y0 + th)))
            tiles.append((x0, y0, x1, y1))
    return tiles


def drop_edge_cut(dets, tile, frame_size, margin=2.0):
    """Drop tile detections that touch an *interior* tile edge.

    An object cut by a tile border shows up as a partial box that NMS cannot
    match to the full box. Thanks to the overlap, the neighbouring tile (or the
    full-frame pass) sees such objects whole, so the cut copy can be discarded.
    Edges that coincide with the frame border are kept.
    """
    if len(dets["boxes"]) == 0:
        return dets
    x0, y0, x1, y1 = tile
    fw, fh = frame_size
    b = dets["boxes"]
    bad = np.zeros(len(b), dtype=bool)
    if x0 > 0:
        bad |= b[:, 0] <= x0 + margin
    if y0 > 0:
        bad |= b[:, 1] <= y0 + margin
    if x1 < fw:
        bad |= b[:, 2] >= x1 - margin
    if y1 < fh:
        bad |= b[:, 3] >= y1 - margin
    return select(dets, np.nonzero(~bad)[0])


def shift(dets, dx, dy):
    if len(dets["boxes"]) == 0:
        return dets
    out = dict(dets)
    out["boxes"] = dets["boxes"] + np.array([dx, dy, dx, dy], dtype=np.float32)
    return out


def intersection_over_smaller(a, b):
    """(N, M) intersection area / area of the smaller of each pair."""
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    smaller = np.minimum(area_a[:, None], area_b[None, :])
    return np.where(smaller > 0, inter / np.maximum(smaller, 1e-12), 0.0)


def drop_group_boxes(dets, contain=0.85, min_members=2, area_ratio=2.0):
    """Drop boxes that wrap several other detections of the same class.

    Zoomed-in tiles make the detector fire on groups ("three people" as one box).
    A box that almost fully contains two or more much smaller boxes is a group,
    not an extra object.
    """
    b, labels = dets["boxes"], dets["labels"]
    n = len(b)
    if n < min_members + 1:
        return dets
    area = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(b[:, None, :2], b[None, :, :2])
    rb = np.minimum(b[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clip(0).prod(axis=2)
    # inside[i, j]: box j lies (almost) inside box i, and i is much bigger
    inside = (inter / np.maximum(area[None, :], 1e-12) >= contain) & (area[:, None] >= area_ratio * area[None, :])
    inside &= labels[:, None] == labels[None, :]
    np.fill_diagonal(inside, False)
    group = inside.sum(axis=1) >= min_members
    return select(dets, np.nonzero(~group)[0])


def nms_ios(dets, threshold):
    """Greedy per-class NMS on intersection-over-smaller: removes nested and fragment copies."""
    keep_all = []
    for lab in np.unique(dets["labels"]):
        idx = np.nonzero(dets["labels"] == lab)[0]
        order = idx[np.argsort(-dets["scores"][idx], kind="stable")]
        while order.size:
            i = order[0]
            keep_all.append(i)
            if order.size == 1:
                break
            ios = intersection_over_smaller(dets["boxes"][i:i + 1], dets["boxes"][order[1:]])[0]
            order = order[1:][ios < threshold]
    keep = np.asarray(keep_all, dtype=np.int64)
    return select(dets, keep[np.argsort(-dets["scores"][keep], kind="stable")])


def merge(dets, iou_threshold=0.5, metric="ios", drop_groups=True):
    """Merge boxes from the full-frame pass and all tiles into one set.

    metric "ios" (default) suppresses a box when it overlaps a higher-scoring one by
    more than ``iou_threshold`` of the smaller box's area; "iou" is plain NMS.
    """
    if len(dets["boxes"]) == 0:
        return empty_detections()
    if drop_groups:
        dets = drop_group_boxes(dets)
    if metric == "iou":
        return select(dets, batched_nms(dets["boxes"], dets["scores"], dets["labels"], iou_threshold))
    return nms_ios(dets, iou_threshold)


def should_tile(full_dets, frame_size, cfg):
    """Adaptive rule: tile only crowded frames or frames with many small boxes."""
    n = len(full_dets["boxes"])
    if n >= cfg.get("min_count", 15):
        return True
    if n == 0:
        # An empty full-frame pass can mean everything is too small to see.
        return cfg.get("tile_when_empty", True)
    _, fh = frame_size
    heights = full_dets["boxes"][:, 3] - full_dets["boxes"][:, 1]
    small = np.mean(heights < cfg.get("small_height_frac", 0.08) * fh)
    return bool(small >= cfg.get("small_fraction", 0.3))


def detect_tiled(detector, frame, cfg, full_dets=None):
    """Full-frame pass plus tiles. Returns (detections, number_of_model_runs, tiled_bool)."""
    h, w = frame.shape[:2]
    if full_dets is None:
        full_dets = detector.detect(frame)
    mode = cfg.get("mode", "off")
    if mode == "off" or (mode == "adaptive" and not should_tile(full_dets, (w, h), cfg)):
        # The full-frame pass makes group boxes too, so clean it the same way.
        if cfg.get("clean_full_frame", True):
            full_dets = merge(full_dets, cfg.get("merge_threshold", 0.6),
                              cfg.get("merge_metric", "ios"), cfg.get("drop_group_boxes", True))
        return full_dets, 1, False

    parts = [full_dets]
    tiles = tile_grid(w, h, cfg.get("rows", 2), cfg.get("cols", 2), cfg.get("overlap", 0.2))
    for (x0, y0, x1, y1) in tiles:
        d = detector.detect(frame[y0:y1, x0:x1])
        d = shift(d, x0, y0)
        d = drop_edge_cut(d, (x0, y0, x1, y1), (w, h), cfg.get("edge_margin", 2.0))
        parts.append(d)
    merged = merge(concat_detections(parts), cfg.get("merge_threshold", 0.6),
                   cfg.get("merge_metric", "ios"), cfg.get("drop_group_boxes", True))
    return merged, 1 + len(tiles), True
