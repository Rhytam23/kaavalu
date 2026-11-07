"""Fast tests for the NumPy parts (no torch needed)."""

import numpy as np
import pytest

from kaavalu.counting import count_frame, points_in_polygon
from kaavalu.decode import batched_nms, decode_boxes, nms, postprocess
from kaavalu.labels import COCO_CATEGORIES, class_ids
from kaavalu.occupancy import AMBER, GREEN, RED, OccupancyTracker, level_for
from kaavalu.tiling import detect_tiled, drop_edge_cut, merge, should_tile, tile_grid
from tools.evaluate import evaluate, match


def dets(boxes, scores=None, labels=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    return {
        "boxes": boxes,
        "scores": np.asarray(scores if scores is not None else [0.9] * n, dtype=np.float32),
        "labels": np.asarray(labels if labels is not None else [1] * n, dtype=np.int64),
    }


def test_class_ids():
    assert class_ids(["person", "car"]) == {1: "person", 3: "car"}
    assert COCO_CATEGORIES[8] == "truck"
    with pytest.raises(ValueError):
        class_ids(["unicorn"])


def test_decode_zero_offsets_returns_anchors():
    anchors = np.array([[10, 20, 50, 80], [0, 0, 5, 5]], dtype=np.float32)
    out = decode_boxes(np.zeros((2, 4)), anchors)
    np.testing.assert_allclose(out, anchors, atol=1e-5)


def test_decode_shift_and_scale():
    anchors = np.array([[0, 0, 10, 10]], dtype=np.float32)
    # dx = 1 anchor width (offset 10 because weight 10); dw = log 2 (offset 5 log 2)
    out = decode_boxes(np.array([[10.0, 0.0, 5 * np.log(2), 0.0]]), anchors)
    np.testing.assert_allclose(out, [[5, 0, 25, 10]], atol=1e-4)


def test_nms_suppresses_overlap_keeps_separate():
    b = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    keep = nms(np.array(b, np.float32), np.array([0.9, 0.8, 0.7], np.float32), 0.5)
    assert list(keep) == [0, 2]


def test_batched_nms_is_per_class():
    b = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32)
    keep = batched_nms(b, np.array([0.9, 0.8], np.float32), np.array([1, 3]), 0.5)
    assert sorted(keep) == [0, 1]


def test_postprocess_threshold_and_filter():
    anchors = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [100, 100, 120, 120]], np.float32)
    logits = np.full((3, 91), -10.0, np.float32)
    logits[0, 1] = 10   # confident person
    logits[1, 3] = 10   # confident car
    logits[2, 0] = 10   # confident background
    out = postprocess(logits, np.zeros((3, 4)), anchors, (320, 320), score_threshold=0.4, class_filter=[1])
    assert list(out["labels"]) == [1]
    np.testing.assert_allclose(out["boxes"], [[0, 0, 10, 10]], atol=1e-4)


def test_tile_grid_covers_frame_with_overlap():
    tiles = tile_grid(1920, 1080, 2, 2, 0.2)
    assert len(tiles) == 4
    assert tiles[0][:2] == (0, 0) and tiles[-1][2:] == (1920, 1080)
    left_tile_right_edge, right_tile_left_edge = tiles[0][2], tiles[1][0]
    assert left_tile_right_edge > right_tile_left_edge  # neighbouring tiles overlap
    assert tile_grid(100, 100, 1, 1, 0.0) == [(0, 0, 100, 100)]


def test_drop_edge_cut_only_interior_edges():
    tile = (0, 0, 100, 100)  # top-left tile of a 200x200 frame
    d = dets([[10, 10, 50, 50], [60, 60, 100, 90], [0, 0, 20, 20]])
    kept = drop_edge_cut(d, tile, (200, 200))
    # middle box touches the right interior edge -> dropped; frame-border box kept
    np.testing.assert_allclose(kept["boxes"], [[10, 10, 50, 50], [0, 0, 20, 20]])


def test_merge_removes_duplicates_across_passes():
    d = dets([[0, 0, 10, 10], [0.5, 0.5, 10, 10], [30, 30, 40, 40]], [0.9, 0.95, 0.8])
    m = merge(d, 0.5)
    assert len(m["boxes"]) == 2


def test_merge_drops_group_box_and_nested_copy():
    people = [[10, 10, 20, 40], [30, 10, 40, 40], [50, 10, 60, 40]]
    group = [[8, 8, 62, 42]]               # one box around all three
    nested = [[11, 11, 19, 30]]            # fragment of the first person
    d = dets(people + group + nested, [0.8, 0.8, 0.8, 0.9, 0.5])
    m = merge(d, 0.6)
    np.testing.assert_allclose(np.sort(m["boxes"][:, 0]), [10, 30, 50])
    plain = merge(d, 0.5, metric="iou", drop_groups=False)
    assert len(plain["boxes"]) > len(m["boxes"])  # plain IoU NMS keeps the group box


def test_should_tile_rules():
    cfg = {"min_count": 3, "small_height_frac": 0.1, "small_fraction": 0.5, "tile_when_empty": False}
    assert should_tile(dets([[0, 0, 5, 5]] * 3), (100, 100), cfg)            # crowded
    assert should_tile(dets([[0, 0, 5, 5], [0, 0, 5, 50]]), (100, 100), cfg)  # half small
    assert not should_tile(dets([[0, 0, 50, 50]]), (100, 100), cfg)
    assert not should_tile(dets(np.zeros((0, 4))), (100, 100), cfg)


def test_detect_tiled_finds_small_objects_and_dedupes():
    # Frame with one big object and four small ones. The fake detector only "sees"
    # objects whose side is at least 6% of the crop it gets, like a low-res model.
    frame = np.zeros((400, 400, 3), np.uint8)
    objs = [(40, 40, 150), (300, 60, 14), (60, 320, 14), (320, 320, 14), (200, 250, 14)]
    for x, y, s in objs:
        frame[y:y + s, x:x + s] = 255

    class Det:
        calls = 0

        def detect(self, crop):
            Det.calls += 1
            h, w = crop.shape[:2]
            found = []
            mask = crop[..., 0] > 0
            seen = np.zeros_like(mask)
            for yy, xx in zip(*np.nonzero(mask)):
                if seen[yy, xx]:
                    continue
                x1 = xx
                while x1 < w and mask[yy, x1]:
                    x1 += 1
                y1 = yy
                while y1 < h and mask[y1, xx]:
                    y1 += 1
                seen[yy:y1, xx:x1] = True
                if min(x1 - xx, y1 - yy) >= 0.06 * max(h, w):
                    found.append([xx, yy, x1, y1])
            return dets(found)

    d = Det()
    full, runs, tiled = detect_tiled(d, frame, {"mode": "off"})
    assert (len(full["boxes"]), runs, tiled) == (1, 1, False)
    out, runs, tiled = detect_tiled(d, frame, {"mode": "on", "rows": 2, "cols": 2, "overlap": 0.2})
    assert runs == 5 and tiled
    assert len(out["boxes"]) == len(objs)  # all small ones recovered, big one not duplicated


def test_points_in_polygon_and_zone_count():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    inside = points_in_polygon([[5, 5], [15, 5], [0.1, 9.9]], square)
    assert list(inside) == [True, False, True]

    d = dets([[2, 0, 4, 5], [20, 0, 22, 5], [2, 0, 4, 5]], labels=[1, 1, 3])
    zone = {"points": square}
    c = count_frame(d, {1: "person"}, zone)
    assert c["per_class"] == {"person": 2}
    assert c["zone_total"] == 1  # the car is ignored, the second person is outside


def test_levels_and_smoothing():
    assert level_for(59.9) == GREEN and level_for(60) == AMBER and level_for(85) == AMBER
    assert level_for(85.1) == RED
    t = OccupancyTracker(capacity=10, fps=2, smooth_s=1.0, rise_points=1000)
    assert t.update(10)["level"] == RED
    assert t.update(0)["smoothed_pct"] == pytest.approx(50.0)  # 2-frame average


def test_fast_rise_gives_early_amber():
    t = OccupancyTracker(capacity=100, fps=1, smooth_s=1.0, rise_points=20, rise_window_s=3)
    t.update(10)
    r = t.update(45)  # +35 points in one second, still GREEN by level
    assert r["rising"] and r["level"] == AMBER


def test_images_mode_has_no_trend():
    t = OccupancyTracker(capacity=10, fps=None, is_video=False)
    t.update(0)
    r = t.update(9)
    assert r["smoothed_pct"] == pytest.approx(90.0) and r["level"] == RED and not r["rising"]


def test_match_is_one_to_one():
    pts = np.array([[5, 5], [6, 6], [50, 50]], np.float32)
    boxes = np.array([[0, 0, 10, 10], [40, 40, 60, 60], [100, 100, 110, 110]], np.float32)
    assert match(pts, boxes, np.array([0.9, 0.8, 0.7], np.float32)) == 2


def test_evaluate_metrics():
    truth = {0: np.array([[5, 5], [50, 50]], np.float32), 1: np.zeros((0, 2), np.float32)}
    run = {
        0: {"boxes": np.array([[0, 0, 10, 10]], np.float32), "scores": np.array([0.9], np.float32)},
        1: {"boxes": np.array([[0, 0, 10, 10]], np.float32), "scores": np.array([0.9], np.float32)},
    }
    r = evaluate(truth, run)
    assert r["mae"] == pytest.approx(1.0)
    assert r["bias"] == pytest.approx(0.0)
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5)
