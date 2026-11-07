"""Box decoding and NMS in NumPy.

The exported .pte returns only the raw head outputs (class logits and box
offsets). This module reproduces torchvision's SSD post-processing so the Pi
does not need torchvision at runtime:

* ``BoxCoder.decode_single`` with weights (10, 10, 5, 5)
* softmax over classes (SSD uses softmax, not sigmoid)
* per-class score threshold + top-k + NMS

Keep the maths identical to torchvision; tests/test_decode.py checks it
against the real model when torch is installed.
"""

import math

import numpy as np

BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BBOX_XFORM_CLIP = math.log(1000.0 / 16)


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def decode_boxes(offsets, anchors, weights=BOX_WEIGHTS):
    """Decode (N, 4) regression offsets against (N, 4) xyxy anchors -> xyxy boxes."""
    offsets = np.asarray(offsets, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    wx, wy, ww, wh = weights

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = offsets[:, 0] / wx
    dy = offsets[:, 1] / wy
    dw = np.minimum(offsets[:, 2] / ww, BBOX_XFORM_CLIP)
    dh = np.minimum(offsets[:, 3] / wh, BBOX_XFORM_CLIP)

    pcx = dx * widths + ctr_x
    pcy = dy * heights + ctr_y
    pw = np.exp(dw) * widths
    ph = np.exp(dh) * heights

    # torchvision computes the half-size as 0.5 * w (same as w / 2).
    return np.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], axis=1)


def box_iou(a, b):
    """IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes, scores, iou_threshold):
    """Greedy NMS. Returns indices of kept boxes, highest score first."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes, scores, labels, iou_threshold):
    """NMS done independently per label (same result as torchvision.ops.batched_nms)."""
    keep = []
    for lab in np.unique(labels):
        idx = np.nonzero(labels == lab)[0]
        k = nms(boxes[idx], scores[idx], iou_threshold)
        keep.append(idx[k])
    if not keep:
        return np.zeros((0,), dtype=np.int64)
    keep = np.concatenate(keep)
    return keep[np.argsort(-scores[keep], kind="stable")]


def postprocess(cls_logits, bbox_regression, anchors, image_size,
                score_threshold=0.4, class_filter=None, nms_threshold=0.55,
                topk_candidates=300, max_detections=300):
    """Turn raw SSD head outputs into detections in model-input pixel space.

    cls_logits: (N, C) or (1, N, C); bbox_regression: (N, 4) or (1, N, 4).
    image_size: (height, width) of the model input (e.g. (320, 320)).
    class_filter: iterable of label indices to keep (None = all non-background).
    Returns dict with 'boxes' (K, 4) xyxy, 'scores' (K,), 'labels' (K,).
    """
    logits = np.asarray(cls_logits, dtype=np.float32).reshape(-1, np.shape(cls_logits)[-1])
    offsets = np.asarray(bbox_regression, dtype=np.float32).reshape(-1, 4)
    h, w = image_size

    boxes = decode_boxes(offsets, anchors)
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    probs = softmax(logits, axis=-1)

    num_classes = probs.shape[1]
    labels_to_check = range(1, num_classes) if class_filter is None else sorted(class_filter)

    all_boxes, all_scores, all_labels = [], [], []
    for lab in labels_to_check:
        s = probs[:, lab]
        idx = np.nonzero(s > score_threshold)[0]
        if idx.size == 0:
            continue
        if idx.size > topk_candidates:
            idx = idx[np.argsort(-s[idx], kind="stable")[:topk_candidates]]
        all_boxes.append(boxes[idx])
        all_scores.append(s[idx])
        all_labels.append(np.full(idx.size, lab, dtype=np.int64))

    if not all_boxes:
        return empty_detections()

    b = np.concatenate(all_boxes)
    s = np.concatenate(all_scores)
    l = np.concatenate(all_labels)
    keep = batched_nms(b, s, l, nms_threshold)[:max_detections]
    return {"boxes": b[keep], "scores": s[keep], "labels": l[keep]}


def empty_detections():
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }


def concat_detections(dets):
    dets = [d for d in dets if len(d["boxes"])]
    if not dets:
        return empty_detections()
    return {
        "boxes": np.concatenate([d["boxes"] for d in dets]).astype(np.float32),
        "scores": np.concatenate([d["scores"] for d in dets]).astype(np.float32),
        "labels": np.concatenate([d["labels"] for d in dets]).astype(np.int64),
    }


def select(dets, idx):
    return {k: v[idx] for k, v in dets.items()}
