"""Everything the run writes: terminal lines, CSV, detections JSON, annotated frames, charts."""

import csv
import json
import os

import cv2
import numpy as np

LEVEL_BGR = {"GREEN": (60, 170, 60), "AMBER": (0, 170, 255), "RED": (40, 40, 220)}
BOX_BGR = (255, 200, 0)
ZONE_BGR = (255, 255, 255)


def terminal_line(frame_name, counts, ms, level=None, tiled=False):
    parts = [f"{name}: {n}" for name, n in counts["per_class"].items()]
    text = ", ".join(parts)
    if counts["zone_total"] is not None:
        text += f" (zone {counts['zone_total']})"
    line = f"{frame_name:>14} | {text} | {ms:7.1f} ms"
    if tiled:
        line += " | tiled"
    if level:
        line += f" | level {level}"
    return line


class CsvWriter:
    FIELDS = ["frame", "name", "class", "count", "zone_count", "model_ms", "total_ms",
              "model_runs", "tiled", "occupancy_pct", "smoothed_pct", "level"]

    def __init__(self, path):
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()

    def write(self, frame_idx, name, counts, model_ms, total_ms, runs, tiled, occ):
        for cls, n in counts["per_class"].items():
            self._w.writerow({
                "frame": frame_idx, "name": name, "class": cls, "count": n,
                "zone_count": "" if counts["zone_per_class"] is None else counts["zone_per_class"][cls],
                "model_ms": f"{model_ms:.2f}", "total_ms": f"{total_ms:.2f}",
                "model_runs": runs, "tiled": int(tiled),
                "occupancy_pct": "" if occ is None else f"{occ['occupancy_pct']:.1f}",
                "smoothed_pct": "" if occ is None else f"{occ['smoothed_pct']:.1f}",
                "level": "" if occ is None else occ["level"],
            })

    def close(self):
        self._f.close()


def detections_record(frame_idx, name, dets, counts, class_map):
    keep = counts["keep"]
    return {
        "frame": frame_idx,
        "name": name,
        "boxes": [
            {"class": class_map[int(l)], "score": round(float(s), 4),
             "box": [round(float(v), 1) for v in b], "in_zone": bool(z)}
            for b, s, l, z, k in zip(dets["boxes"], dets["scores"], dets["labels"],
                                     counts["in_zone"], keep) if k
        ],
    }


def annotate(frame, dets, counts, class_map, zone=None, occ=None, ms=None, title=None):
    img = frame.copy()
    h, w = img.shape[:2]
    thick = max(1, round(min(h, w) / 400))
    level_col = LEVEL_BGR.get(occ["level"]) if occ else None

    if zone is not None:
        pts = zone["points"].astype(np.int32).reshape(-1, 1, 2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], level_col or ZONE_BGR)
        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
        cv2.polylines(img, [pts], True, level_col or ZONE_BGR, thick + 1)

    for b, l, k, z in zip(dets["boxes"], dets["labels"], counts["keep"], counts["in_zone"]):
        if not k:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        col = (level_col if (z and level_col) else BOX_BGR)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, thick)

    parts = [f"{n} {name}" for name, n in counts["per_class"].items()]
    banner = (title + "  " if title else "") + ", ".join(parts)
    if counts["zone_total"] is not None:
        banner += f" | zone {counts['zone_total']}"
    if occ:
        banner += f" | {occ['smoothed_pct']:.0f}% {occ['level']}"
    if ms is not None:
        banner += f" | {ms:.0f} ms"
    scale = max(0.5, min(h, w) / 700)
    (tw, th), base = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, scale, thick + 1)
    cv2.rectangle(img, (0, 0), (min(w, tw + 20), th + base + 16), level_col or (40, 40, 40), -1)
    cv2.putText(img, banner, (10, th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                thick + 1, cv2.LINE_AA)
    return img


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def occupancy_timeline(path, xs, raw, smoothed, amber_at, red_at, capacity,
                       x_label="frame", note=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=130)
    top = max(110.0, max(list(raw) + [0.0]) * 1.1)
    ax.axhspan(0, amber_at, color="#3c9a3c", alpha=0.10, lw=0)
    ax.axhspan(amber_at, red_at, color="#f0a500", alpha=0.14, lw=0)
    ax.axhspan(red_at, top, color="#d03030", alpha=0.12, lw=0)
    ax.plot(xs, raw, color="#888888", lw=1, alpha=0.7, label="per frame")
    ax.plot(xs, smoothed, color="#1f4e8c", lw=2, label="smoothed")
    ax.set_ylim(0, top)
    ax.set_xlabel(x_label)
    ax.set_ylabel("occupancy (% of capacity)")
    ax.set_title(f"Occupancy vs assumed capacity of {capacity:g}  "
                 f"(GREEN < {amber_at:g}%, AMBER {amber_at:g}-{red_at:g}%, RED > {red_at:g}%)",
                 fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color="#555555")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
