"""Kaavalu: count objects in every frame of a clip, offline, on a Raspberry Pi.

    python run.py --input organiser_clip.mp4 --classes person --zone zones/gate.json --capacity 40

Writes to results/ (or --out):
    run_log.txt          every terminal line + the summary block
    counts.csv           frame, class, count, zone count, latency, occupancy, level
    detections.json      kept boxes per frame (used by tools/evaluate.py)
    frames/              every annotated frame (unless outputs.save_all_frames is false)
    samples/             sample_1.jpg, sample_2.jpg, hard_case.jpg (auto-picked suggestions)
    charts/occupancy_timeline.png   when a capacity is given
    run_summary.json     model, precision, average ms per frame, FPS, counts per class
"""

import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np
import yaml

from kaavalu import system
from kaavalu.counting import count_frame, load_zone, scale_zone
from kaavalu.detector import Detector
from kaavalu.frames import iter_frames, source_info
from kaavalu.labels import class_ids
from kaavalu.occupancy import OccupancyTracker
from kaavalu.outputs import (CsvWriter, annotate, detections_record, occupancy_timeline,
                             terminal_line, write_json)
from kaavalu.tiling import detect_tiled

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Per-frame object counting with ExecuTorch.")
    ap.add_argument("--input", required=True, help="video file, image folder or single image")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--model", help="path to .pte, or 'torch' for the PyTorch eager baseline")
    ap.add_argument("--classes", nargs="+", help="COCO class names, e.g. person or car bus truck")
    ap.add_argument("--threshold", type=float, help="confidence threshold")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--tiling", choices=["off", "on", "adaptive"])
    ap.add_argument("--zone", help="zone JSON (points in pixels or 0..1)")
    ap.add_argument("--capacity", type=float, help="safe capacity of the zone (people / vehicles)")
    ap.add_argument("--every", type=int, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out", help="output folder (default results/)")
    ap.add_argument("--no-frames", action="store_true", help="don't save every annotated frame")
    ap.add_argument("--quiet", action="store_true", help="don't print per-frame lines")
    return ap.parse_args(argv)


def merged_config(args):
    cfg = load_config(args.config)
    over = {
        "model": args.model, "classes": args.classes, "score_threshold": args.threshold,
        "threads": args.threads, "zone": args.zone, "capacity": args.capacity,
        "every": args.every, "max_frames": args.max_frames,
    }
    for k, v in over.items():
        if v is not None:
            cfg[k] = v
    cfg.setdefault("tiling", {})
    if args.tiling:
        cfg["tiling"]["mode"] = args.tiling
    cfg.setdefault("occupancy", {})
    cfg.setdefault("outputs", {})
    if args.out:
        cfg["outputs"]["dir"] = args.out
    if args.no_frames:
        cfg["outputs"]["save_all_frames"] = False
    return cfg


class Log:
    def __init__(self, path, quiet=False):
        self._f = open(path, "w", encoding="utf-8")
        self.quiet = quiet

    def __call__(self, line, frame_line=False):
        self._f.write(line + "\n")
        if not (frame_line and self.quiet):
            print(line, flush=True)

    def close(self):
        self._f.close()


def pick_samples(records, threshold):
    """Suggest two strong frames and one honest hard case (the team makes the final call)."""
    scored = []
    for r in records:
        scores = np.array([b["score"] for b in r["boxes"]], dtype=np.float32)
        if scores.size == 0:
            continue
        strength = scores.size * float(scores.mean())
        borderline = float(np.mean(scores < threshold + 0.15))
        scored.append((r["frame"], strength, borderline))
    if not scored:
        return [], None
    best = [f for f, _, _ in sorted(scored, key=lambda t: -t[1])[:2]]
    rest = [t for t in scored if t[0] not in best] or scored
    hard = max(rest, key=lambda t: (t[2], -t[1]))[0]
    return best, hard


def main(argv=None):
    args = parse_args(argv)
    cfg = merged_config(args)
    out = cfg["outputs"].get("dir", "results")
    for sub in ("frames", "samples", "charts"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    log = Log(os.path.join(out, "run_log.txt"), quiet=args.quiet)

    class_map = class_ids(cfg.get("classes") or ["person"])
    kind, fps, n_frames = source_info(args.input)
    model = cfg.get("model", "models/det_int8.pte")
    if model != "torch" and not os.path.isabs(model) and not os.path.exists(model):
        model = os.path.join(HERE, model)
    det = Detector(model, backend=cfg.get("backend"), score_threshold=cfg.get("score_threshold", 0.4),
                   class_filter=list(class_map), nms_threshold=cfg.get("nms_threshold", 0.55),
                   threads=cfg.get("threads"))

    zone = load_zone(cfg["zone"]) if cfg.get("zone") else None
    capacity = cfg.get("capacity") or (zone or {}).get("capacity")
    occ_cfg = cfg["occupancy"]
    every = int(cfg.get("every") or 1)
    eff_fps = fps / every if fps else None
    tracker = None
    if capacity:
        tracker = OccupancyTracker(capacity, fps=eff_fps, is_video=(kind == "video"),
                                   smooth_s=occ_cfg.get("smooth_s", 1.0),
                                   amber_at=occ_cfg.get("amber_at", 60), red_at=occ_cfg.get("red_at", 85),
                                   rise_points=occ_cfg.get("rise_points", 20),
                                   rise_window_s=occ_cfg.get("rise_window_s", 3))

    tiling_cfg = cfg["tiling"]
    log(f"Kaavalu | {det.description} | threads {cfg.get('threads')} "
        f"({'applied' if det.threads_applied else 'runtime default'})")
    log(f"input {args.input} ({kind}, {n_frames or '?'} frames"
        f"{f', {fps:.1f} fps' if fps else ''}) | every {every} | classes {', '.join(class_map.values())} "
        f"| threshold {det.score_threshold} | tiling {tiling_cfg.get('mode', 'off')}"
        f"{f' | capacity {capacity:g}' if capacity else ''}")
    log("-" * 78)

    csvw = CsvWriter(os.path.join(out, "counts.csv"))
    records, timeline = [], {"x": [], "raw": [], "smoothed": [], "level": []}
    model_ms, total_ms, runs, tiled_frames = [], [], [], 0
    totals = {name: 0 for name in class_map.values()}
    save_all = cfg["outputs"].get("save_all_frames", True)
    frame_files = {}
    warm = int(cfg["outputs"].get("warmup_runs", 3))
    zone_scaled = None

    for idx, name, frame in iter_frames(args.input, every=every, max_frames=cfg.get("max_frames")):
        h, w = frame.shape[:2]
        if warm:
            for _ in range(warm):
                det.detect(frame)
            warm = 0
        if zone is not None and zone_scaled is None:
            zone_scaled = scale_zone(zone, (w, h))

        before = det.total_model_ms
        t0 = time.perf_counter()
        dets, n_runs, tiled = detect_tiled(det, frame, tiling_cfg)
        counts = count_frame(dets, class_map, zone_scaled)
        elapsed = (time.perf_counter() - t0) * 1000.0
        m_ms = det.total_model_ms - before

        occ = None
        if tracker:
            occ = tracker.update(counts["zone_total"] if zone_scaled is not None else counts["total"])
            timeline["x"].append(idx / fps if fps else idx)
            timeline["raw"].append(occ["occupancy_pct"])
            timeline["smoothed"].append(occ["smoothed_pct"])
            timeline["level"].append(occ["level"])

        model_ms.append(m_ms)
        total_ms.append(elapsed)
        runs.append(n_runs)
        tiled_frames += int(tiled)
        for k, v in counts["per_class"].items():
            totals[k] += v

        log(terminal_line(name, counts, elapsed, occ["level"] if occ else None, tiled), frame_line=True)
        csvw.write(idx, name, counts, m_ms, elapsed, n_runs, tiled, occ)
        records.append(detections_record(idx, name, dets, counts, class_map))

        img = annotate(frame, dets, counts, class_map, zone_scaled, occ, elapsed)
        # Written even with save_all_frames off: samples are picked from these at the
        # end, and keeping hundreds of full frames in RAM would not fit on a Pi.
        path = os.path.join(out, "frames", f"{idx:05d}.jpg")
        cv2.imwrite(path, img)
        frame_files[idx] = path

    csvw.close()
    write_json(os.path.join(out, "detections.json"), records)
    if not records:
        log("no frames read from the input")
        log.close()
        return 1

    # Suggested samples for the submission (check them by eye before submitting).
    best, hard = pick_samples(records, det.score_threshold)
    picks = [(f, f"sample_{i + 1}.jpg") for i, f in enumerate(best)]
    if hard is not None:
        picks.append((hard, "hard_case.jpg"))
    for f, fname in picks:
        shutil.copyfile(frame_files[f], os.path.join(out, "samples", fname))
    if not save_all:
        shutil.rmtree(os.path.join(out, "frames"), ignore_errors=True)

    if tracker and timeline["x"]:
        note = (f"Assumed capacity {capacity:g}. Thresholds and capacity are configurable; "
                f"a venue sets capacity = zone area x safe density.")
        occupancy_timeline(os.path.join(out, "charts", "occupancy_timeline.png"), timeline["x"],
                           timeline["raw"], timeline["smoothed"], occ_cfg.get("amber_at", 60),
                           occ_cfg.get("red_at", 85), capacity,
                           x_label="seconds" if fps else "image", note=note)

    n = len(records)
    mean_model = float(np.mean(model_ms))
    mean_total = float(np.mean(total_ms))
    summary = {
        "model": det.meta.get("model"), "precision": det.meta.get("precision"),
        "backend": det.backend, "input_size": det.size, "threads": cfg.get("threads"),
        "threads_applied": det.threads_applied, "input": args.input, "frames": n, "every": every,
        "classes": list(class_map.values()), "score_threshold": det.score_threshold,
        "tiling": tiling_cfg.get("mode", "off"), "tiled_frames": tiled_frames,
        "avg_model_runs_per_frame": float(np.mean(runs)),
        "avg_model_ms_per_frame": mean_model,
        "avg_total_ms_per_frame": mean_total,
        "p95_total_ms_per_frame": float(np.percentile(total_ms, 95)),
        "throughput_fps": 1000.0 / mean_total if mean_total else None,
        "total_count": totals,
        "avg_count_per_frame": {k: v / n for k, v in totals.items()},
        "capacity": capacity, "samples": {fname: int(f) for f, fname in picks},
        "cpu_temp_c_end": system.cpu_temp_c(), "throttled": system.throttled(),
        "peak_rss_mb": system.peak_rss_mb(), "machine": system.machine(),
        "warmup_runs_excluded": int(cfg["outputs"].get("warmup_runs", 3)),
    }
    write_json(os.path.join(out, "run_summary.json"), summary)

    log("-" * 78)
    log(f"model          {summary['model']} | {summary['precision']} | {det.backend} | {det.size}x{det.size}")
    log(f"frames         {n} (every {every}), tiled {tiled_frames}, "
        f"{summary['avg_model_runs_per_frame']:.2f} model runs per frame")
    log(f"avg per frame  {mean_model:.1f} ms in the model, {mean_total:.1f} ms end to end "
        f"(p95 {summary['p95_total_ms_per_frame']:.1f} ms)")
    log(f"throughput     {summary['throughput_fps']:.2f} frames per second")
    for k in totals:
        log(f"count {k:<9} total {totals[k]}, average {summary['avg_count_per_frame'][k]:.2f} per frame")
    if summary["cpu_temp_c_end"] is not None:
        log(f"cpu temp       {summary['cpu_temp_c_end']:.1f} C, throttled {summary['throttled']}")
    log(f"outputs        {os.path.abspath(out)}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
