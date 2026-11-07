"""Compare run outputs with a hand count: the accuracy proof.

    python tools/evaluate.py --truth data/organiser_handcount.csv \
        --run full=results_full --run tiled=results_tiled --class person --out results

Each --run is NAME=RESULTS_DIR (a folder written by run.py). Writes
results/evaluation.json and results/charts/count_per_frame.png.

Metrics, on the frames that were hand-counted:
  MAE            mean |predicted - true| per frame
  bias           mean (predicted - true): negative = we under-count
  MAPE           mean |error| / true, over frames with true > 0
  total error %  (sum predicted - sum true) / sum true
  precision / recall   clicks matched one-to-one to boxes that contain them
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_truth(paths):
    truth = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pts = []
                if row.get("points"):
                    for p in row["points"].split(";"):
                        x, y = p.split(":")
                        pts.append((float(x), float(y)))
                truth[int(row["frame"])] = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return truth


def load_run(folder, cls):
    with open(os.path.join(folder, "detections.json"), encoding="utf-8") as f:
        records = json.load(f)
    dets = {}
    for r in records:
        boxes = [b for b in r["boxes"] if cls is None or b["class"] == cls]
        dets[int(r["frame"])] = {
            "boxes": np.asarray([b["box"] for b in boxes], dtype=np.float32).reshape(-1, 4),
            "scores": np.asarray([b["score"] for b in boxes], dtype=np.float32),
        }
    summary = {}
    sp = os.path.join(folder, "run_summary.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            summary = json.load(f)
    return dets, summary


def match(points, boxes, scores):
    """One-to-one matching: each box (highest score first) takes the closest unmatched click inside it."""
    if len(points) == 0 or len(boxes) == 0:
        return 0
    used = np.zeros(len(points), dtype=bool)
    tp = 0
    for i in np.argsort(-scores, kind="stable"):
        x0, y0, x1, y1 = boxes[i]
        inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1) & ~used
        cand = np.nonzero(inside)[0]
        if cand.size == 0:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        j = cand[np.argmin((points[cand, 0] - cx) ** 2 + (points[cand, 1] - cy) ** 2)]
        used[j] = True
        tp += 1
    return tp


def evaluate(truth, dets):
    frames = sorted(f for f in truth if f in dets)
    missing = sorted(f for f in truth if f not in dets)
    per_frame = []
    tp = fp = fn = 0
    for f in frames:
        t = len(truth[f])
        p = len(dets[f]["boxes"])
        m = match(truth[f], dets[f]["boxes"], dets[f]["scores"])
        tp, fp, fn = tp + m, fp + (p - m), fn + (t - m)
        per_frame.append({"frame": f, "true": t, "predicted": p, "error": p - t, "matched": m})
    if not per_frame:
        raise SystemExit("no overlap between hand-counted frames and run frames; same --input/--every?")
    err = np.array([r["error"] for r in per_frame], dtype=np.float64)
    true = np.array([r["true"] for r in per_frame], dtype=np.float64)
    pos = true > 0
    return {
        "frames_evaluated": len(per_frame),
        "frames_missing_from_run": missing,
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mape_pct": float(np.mean(np.abs(err[pos]) / true[pos]) * 100) if pos.any() else None,
        "total_true": int(true.sum()),
        "total_predicted": int(sum(r["predicted"] for r in per_frame)),
        "total_error_pct": float((err.sum() / true.sum()) * 100) if true.sum() else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "per_frame": per_frame,
    }


def chart(path, truth, results, x_is_index=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = sorted(truth)
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=130)
    ax.plot(range(len(frames)), [len(truth[f]) for f in frames], color="black", lw=2.2,
            marker="o", ms=3, label="hand count")
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for k, (name, res) in enumerate(results.items()):
        pf = {r["frame"]: r["predicted"] for r in res["per_frame"]}
        xs = [i for i, f in enumerate(frames) if f in pf]
        ax.plot(xs, [pf[frames[i]] for i in xs], color=palette[k % len(palette)], lw=1.6,
                label=f"{name} (MAE {res['mae']:.2f})")
    ax.set_xlabel("hand-counted frame")
    ax.set_ylabel("count per frame")
    ax.set_title("Detector count vs hand count", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", nargs="+", required=True, help="hand-count CSV(s) from tools/handcount.py")
    ap.add_argument("--run", action="append", required=True, help="NAME=RESULTS_DIR (repeatable)")
    ap.add_argument("--class", dest="cls", default=None, help="class to evaluate (default: all kept boxes)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    truth = load_truth(args.truth)
    results = {}
    for spec in args.run:
        name, _, folder = spec.partition("=")
        if not folder:
            name, folder = os.path.basename(os.path.normpath(spec)), spec
        dets, summary = load_run(folder, args.cls)
        res = evaluate(truth, dets)
        res["run_dir"] = folder
        res["avg_total_ms_per_frame"] = summary.get("avg_total_ms_per_frame")
        res["avg_model_runs_per_frame"] = summary.get("avg_model_runs_per_frame")
        res["precision_mode"] = summary.get("precision")
        res["tiling"] = summary.get("tiling")
        res["score_threshold"] = summary.get("score_threshold")
        results[name] = res

    os.makedirs(args.out, exist_ok=True)
    out = {"truth_files": args.truth, "class": args.cls, "runs": results}
    with open(os.path.join(args.out, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    chart(os.path.join(args.out, "charts", "count_per_frame.png"), truth, results)

    print(f"{'run':<14}{'frames':>7}{'MAE':>8}{'bias':>8}{'MAPE%':>8}{'total%':>8}{'prec':>7}{'recall':>7}{'ms/frame':>10}")
    fmt = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    for name, r in results.items():
        print(f"{name:<14}{r['frames_evaluated']:>7}{fmt(r['mae']):>8}{fmt(r['bias']):>8}"
              f"{fmt(r['mape_pct'], 1):>8}{fmt(r['total_error_pct'], 1):>8}{fmt(r['precision']):>7}"
              f"{fmt(r['recall']):>7}{fmt(r['avg_total_ms_per_frame'], 1):>10}")
    print(f"\nwrote {os.path.join(args.out, 'evaluation.json')} and charts/count_per_frame.png")


if __name__ == "__main__":
    main()
