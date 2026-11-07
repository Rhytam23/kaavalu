"""Draw the slide charts from the result files, so every number traces back to a file.

    python tools/charts.py --results results --chosen tiled

Reads results/benchmark_report.json and/or results/evaluation.json and writes:
    charts/latency_by_config.png    network latency per configuration (+ energy when measured)
    charts/tiling_cost.png          ms per frame and model runs per frame, tiling off / adaptive / on
    charts/speed_vs_accuracy.png    ms per frame vs count error for each evaluated run
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, GREY, ACCENT = "#1f4e8c", "#9aa5b1", "#d9822b"


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def latency_chart(report, path):
    cfgs = report["configs"]
    if not cfgs:
        return False
    names = [c["name"] for c in cfgs]
    ms = [c["net_ms_mean"] for c in cfgs]
    p95 = [c["net_ms_p95"] for c in cfgs]
    best = min(range(len(ms)), key=lambda i: ms[i])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(cfgs) + 1.4), dpi=130)
    ys = range(len(cfgs))[::-1]
    colors = [BLUE if i == best else GREY for i in range(len(cfgs))]
    ax.barh(list(ys), ms, color=colors, height=0.6)
    ax.errorbar(p95, list(ys), fmt="|", color="black", ms=10, label="p95")
    for y, c, m, p in zip(ys, cfgs, ms, p95):
        label = f"{m:.1f} ms"
        if c.get("energy_mj_per_frame") is not None:
            label += f"  |  {c['energy_mj_per_frame']:.0f} mJ/frame"
        ax.text(p + max(p95) * 0.02, y, label, va="center", fontsize=8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, max(p95) * 1.45)
    ax.set_xlabel("network latency per frame (ms), bar = mean, tick = p95")
    ax.set_title(f"Fastest: {names[best]} at {ms[best]:.0f} ms per frame", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def tiling_chart(report, path):
    rows = report.get("tiling")
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=130)
    modes = [r["tiling"] for r in rows]
    ms = [r["avg_total_ms_per_frame"] for r in rows]
    ax.bar(modes, ms, color=[GREY, BLUE, GREY][: len(rows)])
    for i, r in enumerate(rows):
        ax.text(i, ms[i], f"{ms[i]:.0f} ms\n{r['avg_model_runs_per_frame']:.1f} runs", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylabel("end-to-end ms per frame")
    ax.set_ylim(0, max(ms) * 1.3)
    ax.set_title("What tiling costs on the Pi", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def speed_accuracy_chart(evaluation, path, chosen=None):
    runs = {k: v for k, v in evaluation["runs"].items() if v.get("avg_total_ms_per_frame")}
    if len(runs) < 2:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=130)
    for name, r in runs.items():
        is_chosen = name == chosen
        ax.scatter(r["avg_total_ms_per_frame"], r["mae"], s=70 if is_chosen else 40,
                   color=ACCENT if is_chosen else BLUE, zorder=3)
        ax.annotate(name + (" (chosen)" if is_chosen else ""), (r["avg_total_ms_per_frame"], r["mae"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("end-to-end ms per frame (lower is faster)")
    ax.set_ylabel("count error, MAE per frame (lower is better)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title("Speed vs counting accuracy", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--chosen", help="run name to highlight on the speed-vs-accuracy chart")
    args = ap.parse_args()
    charts = os.path.join(args.results, "charts")
    os.makedirs(charts, exist_ok=True)
    made = []

    bpath = os.path.join(args.results, "benchmark_report.json")
    if os.path.exists(bpath):
        with open(bpath, encoding="utf-8") as f:
            report = json.load(f)
        if latency_chart(report, os.path.join(charts, "latency_by_config.png")):
            made.append("latency_by_config.png")
        if tiling_chart(report, os.path.join(charts, "tiling_cost.png")):
            made.append("tiling_cost.png")

    epath = os.path.join(args.results, "evaluation.json")
    if os.path.exists(epath):
        with open(epath, encoding="utf-8") as f:
            evaluation = json.load(f)
        if speed_accuracy_chart(evaluation, os.path.join(charts, "speed_vs_accuracy.png"), args.chosen):
            made.append("speed_vs_accuracy.png")

    print("wrote " + ", ".join(made) if made else "nothing to draw: run benchmark.py / evaluate.py first")


if __name__ == "__main__":
    main()
