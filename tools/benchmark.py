"""Benchmark matrix on the Pi: PyTorch eager vs ExecuTorch FP32 vs INT8, threads, tiling, energy.

    python tools/benchmark.py --image data/practice/frame.jpg --clip data/practice/street.mp4 \
        --power results/power.json --out results

Each model x threads configuration runs in its own process (clean RAM and thread
settings): 20 warm-up runs, then 200 timed runs. Records mean / p50 / p95
latency, FPS, peak RAM, model file size, CPU temperature and throttle flag.

--clip also times the whole pipeline with tiling off / on / adaptive (via run.py).

Energy: read the USB-C meter at idle and while each configuration runs
(--hold 20 keeps each configuration busy for 20 s so you can read it), then put
the readings in a JSON file and pass it with --power:
    {"idle_w": 2.7, "load_w": {"executorch-int8-320-t4": 6.1, "torch-fp32-320-t4": 7.4}}
Energy per frame (mJ) = (load W - idle W) x seconds per frame x 1000.
Writes results/benchmark_report.json; tools/charts.py draws the charts from it.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def config_name(backend, precision, size, threads):
    return f"{backend}-{precision}-{size}-t{threads}"


def worker(spec):
    import cv2

    from kaavalu import system
    from kaavalu.detector import Detector

    det = Detector(spec["model"], score_threshold=spec.get("threshold", 0.4),
                   class_filter=spec.get("class_filter"), threads=spec["threads"])
    img = cv2.imread(spec["image"]) if spec.get("image") else None
    if img is None:
        img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    x = det.preprocess(img)
    temp_start = system.cpu_temp_c()

    for _ in range(spec["warmup"]):
        det.raw(x)
    net = []
    for _ in range(spec["runs"]):
        det.raw(x)
        net.append(det.model_ms)
    full = []
    for _ in range(max(20, spec["runs"] // 4)):
        t0 = time.perf_counter()
        det.detect(img)
        full.append((time.perf_counter() - t0) * 1000.0)

    if spec.get("hold"):
        name = config_name(det.backend, det.meta.get("precision"), det.size, spec["threads"])
        print(f"  >> read the power meter now: {name} runs for {spec['hold']} s", file=sys.stderr, flush=True)
        end = time.time() + spec["hold"]
        while time.time() < end:
            det.raw(x)

    net = np.asarray(net)
    model_file = spec["model"] if spec["model"] != "torch" else None
    return {
        "name": config_name(det.backend, det.meta.get("precision"), det.size, spec["threads"]),
        "backend": det.backend, "precision": det.meta.get("precision"), "input_size": det.size,
        "threads": spec["threads"], "threads_applied": det.threads_applied,
        "model": det.description,
        "model_file": model_file,
        "model_file_mb": os.path.getsize(model_file) / 1e6 if model_file else None,
        "warmup_runs": spec["warmup"], "timed_runs": spec["runs"],
        "net_ms_mean": float(net.mean()), "net_ms_p50": float(np.percentile(net, 50)),
        "net_ms_p95": float(np.percentile(net, 95)), "net_ms_std": float(net.std()),
        "net_fps": 1000.0 / float(net.mean()),
        "detect_ms_mean": float(np.mean(full)),
        "detect_fps": 1000.0 / float(np.mean(full)),
        "peak_rss_mb": system.peak_rss_mb(),
        "cpu_temp_c_start": temp_start, "cpu_temp_c_end": system.cpu_temp_c(),
        "throttled": system.throttled(),
    }


def run_worker(spec):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)]
    # stderr passes straight through so "read the power meter now" reaches the terminal.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, cwd=ROOT)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"benchmark worker failed for {spec['model']} (exit {proc.returncode}); see errors above")


def tiling_runs(clip, model, threads, out, every, max_frames):
    rows = []
    for mode in ("off", "adaptive", "on"):
        d = os.path.join(out, "bench_tiling", mode)
        cmd = [sys.executable, os.path.join(ROOT, "run.py"), "--input", clip, "--model", model,
               "--threads", str(threads), "--tiling", mode, "--out", d, "--no-frames", "--quiet",
               "--every", str(every)]
        if max_frames:
            cmd += ["--max-frames", str(max_frames)]
        print(f"tiling {mode}: running pipeline on {clip}")
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        with open(os.path.join(d, "run_summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        rows.append({"tiling": mode, "results_dir": d, "frames": s["frames"],
                     "tiled_frames": s["tiled_frames"],
                     "avg_model_runs_per_frame": s["avg_model_runs_per_frame"],
                     "avg_total_ms_per_frame": s["avg_total_ms_per_frame"],
                     "throughput_fps": s["throughput_fps"],
                     "avg_count_per_frame": s["avg_count_per_frame"]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--models", nargs="+", default=["torch", "models/det_fp32.pte", "models/det_int8.pte"],
                    help="'torch' = PyTorch eager baseline; others are .pte files")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--image", help="a real frame to time on (default: random pixels)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--cooldown", type=float, default=10.0, help="seconds to rest between configs")
    ap.add_argument("--hold", type=float, default=0.0, help="keep each config busy N s for meter readings")
    ap.add_argument("--clip", help="also time the full pipeline with tiling off / adaptive / on")
    ap.add_argument("--tiling-model", default="models/det_int8.pte")
    ap.add_argument("--tiling-threads", type=int, default=4)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--power", help="JSON with idle_w and load_w per config name")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.worker:
        print("RESULT " + json.dumps(worker(json.loads(args.worker))), flush=True)
        return

    from kaavalu import system

    models = [m for m in args.models if m == "torch" or os.path.exists(os.path.join(ROOT, m)) or os.path.exists(m)]
    skipped = sorted(set(args.models) - set(models))
    if skipped:
        print(f"skipping missing models: {', '.join(skipped)}")
    image = os.path.abspath(args.image) if args.image else None

    configs = []
    for model in models:
        for t in args.threads:
            spec = {"model": model, "threads": t, "warmup": args.warmup, "runs": args.runs,
                    "image": image, "hold": args.hold}
            print(f"benchmark {model} threads={t} ...", flush=True)
            r = run_worker(spec)
            print(f"  {r['name']:<28} {r['net_ms_mean']:7.1f} ms (p95 {r['net_ms_p95']:.1f})  "
                  f"{r['net_fps']:5.1f} fps  temp {r['cpu_temp_c_end']}  throttled {r['throttled']}")
            configs.append(r)
            if args.cooldown:
                time.sleep(args.cooldown)

    power = None
    if args.power:
        with open(args.power, encoding="utf-8") as f:
            power = json.load(f)
        idle = float(power["idle_w"])
        for r in configs:
            load = power.get("load_w", {}).get(r["name"])
            if load is not None:
                r["load_w"] = float(load)
                r["idle_w"] = idle
                r["energy_mj_per_frame"] = (float(load) - idle) * r["net_ms_mean"]  # W x ms = mJ

    baseline = next((r for r in configs if r["backend"] == "torch" and r["threads"] == max(args.threads)), None)
    if baseline:
        for r in configs:
            r["speedup_vs_torch"] = baseline["net_ms_mean"] / r["net_ms_mean"]

    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": system.machine(),
        "image": image or "random 1280x720", "configs": configs,
        "baseline": baseline["name"] if baseline else None,
        "power_file": args.power, "power": power,
    }
    if args.clip:
        report["tiling"] = tiling_runs(os.path.abspath(args.clip), args.tiling_model, args.tiling_threads,
                                       os.path.abspath(args.out), args.every, args.max_frames)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "benchmark_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'config':<28}{'mean ms':>9}{'p95':>8}{'fps':>7}{'x torch':>9}{'MB':>7}{'RAM MB':>8}{'temp C':>8}{'mJ/frame':>10}")
    for r in configs:
        fmt = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        print(f"{r['name']:<28}{r['net_ms_mean']:>9.1f}{r['net_ms_p95']:>8.1f}{r['net_fps']:>7.1f}"
              f"{fmt(r.get('speedup_vs_torch'), 2):>9}{fmt(r['model_file_mb']):>7}{fmt(r['peak_rss_mb'], 0):>8}"
              f"{fmt(r['cpu_temp_c_end']):>8}{fmt(r.get('energy_mj_per_frame'), 0):>10}")
    for r in report.get("tiling", []):
        print(f"tiling {r['tiling']:<9} {r['avg_total_ms_per_frame']:7.1f} ms/frame, "
              f"{r['avg_model_runs_per_frame']:.2f} runs/frame, {r['tiled_frames']}/{r['frames']} frames tiled")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
