# Kaavalu (ಕಾವಲು, "guard")

Kaavalu turns a Raspberry Pi into an offline counter for public choke points. An approved
pretrained detector, exported to ExecuTorch and quantised to INT8, counts people (or vehicles)
in every frame of a clip. Counts become a green / amber / red occupancy level. No cloud, no
tracking, no identities: only counts.

```bash
python run.py --input organiser_clip.mp4 --classes person --zone zones/gate.json --capacity 40
```

## Requirements matrix

| Official deliverable | Where we prove it |
| --- | --- |
| Process the organiser's 10–20 s clip or 10–20 images with an approved pretrained detector on ExecuTorch + Raspberry Pi | `run.py`, `models/det_int8.pte`, [Model](#model) |
| Detect at least one category and display counts per frame / image | terminal, `results/run_log.txt`, `results/counts.csv` |
| Category + counts + at least two sample detection outputs | `results/samples/sample_1.jpg`, `sample_2.jpg` (+ `hard_case.jpg`) |
| Average inference time per frame, model used, approximate throughput | summary block in `results/run_log.txt`, `results/run_summary.json`, `results/benchmark_report.json` |
| Societal issue, how counting helps, one practical application (≤ 2 min) | `docs/pitch.md`, slides, backup video |
| *Beyond the brief:* count error vs hand count, tiling, energy, occupancy level | `results/evaluation.json`, `results/charts/` |

## Model

- **SSDLite320-MobileNetV3-Large**, torchvision `DEFAULT` COCO weights, unchanged. Swap in the
  organisers' approved model if it differs; the pattern (export the core, decode outside) is the same.
- Exported: backbone + head only. The `.pte` returns raw class logits and box offsets; box
  decoding, softmax and NMS run in NumPy (`kaavalu/decode.py`), matching torchvision's maths
  (checked by `tests/test_torch_parity.py` and `export_detector.py --verify`).
- INT8: PT2E post-training quantisation with the XNNPACK quantizer, per-channel weights,
  calibrated on ~100 real frames. The `.json` next to each `.pte` records size, normalisation,
  precision and calibration sources.

## Setup

**Laptop / Colab (export):**

```bash
pip install -r requirements-export.txt
python export/export_detector.py --calib data/practice --out models --verify data/practice
```

`--calib` takes videos or image folders (use real street / crowd footage, not the organiser clip).
`--verify` checks that the FP32 `.pte` reproduces the full torchvision model's boxes on 5 frames
(pass/fail), and reports how far INT8 drifts.

> Windows: if export fails with `FileNotFoundError` inside `_run_flatc`, point ExecuTorch at its
> bundled flatc: `set FLATC_EXECUTABLE=<venv>\Lib\site-packages\executorch\data\bin\flatc.exe`.

**Pi (run):** copy `models/` over, then

```bash
bash scripts/setup_pi.sh          # venv, dependencies, tests, cooling / power check
source .venv/bin/activate
python run.py --input data/practice/street.mp4
```

**Tests:** `bash scripts/test.sh` (Linux / Pi) or `powershell -File scripts\test.ps1` (Windows).
GitHub Actions runs the same tests on every push (`.github/workflows/tests.yml`).

## Workflow

| Step | Command | Output |
| --- | --- | --- |
| Count a clip | `python run.py --input clip.mp4 [--tiling adaptive] [--zone z.json --capacity 40]` | `results/` |
| Hand count | `python tools/handcount.py --input clip.mp4 --out data/clip_handcount.csv` | click points per frame |
| Accuracy | `python tools/evaluate.py --truth data/clip_handcount.csv --run full=results_off --run tiled=results_adaptive --class person` | `evaluation.json`, `charts/count_per_frame.png` |
| Speed + energy | `python tools/benchmark.py --image frame.jpg --clip clip.mp4 --power results/power.json` | `benchmark_report.json` |
| Charts | `python tools/charts.py --results results --chosen tiled` | `charts/*.png` |

Run each configuration to its own folder (`--out results_off`, `--out results_adaptive`) to compare them.

### Hand count on the day

Split the frames between three people with `--start` / `--stop` (frame positions), each writing
their own CSV, then pass all of them to `evaluate.py --truth a.csv b.csv c.csv`. Use the same
`--input` and `--every` as `run.py` so frame numbers line up.

**Don't tune on the organiser clip.** Pick `score_threshold` and the tiling settings on practice
clips, then run the organiser clip once.

### Energy per frame

Read the USB-C meter at idle, then while each configuration runs (`benchmark.py --hold 20` keeps
each one busy for 20 s and prints its name). Write the readings to `results/power.json`:

```json
{"idle_w": 2.7, "load_w": {"executorch-int8-320-t4": 6.1, "torch-fp32-320-t4": 7.4}}
```

Energy per frame (mJ) = (load W − idle W) × ms per frame.

## Configuration

`config.yaml` holds every default; command-line flags override it.

- `classes`: COCO names, e.g. `[person]` or `[car, bus, truck, motorcycle]`.
- `tiling.mode`: `off`, `on` (2×2 tiles + full frame = 5 model runs) or `adaptive`
  (tile only crowded frames or frames with many small boxes).
- Merging: boxes from all passes are merged by intersection-over-smaller (removes nested copies
  and fragments), and a box that wraps two or more smaller detections is dropped as a "group" box.
  The same clean-up runs on untiled frames (`clean_full_frame: false` gives raw detector output).
- Zone JSON: `{"points": [[x, y], ...], "capacity": 40}`; points in pixels or 0..1. A person
  counts as inside when the bottom-centre of their box (their feet) is inside.
- Occupancy: count in zone ÷ capacity, averaged over 1 s; GREEN < 60%, AMBER 60–85%, RED > 85%.
  A rise of more than 20 points within 3 s gives an early AMBER. For separate images, each image
  is judged on its own. **State the assumed capacity**: a venue sets it as zone area × safe density.

## Layout

```text
kaavalu/
├── run.py                # the one command
├── config.yaml           # classes, threshold, tiling, zone, capacity
├── export/               # export_detector.py (laptop / Colab)
├── models/               # det_fp32.pte, det_int8.pte + .json metadata, anchors_320.npy
├── kaavalu/              # detector, decode, tiling, counting, occupancy, outputs, frames, system
├── tools/                # handcount.py, evaluate.py, benchmark.py, charts.py
├── zones/                # zone polygons
├── scripts/              # setup_pi.sh, test.sh, test.ps1
├── tests/                # pytest: NumPy logic + parity with torchvision
├── data/practice/        # practice clips (not committed) + hand-count CSVs
├── docs/                 # architecture, contributing, security, pitch
└── results/              # run_log.txt, counts.csv, detections.json, samples/, charts/, *.json
```

## Docs

- [Architecture](docs/ARCHITECTURE.md): how a frame becomes a count and a level
- [Contributing](docs/CONTRIBUTING.md): roles, workflow, phases, freeze rule
- [Agents](docs/AGENTS.md): conventions and commands for changing the code
- [Security](docs/SECURITY.md): privacy stance, what is stored, reporting
- [Pitch](docs/pitch.md): the two-minute pitch and where each number comes from

## Known limits

- SSDLite at 320×320 misses small and distant people; tiling recovers some at about 5× the
  compute per tiled frame. Measure the trade-off on the Pi; don't assume it.
- INT8 changes borderline detections. Compare FP32 and INT8 count error with `evaluate.py`
  before choosing.
- `threads` uses ExecuTorch's XNNPACK threadpool; the log says whether it was applied.
