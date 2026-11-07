# Architecture

One command, `run.py`, takes a clip or image folder and writes every output to files.

```text
clip / images
   │  kaavalu/frames.py        every Nth frame, original frame index kept
   ▼
detector (.pte, INT8, XNNPACK)  kaavalu/detector.py   full-frame pass
   │      └── tiled pass         kaavalu/tiling.py     same model on 2x2 overlapping tiles, when needed
   ▼
decode + NMS                     kaavalu/decode.py     NumPy port of torchvision's SSD post-processing
   ▼
merge + clean-up                 kaavalu/tiling.py     drop group boxes, de-duplicate by overlap of the smaller box
   ▼
count per class, in zone         kaavalu/counting.py   bottom-centre of each box tested against the zone polygon
   ▼
occupancy level                  kaavalu/occupancy.py  count / capacity, 1 s smoothing, GREEN / AMBER / RED, fast-rise flag
   ▼
outputs                          kaavalu/outputs.py    terminal, counts.csv, detections.json, frames, samples, timeline chart
```

## Why the model is split

The `.pte` holds only the backbone and head (`kaavalu/model_core.py`). Box decoding and NMS
use data-dependent shapes that export poorly, so they run in NumPy on the Pi. This keeps
the `.pte` small, lets the PyTorch baseline share exactly the same post-processing, and means
the Pi needs no torchvision at runtime.

Correctness is checked twice: `tests/test_torch_parity.py` compares `decode.postprocess` with
torchvision's own `postprocess_detections`, and `export/export_detector.py --verify` compares the
exported FP32 `.pte` with the full torchvision model on real frames.

## Model files

Each `.pte` has a `.json` next to it with input size, normalisation, precision, anchors file
and calibration sources. `Detector` reads it, so a different input size or model only needs a
new export.

## Tiling

At 320×320, distant people shrink to a few pixels. Tiling runs the same model on four
overlapping crops plus the full frame (5 runs). Three steps keep counts honest:

1. Tile boxes touching an interior tile edge are dropped; the overlap means a neighbouring
   tile or the full frame sees that object whole.
2. A box that wraps two or more much smaller boxes of the same class is dropped as a group box.
3. Remaining boxes are de-duplicated with greedy NMS on intersection-over-smaller.

`adaptive` mode tiles only frames where the full-frame pass sees a crowd, many small boxes or
nothing at all. The rules are in `config.yaml` under `tiling`.

## Tools

| Tool | Reads | Writes |
| --- | --- | --- |
| `tools/handcount.py` | the clip | hand-count CSV: click points per frame |
| `tools/evaluate.py` | hand-count CSVs, `detections.json` of each run | `evaluation.json`, `count_per_frame.png` |
| `tools/benchmark.py` | `.pte` files, a frame, optional clip and power readings | `benchmark_report.json` |
| `tools/charts.py` | `benchmark_report.json`, `evaluation.json` | `latency_by_config.png`, `tiling_cost.png`, `speed_vs_accuracy.png` |
